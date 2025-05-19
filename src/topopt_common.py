"""Optimization routines for topology optimization."""
import enum
import numpy as np
from topopt_filters import *
import matplotlib.pyplot as plt
import linear_solvers as lin_solv
import hex_mesher
import hex_structural_fea 
import hex_element_stiffness
from topopt_material_model import *
import hex_thermal_fea 
import deflation
import linear_solvers

DIRECT_SOLVER_DOF_CUTOFF = 100000 #  dof limit for direct solver, for greater number of dof, iterative solver is used

class TO_METHODS(enum.Enum):
	DENSITYMMA = enum.auto()
	DENSITYOC = enum.auto()
	PARETO = enum.auto()
	LEVELSET = enum.auto()

class TO_QOI(enum.Enum): # Topology optimization; Various Quantity of Interest
	VOLUME_FRACTION = enum.auto() # Volume fraction
	MASS_FRACTION = enum.auto() # Mass fraction
	COMPLIANCE = enum.auto() # With respect to the initial compliance
	PNORM_STRESS = enum.auto()
	GVECTOR = enum.auto() # g'* u
	GFUNCTION = enum.auto() # g(u)

class TOParams: # These are the default parameters
    Comment = "" # Comment for the topology optimization problem
    Objective = (TO_QOI.COMPLIANCE,None) # Tuple of objective type and auxiliary function/vector	
    Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] # Collection of tuples of constraint type, auxiliary function/vector, and upper bound
    nDOFDesired = 50000 # Desired number of degrees of freedom in the finite element problem
    APPLY_FILTER_TO_SENSITIVITY = True # Apply filter to density
    APPLY_FILTER_TO_DENSITY = False # Apply filter to density
    RelativeFilterRadius = 1.5 #relative to the element size
    XSymmetry = False # Desired symmetry in YZ plane
    YSymmetry = False
    ZSymmetry = False
    XAxisAngularSymmetry = 0 # Desired symmetry sectors about X axis
    YAxisAngularSymmetry = 0
    ZAxisAngularSymmetry = 0
    ExtrudeX = False # Should the design be extrudable in X direction
    ExtrudeY = False
    ExtrudeZ = False
    KeepFixedElems = False # Should the elements with Dirichlet dof be retained?
    RemoveHangingElems = False # Should the hanging elements be removed?
    AMBuildConstraint = False
    ElemsToKeep = None # List of additional elements to retain in the design


def find_elements_with_forces(mesh: hex_mesher.HexMesher, force,nDOFPerNode) -> np.ndarray:
	"""Find all elements that have nodes on which force has been applied.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with applied forces.
	"""
	force_dofs = np.where(force != 0)[0]
	forced_nodes = set(force_dofs // nDOFPerNode)  # Convert DOFs to node indices
	elements_with_forces = []

	for elem in range(mesh.num_elems):
		nodes = mesh.elemArray[elem]
		if any(node in forced_nodes for node in nodes):
			elements_with_forces.append(elem)

	return np.array(elements_with_forces)

    
def find_elements_with_fixedDOF(mesh, bc,nDOFPerNode ) -> np.ndarray:
	"""Find all elements that have nodes with fixed degrees of freedom.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with fixed degrees of freedom.
	"""
	fixed_dofs = bc.fixed_dofs
	fixed_nodes = set(fixed_dofs // nDOFPerNode)  # Convert DOFs to node indices
	elements_with_fixed_dofs = []

	for elem in range(mesh.num_elems):
		nodes =mesh.elemArray[elem]
		if any(node in fixed_nodes for node in nodes):
			elements_with_fixed_dofs.append(elem)

	return np.array(elements_with_fixed_dofs)

def compute_volume_constraint_and_gradient(x: np.ndarray,
											 volfracUpper: float,
											 )-> np.ndarray:
	"""Compute the volume constraint.
	
	Args:
		density: Array of (num_elems,) containing the element densities.
		volfrac: The target volume fraction.
	
	Returns: The volume constraint. The constraint is satisfied when the
		returned value is zero. The constraint is inactive when the returned
		value is negative.
	"""

	volConstraint = (np.mean(x)/volfracUpper) - 1.0
	volConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
	return volConstraint, volConstraint_gradient


def compute_compliance_and_gradient(sol: np.ndarray, x: np.ndarray,
				fe_solver, KE,
				material_model = None) -> np.ndarray:
	"""Compute the  compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)
	
	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
		compliance_grad = -get_structural_material_model_sensitivity(x,material_model) * ce
	
	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)
		compliance_grad = -get_thermal_material_model_sensitivity(x,material_model) * ce
	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")
	
	
	compliance = np.sum(materialScaling * ce)
	return compliance, compliance_grad


def compute_solution_dotproduct_and_gradient(sol: np.ndarray, x,fe_solver,KE,material_model,g: np.ndarray,
				) -> np.ndarray:
	"""Compute the objective g'* sol, and its gradient.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	obj = np.dot(sol, g)
	

	adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
	
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		compliance_grad = get_structural_material_model_sensitivity(x,material_model) * ce
	
	elif (nRows == 8): # thermal hex
		compliance_grad = get_thermal_material_model_sensitivity(x,material_model) * ce
	return obj, compliance_grad

def compute_objective_and_gradient(to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None) -> tuple:
							
	objectiveType  = to_params.Objective[0]	# first entry is the type of objective			
	if (objectiveType == TO_QOI.COMPLIANCE): 
		compliance, compliance_grad = compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)
		return compliance, compliance_grad
	elif (objectiveType == TO_QOI.GVECTOR):
		g = to_params.Objective[1]
		compliance, compliance_grad = compute_solution_dotproduct_and_gradient(sol, x, fe_solver, KE,material_model,g)
		return compliance, compliance_grad
	else:
		raise NotImplementedError(" objective is not implemented yet.")
	

def createFilters(fe_solver: hex_structural_fea.HexStructuralFEA,to_params):
	# Create  filters
	H, Hs = createSmoothingFilter(fe_solver.mesh, rel_filter_radius=to_params.RelativeFilterRadius)
	# Accumulate all other filters
	if to_params.XSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
		H = H*HX
	if to_params.YSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
		H = H*HY
	if to_params.ZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)
		H = H*HZ
	if to_params.XAxisAngularSymmetry > 0:
		HAAX = createXAngularSymmetryFilter(fe_solver.mesh, to_params.XAxisAngularSymmetry)
		H = H*HAAX
	if to_params.YAxisAngularSymmetry > 0:
		HAAY = createYAngularSymmetryFilter(fe_solver.mesh, to_params.YAxisAngularSymmetry)
		H = H*HAAY
	if to_params.ZAxisAngularSymmetry >	0:
		HAZ = createZAngularSymmetryFilter(fe_solver.mesh, to_params.ZAxisAngularSymmetry)
		H = H*HAZ
	if (to_params.ExtrudeY):
		HEY = createYExtrudeFilter(fe_solver.mesh)
		H = H*HEY
	if (to_params.ExtrudeX):
		HEX = createXExtrudeFilter(fe_solver.mesh)
		H = H*HEX
	if (to_params.ExtrudeZ):
		HEZ = createZExtrudeFilter(fe_solver.mesh)
		H = H*HEZ
	if (to_params.AMBuildConstraint):
		HZAM = createAMBuildFilter(fe_solver.mesh)
		H = H*HZAM

	return H, Hs

def computeTopologicalSensitivity(to_params,fe_solver,x):
	fe_solver.postprocess()
	objectiveType  = to_params.Objective[0]	# first entry is the type of objective			
	if (objectiveType == TO_QOI.COMPLIANCE): 
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			T = computeStructuralTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio,fe_solver.strainComponents,fe_solver.stressComponents,x)
		else:
			T = computeThermalTopologicalSensitivity(fe_solver.mat_prop.thermal_conductivity,fe_solver.strain,x)
	elif (objectiveType == TO_QOI.GVECTOR):
		g = to_params.Objective[1]
		obj = np.dot(fe_solver.sol, g)
	
		print("Not implemented yet")
		adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
	return T
	

def computeStructuralTopologicalSensitivity(poissons_ratio,strains,stresses,x):
	stress_tensor = x[:, None, None] * np.array([
		[stresses[:, 0], stresses[:, 3], stresses[:, 4]],
		[stresses[:, 3], stresses[:, 1], stresses[:, 5]],
		[stresses[:, 4], stresses[:, 5], stresses[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
	strain_tensor = np.array([
		[strains[:, 0], strains[:, 3]/2, strains[:, 4]/2],
		[strains[:, 3]/2, strains[:, 1], strains[:, 5]/2],
		[strains[:, 4]/2, strains[:, 5]/2, strains[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
	# Compute topological sensitivity
	trace_stress = np.trace(stress_tensor, axis1=1, axis2=2)
	trace_strain = np.trace(strain_tensor, axis1=1, axis2=2)
	if isinstance(poissons_ratio, list):
		# Handle multiple materials based on element component ID
		
		# This needs to be fixed to handle different nu values
		nu = poissons_ratio[0]
		
		T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1,2)) -
			 (1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)
	else:
		# Single material case
		nu = poissons_ratio
		T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1, 2)) -
			(1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)
	return T


def computeThermalTopologicalSensitivity(conductivity,strains,x):
	# For thermal problems, topological sensitivity is related to conductivity * gradient^2
	# Multiply by density (x) to scale based on material distribution
	if isinstance(conductivity, list):
		# Handle multiple materials
		# Using the first conductivity value for now - this would need to be updated
		# to properly handle multiple materials
		k = conductivity[0]
	else:
		# Single material case
		k = conductivity
		
	T = x * k * np.sum(strains**2, axis=0)

	return T

