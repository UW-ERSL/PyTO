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
	DENSITYOCM = enum.auto()
	PARETO = enum.auto()
	LEVELSET = enum.auto()

class TO_QOI(enum.Enum): # Topology optimization; Various Quantity of Interest
	VOLUME_FRACTION = enum.auto() # Volume fraction
	MASS = enum.auto() # Mass total
	COMPLIANCE = enum.auto()
	PNORM_STRESS = enum.auto()
	MAX_VONMISES_STRESS = enum.auto()
	GVECTOR = enum.auto() # g'* u
	GFUNCTION = enum.auto() # g(u)

class TOParams: # These are the default parameters
    Comment = "" # Comment for the topology optimization problem
    Objective = (TO_QOI.COMPLIANCE,None) # Tuple of objective type and auxiliary scalar/vector/function	
    Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)] # Collection of tuples of constraint type, auxiliary scalar/vector/function, and upper bound
    nDOFDesired = 25000 # Desired number of degrees of freedom in the finite element problem
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
    AMBuildDir = '' # Direction of AM build, '','X','Y','Z'
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
	Hs = np.array(H.sum(1)).squeeze()
	return H, Hs


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

	volConstraint = ((np.mean(x)/volfracUpper) - 1.0)
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
	
def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x,
										  fe_solver,KE,material_model, p=6):
	"""
    Compute von Mises stress and sensitivity with respect to x for p-norm stress.
    """
	# "An efficient 146-line 3D sensitivity analysis code of 
	# stress based topology optimization written in MATLAB"
	# Optimization and Engineering (2022) 23:1733–1757
	# The sensitivity of pnorm von mises stress with respect to x has 2 terms: T1 and T2
	# T1 arises due to the stress relaxation: x**STRESS_RELAXATION
	# T2 arises indirectly via the solution sensitivity via the adjoint
	mesh = fe_solver.mesh
	nelems = mesh.num_elems
	q = 1 # STRESS_RELAXATION factor
	
	E = fe_solver.mat_prop.youngs_modulus 
	nu = fe_solver.mat_prop.poissons_ratio
	D = E / ((1 + nu) * (1 - 2*nu)) * np.array([
		[1-nu, nu, nu, 0, 0, 0],
		[nu, 1-nu, nu, 0, 0, 0],
		[nu, nu, 1-nu, 0, 0, 0],
		[0, 0, 0, (1-2*nu)/2, 0, 0],
		[0, 0, 0, 0, (1-2*nu)/2, 0],
		[0, 0, 0, 0, 0, (1-2*nu)/2]
	])
	gradN = (1 / 8) * np.array([
		[-1, 1, 1, -1, -1, 1, 1, -1],
		[-1, -1, 1, 1, -1, -1, 1, 1],
		[-1, -1, -1, -1, 1, 1, 1, 1]
	])
	# Define the B matrix (strain-displacement matrix) for a hexahedral element at the center (xi=0, eta=0, zeta=0)
	B = np.zeros((6, 24))
	# Vectorized construction of B matrix for all 8 nodes at once
	Bi = np.zeros((6, 3, 8))
	Bi[0, 0, :] = gradN[0, :]
	Bi[1, 1, :] = gradN[1, :]
	Bi[2, 2, :] = gradN[2, :]
	Bi[3, 0, :] = gradN[1, :]
	Bi[3, 1, :] = gradN[0, :]
	Bi[4, 0, :] = gradN[2, :]
	Bi[4, 2, :] = gradN[0, :]
	Bi[5, 1, :] = gradN[2, :]
	Bi[5, 2, :] = gradN[1, :]
	# Vectorized assignment to B
	idx = np.arange(8)
	B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)
	F = D @ B  # shape (6, 24)
	g_elem = np.zeros((nelems, 24))
	vm_elems = np.zeros(nelems)
	T1 = np.zeros(nelems)
	T2 = np.zeros(nelems)

	for e in range(nelems):
		# First compute the stress without relaxation for sensitivity term T1
		stress_elem = fe_solver.stressComponents[e]
		sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
		vm = np.sqrt(0.5*((sigma11 - sigma22)**2 +(sigma22-sigma33)**2 + (sigma33-sigma11)**2) +
                3*(sigma12**2 + sigma13**2 + sigma23**2))

		T1[e] = p*q*(x[e]**(p*q-1)) * vm

		# Now compute the stress with relaxation for sensitivity term T2 and pnorm stress
		stress_elem = (x[e]**q)* fe_solver.stressComponents[e]
		sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
		vm_elems[e] = np.sqrt(0.5*((sigma11 - sigma22)**2 +(sigma22-sigma33)**2 + (sigma33-sigma11)**2) +
                3*(sigma12**2 + sigma13**2 + sigma23**2))

		g_e = ((sigma11 - sigma22) * (F[0] - F[1]) +
    	(sigma11 - sigma33) * (F[0] - F[2]) +
    	(sigma22 - sigma33) * (F[1] - F[2]) +
    	6 * sigma12 * F[3] + 6 * sigma13 * F[4] + 6 * sigma23 * F[5]) / np.sqrt(2)
		g_elem[e] = p * vm_elems[e] ** (p - 2) * g_e
	
	max_vm = np.max(vm_elems)
	# Note that we are using the relaxed von Mises below
	vm_pnorm = np.sum(vm_elems**p)**(1/p)
	T1 *= (1 / p) * (np.sum(vm_elems ** p) ** (1/p - 1) ) 

	# Now compute the rhs of adjoint eqn 
	g = np.zeros(fe_solver.bc.num_dofs)
	for e in range(nelems): # assemble  g vector
		edof = mesh.edofMat[e]
		g[edof] += g_elem[e]
	g *= -(1 / p) * (np.sum(vm_elems ** p) ** (1/p - 1) )

    # Solve the adjoint	
	adjointSol =  linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
	
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	T2 = get_structural_material_model_sensitivity(x,material_model) * ce
	vm_pnorm_sensitivity = T1+ T2

	return vm_pnorm,vm_pnorm_sensitivity,max_vm

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
	
def compute_constraint_and_gradient(to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None,) -> tuple:
	
	nConstraints = len(to_params.Constraints)
	c = np.zeros((nConstraints,1))	
	dc = np.zeros((nConstraints,x.size))
	
	for m in range(nConstraints):
		constraintType  = to_params.Constraints[m][0]	# first entry is the type of constraint		
		constraintUpperLimit = to_params.Constraints[m][2] # third entry is the constraint value	
		if (constraintType == TO_QOI.COMPLIANCE): 
			compliance, compliance_grad = compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)
			complianceConstraint =  (compliance/constraintUpperLimit - 1.0)
			complianceConstraint_gradient =  (compliance_grad/constraintUpperLimit)
			c[m,0],dc[m,:] = complianceConstraint, complianceConstraint_gradient[np.newaxis]
		elif (constraintType == TO_QOI.VOLUME_FRACTION):
			volConstraint, volConstraint_gradient = compute_volume_constraint_and_gradient(x,to_params.Constraints[m][2])
			c[m,0], dc[m,:] = volConstraint, volConstraint_gradient[np.newaxis]
		elif (constraintType == TO_QOI.PNORM_STRESS):
			pNormValue = to_params.Objective[1] or 6 # default p-norm value  of 6
			stressObj, stress_gradient = compute_pnorm_stress_and_sensitivity(sol, x, fe_solver,KE,material_model,pNormValue)
			c[m,0] = (stressObj/constraintUpperLimit - 1.0)
			dc[m,:] = (stress_gradient/constraintUpperLimit)
		elif (constraintType == TO_QOI.MAX_VONMISES_STRESS):
			pNormValue = to_params.Objective[1] or 6 # default p-norm value  of 6
			pnorm_stressObj, pnorm_stress_gradient, max_von_mises = compute_pnorm_stress_and_sensitivity(sol, x, fe_solver,KE,material_model, pNormValue)
			c[m,0] = (max_von_mises/constraintUpperLimit - 1.0)
			dc[m,:] = (pnorm_stress_gradient/constraintUpperLimit)
		else:
			raise NotImplementedError(" constraint is not implemented yet.")
	return c, dc


def compute_objective_and_gradient(to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None) -> tuple:
							
	objectiveType  = to_params.Objective[0]	# first entry is the type of objective
	if (objectiveType == TO_QOI.COMPLIANCE): 
		compliance, compliance_grad = compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)
		return compliance, compliance_grad
	elif (objectiveType == TO_QOI.VOLUME_FRACTION):
		volfracObj = np.mean(x)
		volFrac_gradient = np.ones_like(x) / x.size
		return volfracObj, volFrac_gradient
	elif (objectiveType == TO_QOI.PNORM_STRESS):
		pNormValue = to_params.Objective[1] or 6 # default p-norm value  of 6
		[stressObj, stress_gradient] = compute_pnorm_stress_and_sensitivity(sol, x, fe_solver,KE,material_model,pNormValue)
		return stressObj, stress_gradient
	elif (objectiveType == TO_QOI.GVECTOR):
		g = to_params.Objective[1]
		compliance, compliance_grad = compute_solution_dotproduct_and_gradient(sol, x, fe_solver, KE,material_model,g)
		return compliance, compliance_grad
	else:
		raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")
	

def compute_objective_topological_sensitivity_compliance(to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None):
	
	
	# Compute the compliance independent of objective
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)
	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)
	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")
	compliance = np.sum(materialScaling * ce)		

	# depending on the objective type, compute the topological sensitivity
	objectiveType  = to_params.Objective[0]	# first entry is the type of objective	
	if (objectiveType == TO_QOI.COMPLIANCE): 
		if (nRows == 24): # structural hex
			T = computeStructuralTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio,fe_solver.strainComponents,fe_solver.stressComponents,x)
		elif (nRows == 8): # thermal hex
			T = computeThermalTopologicalSensitivity(fe_solver.mat_prop.thermal_conductivity,fe_solver.strain,x)
		else:
			raise ValueError("Invalid number of rows in element stiffness matrix.")
		obj = compliance
		return obj,T,compliance
	elif (objectiveType == TO_QOI.PNORM_STRESS):
		pNormValue = to_params.Objective[1] or 6
		[stressObj, T] = compute_pnorm_stress_and_TS(sol, x, fe_solver,KE,material_model,pNormValue)
		return stressObj, T, compliance
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
		return obj,T, compliance
	

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


def compute_pnorm_stress_and_TS(sol: np.ndarray, x,
										  fe_solver,KE,material_model, p=6):
	"""
    Compute von Mises stress and topological sensitivity for p-norm stress.
    """
	# "An efficient 146-line 3D sensitivity analysis code of 
	# stress based topology optimization written in MATLAB"
	# Optimization and Engineering (2022) 23:1733–1757
	# The sensitivity of pnorm von mises stress with respect to x has 2 terms: T1 and T2
	# T1 arises due to the stress relaxation: x**STRESS_RELAXATION
	# T2 arises indirectly via the solution sensitivity via the adjoint
	# T1 is small and can be ignored for large p, so we can use the adjoint sensitivity
	# to compute the topological sensitivity
	mesh = fe_solver.mesh
	nelems = mesh.num_elems
	q = 1 # STRESS_RELAXATION factor
	
	E = fe_solver.mat_prop.youngs_modulus 
	nu = fe_solver.mat_prop.poissons_ratio
	D = E / ((1 + nu) * (1 - 2*nu)) * np.array([
		[1-nu, nu, nu, 0, 0, 0],
		[nu, 1-nu, nu, 0, 0, 0],
		[nu, nu, 1-nu, 0, 0, 0],
		[0, 0, 0, (1-2*nu)/2, 0, 0],
		[0, 0, 0, 0, (1-2*nu)/2, 0],
		[0, 0, 0, 0, 0, (1-2*nu)/2]
	])
	gradN = (1 / 8) * np.array([
		[-1, 1, 1, -1, -1, 1, 1, -1],
		[-1, -1, 1, 1, -1, -1, 1, 1],
		[-1, -1, -1, -1, 1, 1, 1, 1]
	])
	# Define the B matrix (strain-displacement matrix) for a hexahedral element at the center (xi=0, eta=0, zeta=0)
	B = np.zeros((6, 24))
	# Vectorized construction of B matrix for all 8 nodes at once
	Bi = np.zeros((6, 3, 8))
	Bi[0, 0, :] = gradN[0, :]
	Bi[1, 1, :] = gradN[1, :]
	Bi[2, 2, :] = gradN[2, :]
	Bi[3, 0, :] = gradN[1, :]
	Bi[3, 1, :] = gradN[0, :]
	Bi[4, 0, :] = gradN[2, :]
	Bi[4, 2, :] = gradN[0, :]
	Bi[5, 1, :] = gradN[2, :]
	Bi[5, 2, :] = gradN[1, :]
	# Vectorized assignment to B
	idx = np.arange(8)
	B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)
	F = D @ B  # shape (6, 24)
	g_elem = np.zeros((nelems, 24))
	vm_elems = np.zeros(nelems)
	

	for e in range(nelems):

		#  compute the stress with relaxation for sensitivity term T2 and pnorm stress
		stress_elem = (x[e]**q)* fe_solver.stressComponents[e]
		sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
		vm_elems[e] = np.sqrt(0.5*((sigma11 - sigma22)**2 +(sigma22-sigma33)**2 + (sigma33-sigma11)**2) +
                3*(sigma12**2 + sigma13**2 + sigma23**2))

		g_e = ((sigma11 - sigma22) * (F[0] - F[1]) +
    	(sigma11 - sigma33) * (F[0] - F[2]) +
    	(sigma22 - sigma33) * (F[1] - F[2]) +
    	6 * sigma12 * F[3] + 6 * sigma13 * F[4] + 6 * sigma23 * F[5]) / np.sqrt(2)
		g_elem[e] = p * vm_elems[e] ** (p - 2) * g_e
	
	# Note that we are using the relaxed von Mises below
	vm_pnorm = np.sum(vm_elems**p)**(1/p)
	

	# Now compute the rhs of adjoint eqn 
	g = np.zeros(fe_solver.bc.num_dofs)
	for e in range(nelems): # assemble  g vector
		edof = mesh.edofMat[e]
		g[edof] += g_elem[e]
	g *= -(1 / p) * (np.sum(vm_elems ** p) ** (1/p - 1) )

    # Solve the adjoint	
	adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
	
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	strain_adj = np.zeros((num_elems, 6))
	for e in range(num_elems):
		edof = dofMat[e]
		u_e = adjointSol[edof]
		strain_adj[e] = np.dot(B, u_e)
	
	T = computeStructuralTopologicalSensitivity(nu,strain_adj,fe_solver.stressComponents,x)
	

	return vm_pnorm,T

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

