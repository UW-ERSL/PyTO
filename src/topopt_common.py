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
	STRESS_SAFETY_FACTOR = enum.auto()
	GVECTOR = enum.auto() # g'* u
	GFUNCTION = enum.auto() # g(u)
	COST = enum.auto() # Generic cost function
	MAX_CRITICALITY = enum.auto() # Captures availability of material
	MEAN_CRITICALITY = enum.auto() # Captures availability of material

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
    MaxIterations = 150 # Maximum number of iterations
    PNormExponent = 6 # p-norm exponent for stress constraint/objective
    Enforce_Constraints_MMA = False # Should the constraints be enforced more strongly in GCMMA?
    Eliminate_Hanging_Elements = True # Should the hanging elements be eliminated after optimization?

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
