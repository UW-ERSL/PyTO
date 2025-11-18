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
from dataclasses import dataclass, field
from typing import Tuple
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
	VOLUME = enum.auto() # Volume total
	MASS = enum.auto() # Mass total
	COMPLIANCE = enum.auto()
	PNORM_STRESS = enum.auto()
	MAX_VONMISES_STRESS = enum.auto()
	STRESS_SAFETY_FACTOR = enum.auto()
	GVECTOR = enum.auto() # g'* u
	GFUNCTION = enum.auto() # g(u)
	COST = enum.auto() # Generic cost function
	TEMPERATURE_SAFETY_FACTOR = enum.auto() # Maximum temperature in thermal problem
	MAX_CRITICALITY = enum.auto() # Captures availability of material
	MEAN_CRITICALITY = enum.auto() # Captures availability of material


@dataclass(slots=True) # avoid accidental modification
class TOParams: # These are the default parameters
    Comment: str = "" # Comment for the topology optimization problem
    Objective: Tuple[TO_QOI, None] = (TO_QOI.COMPLIANCE,None) # Tuple of objective type and auxiliary scalar/vector/function	
    Constraints: list[Tuple[TO_QOI, None, float]] = field(
        default_factory=lambda: [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    )# Collection of tuples of constraint type, auxiliary scalar/vector/function, and upper bound
    nDOFDesired: int = 25000 # Desired number of degrees of freedom in the finite element problem
    APPLY_FILTER_TO_SENSITIVITY: bool = True # Apply filter to density
    APPLY_FILTER_TO_DENSITY: bool = False # Apply filter to density
    RelativeFilterRadius: float = 1.5 #relative to the element size
    XSymmetry: bool = False # Desired symmetry in YZ plane
    YSymmetry: bool = False
    ZSymmetry: bool = False
    XAxisAngularSymmetry: int = 0 # Desired symmetry sectors about X axis
    YAxisAngularSymmetry: int = 0
    ZAxisAngularSymmetry: int = 0
    ExtrudeX: bool = False # Should the design be extrudable in X direction
    ExtrudeY: bool = False
    ExtrudeZ: bool = False
    KeepFixedElems: bool = False # Should the elements with Dirichlet dof be retained?
    RemoveHangingElems: bool = False # Should the hanging elements be removed?
    AMBuildDir: str = '' # Direction of AM build, '','X','Y','Z'
    ElemsToKeep: list[int] = None # List of additional elements to retain in the design
    MaxIterations: int = 150 # Maximum number of iterations
    PNormExponent: int = 6 # p-norm exponent for stress constraint/objective
    Enforce_Constraints_MMA: bool = False # Should the constraints be enforced more strongly in GCMMA?
    Eliminate_Hanging_Elements: bool = False # Should the hanging elements be eliminated after optimization?
    MaterialsExcelFile: str = '' # Path to the Excel file containing material properties for MMTO problems

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
