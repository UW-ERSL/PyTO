"""Optimization routines for topology optimization."""
import enum
import numpy as np
import hex_element_stiffness

from topopt_filters import *
import matplotlib.pyplot as plt
import linear_solvers as lin_solv
import hex_mesher
import hex_structural_fea 
import deflation

DIRECT_SOLVER_DOF_CUTOFF = 50000 # 10k dof for direct solver

class TO_METHODS(enum.Enum):
	DENSITYMMA = enum.auto()
	DENSITYOC = enum.auto()
	PARETO = enum.auto()
	LEVELSET = enum.auto()



class TOParams: # These are the default parameters
    Comment = "" # Comment for the topology optimization problem
    nDOFDesired = 20000 # Desired number of degrees of freedom in the finite element problem
    DesiredVolFraction = 0.5
    ExactVolumeFraction = False # If True, the volume fraction is exactly met
    RelativeFilterRadius = 1.5 #relative to the element size
    XSymmetry = False
    YSymmetry = False
    ZSymmetry = False
    XAxisAngularSymmetry = 0
    YAxisAngularSymmetry = 0
    ZAxisAngularSymmetry = 0
    ExtrudeX = False
    ExtrudeY = False
    ExtrudeZ = False
    KeepFixedElems = False
    RemoveHangingElems = False
    AMBuildConstraint = False
    ElemsToKeep = None

def find_elements_with_forces(mesh: hex_mesher.HexMesher, force) -> np.ndarray:
	"""Find all elements that have nodes on which force has been applied.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with applied forces.
	"""
	force_dofs = np.where(force != 0)[0]
	forced_nodes = set(force_dofs // 3)  # Convert DOFs to node indices
	elements_with_forces = []

	for elem in range(mesh.num_elems):
		nodes = mesh.elemArray[elem]
		if any(node in forced_nodes for node in nodes):
			elements_with_forces.append(elem)

	return np.array(elements_with_forces)


def volume_fraction_upperlimit(density: np.ndarray,
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
	return (np.mean(density)/volfracUpper) - 1.0

def volume_fraction_lowerlimit(density: np.ndarray,
											 volfracLower: float,
											 )-> np.ndarray:
	"""Compute the volume constraint.
	
	Args:
		density: Array of (num_elems,) containing the element densities.
		volfrac: The target volume fraction.
	
	Returns: The volume constraint. The constraint is satisfied when the
		returned value is zero. The constraint is inactive when the returned
		value is negative.
	"""
	return 1- (np.mean(density)/volfracLower)

def compliance(x: np.ndarray,
				fe_solver: hex_structural_fea.HexStructuralFEA,
						material_model = None,
													) -> np.ndarray:
	"""Compute the structural compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	u = fe_solver.solve(x, material_model)
	return np.einsum('i, i -> ', fe_solver.total_force, u), u


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


def computeTopologicalSensitivity(poissons_ratio,strains,stresses,x):
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

