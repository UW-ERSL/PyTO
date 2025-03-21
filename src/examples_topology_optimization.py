import enum
from examples_structural import *
import struct_fea as sfea

class StructuralTOExamples(enum.Enum):
	EdgeCantilever = enum.auto()
	ThreeHoleBracket = enum.auto()
	MBB = enum.auto()
	DistributedLoad = enum.auto()
	Multiload = enum.auto()
	GravityPlate = enum.auto()
	LBracket = enum.auto()
	CentrifugalPlate = enum.auto()
	TorquePlate = enum.auto()
	BliskWithBlade = enum.auto()

class TOParams:
    nDOFDesired = 20000,
    desiredVolFraction = 0.5
 
class TOConstraints:
    XSymmetry = False
    YSymmetry = False
    ZSymmetry = False
    ZAxisAngularSymmetry = 0
    ExtrudeX = False
    ExtrudeY = False
    ExtrudeZ = False
    KeepFixedElems = False
    RemoveHangingElems = False
    AMBuildConstraint = False
    ElemsToKeep = None
    
def find_elements_with_fixedDOF(mesh, bc) -> np.ndarray:
	"""Find all elements that have nodes with fixed degrees of freedom.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with fixed degrees of freedom.
	"""
	fixed_dofs = bc.fixed_dofs
	fixed_nodes = set(fixed_dofs // 3)  # Convert DOFs to node indices
	elements_with_fixed_dofs = []

	for elem in range(mesh.num_elems):
		nodes =mesh.elemArray[elem]
		if any(node in fixed_nodes for node in nodes):
			elements_with_fixed_dofs.append(elem)

	return np.array(elements_with_fixed_dofs)

def getStructuralTOProblem(to_problem: StructuralTOExamples, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_constraints = TOConstraints()
    to_params = TOParams()
    if to_problem == StructuralTOExamples.EdgeCantilever:
        structural_problem = StructuralExamples.EdgeCantilever
        to_constraints.YSymmetry = True
        to_constraints.AMBuildConstraint = True
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.5
    
    elif to_problem == StructuralTOExamples.ThreeHoleBracket:
        structural_problem = StructuralExamples.ThreeHoleBracket
        to_constraints.ZSymmetry = True
        to_constraints.KeepFixedElems = True
        to_params.nDOFDesired = 40000
        to_params.desiredVolFraction = 0.35
  
    elif to_problem == StructuralTOExamples.MBB:
        structural_problem = StructuralExamples.MBB
        to_constraints.XSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.25

    elif to_problem == StructuralTOExamples.DistributedLoad:
        structural_problem = StructuralExamples.DistributedLoad
        to_constraints.XSymmetry = True 
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.25

    elif to_problem == StructuralTOExamples.Multiload:
        structural_problem = StructuralExamples.Multiload
        to_constraints.ZSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.25

    elif to_problem == StructuralTOExamples.GravityPlate:
        structural_problem = StructuralExamples.GravityPlate
        to_constraints.XSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.2
 
    elif to_problem == StructuralTOExamples.LBracket:
        structural_problem = StructuralExamples.LBracket
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.25

    elif to_problem == StructuralTOExamples.CentrifugalPlate:
        structural_problem = StructuralExamples.CentrifugalPlate
        to_constraints.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.5

    elif to_problem == StructuralTOExamples.TorquePlate:
        structural_problem = StructuralExamples.TorquePlate
        to_constraints.ZAxisAngularSymmetry = 6
        to_constraints.ExtrudeZ = True
        to_params.nDOFDesired = 50000
        to_params.desiredVolFraction = 0.5

    elif to_problem == StructuralTOExamples.BliskWithBlade:
        structural_problem = StructuralExamples.BliskWithBlade
        to_constraints.KeepFixedElems = True
        to_constraints.RemoveHangingElems = True
        to_params.nDOFDesired = 100000
        to_params.desiredVolFraction = 0.25

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getStructuralProblem(structural_problem,nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_constraints.ElemsToKeep  = None # default value

    if (to_constraints.KeepFixedElems):
        to_constraints.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc)

    if to_problem == StructuralTOExamples.BliskWithBlade:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.0558
        outerRadius2 = 0.1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_constraints.ElemsToKeep = np.union1d(to_constraints.ElemsToKeep, bladeElements)

    return mesh, mat_prop, bc, elem_body_force, to_constraints, to_params