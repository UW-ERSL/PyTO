import enum
from examples_structural import *
import struct_fea as sfea

class StructuralTOExamples(enum.Enum):
	EdgeCantilever = enum.auto()
	MBB = enum.auto()
	LBracket = enum.auto()
	DistributedLoad = enum.auto()
	Multiload = enum.auto()
	GravityPlate = enum.auto()
	ThreeHoleBracket = enum.auto()
	TorquePlate = enum.auto()
	CentrifugalPlate = enum.auto()
	KnuckleAssembly = enum.auto()
	Table = enum.auto()
	BliskWithBlade = enum.auto()
     
class TOParams:
    Comment = "" # Comment for the topology optimization problem
    nDOFDesired = 20000 # Desired number of degrees of freedom in the finite element problem
    DesiredVolFraction = 0.5
    RelativeFilterRadius = 1.5 #relative to the element size
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
    
    to_params = TOParams()
    if to_problem == StructuralTOExamples.EdgeCantilever:
        structural_problem = StructuralExamples.EdgeCantilever
        to_params.Comment = "Classic TO Problem"
        to_params.YSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.5

    elif to_problem == StructuralTOExamples.ThreeHoleBracket:
        structural_problem = StructuralExamples.ThreeHoleBracket
        to_params.Comment  = "Retain Material Around Holes"
        to_params.ZSymmetry = True
        to_params.KeepFixedElems = True
        to_params.nDOFDesired = 40000
        to_params.DesiredVolFraction = 0.35
    elif to_problem == StructuralTOExamples.MBB:
        structural_problem = StructuralExamples.MBB
        to_params.Comment  = "Classic TO Poblem"
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.25
    elif to_problem == StructuralTOExamples.DistributedLoad:
        structural_problem = StructuralExamples.DistributedLoad
        to_params.Comment  = "Distributed Load"
        to_params.XSymmetry = True 
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.25
    elif to_problem == StructuralTOExamples.Multiload:
        structural_problem = StructuralExamples.Multiload
        to_params.Comment  = "Multiple Loading"
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.25
    elif to_problem == StructuralTOExamples.GravityPlate:
        structural_problem = StructuralExamples.GravityPlate
        to_params.Comment  = "Body Force TO"
        to_params.XSymmetry = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.1
    elif to_problem == StructuralTOExamples.LBracket:
        structural_problem = StructuralExamples.LBracket
        to_params.Comment  = "Classic TO Problem"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.25
    elif to_problem == StructuralTOExamples.CentrifugalPlate:
        structural_problem = StructuralExamples.CentrifugalPlate
        to_params.Comment  = "Centrifigal + Radial Loading"
        to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.5
        to_params.KeepFixedElems = True  # Keep elements that are fixed in the centrifugal plate example
    elif to_problem == StructuralTOExamples.TorquePlate:
        structural_problem = StructuralExamples.TorquePlate
        to_params.Comment  = "Circular Symmetry"
        to_params.ZAxisAngularSymmetry = 6
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.5
    elif to_problem == StructuralTOExamples.BliskWithBlade:
        structural_problem = StructuralExamples.BliskWithBlade
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = True
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.25

    elif to_problem == StructuralTOExamples.KnuckleAssembly:
        structural_problem = StructuralExamples.KnuckleAssembly
        to_params.Comment = "Assembly TO"
        to_params.XSymmetry = True
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.5
    elif to_problem == StructuralTOExamples.Table:
        structural_problem = StructuralExamples.Table
        to_params.Comment = "Large DOF"
        to_params.XSymmetry = True
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 500000
        to_params.DesiredVolFraction = 0.1
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getStructuralProblem(structural_problem,nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc)

    if to_problem == StructuralTOExamples.BliskWithBlade:
        centerPt = [0,0,0]
        axis = [0,0,1]
        outerRadius1 = 0.0558
        outerRadius2 = 0.1
        bladeElements = mesh.get_elems_within_annular_region(centerPt,axis,outerRadius1,outerRadius2)
        to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, bladeElements)

    if to_problem == StructuralTOExamples.KnuckleAssembly:
         to_params.ElemsToKeep = np.where(mesh.elemComponentId == 2)[0]
         #print("Elems to keep", to_params.ElemsToKeep.shape)
    return mesh, mat_prop, bc, elem_body_force, to_params