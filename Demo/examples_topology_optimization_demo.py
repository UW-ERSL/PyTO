import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import enum
#from examples_structural import *
from examples_topology_optimization import *
from examples_structural_demo import *
import struct_fea as sfea


class StructuralTOExamplesDemo(enum.Enum):
	KnuckleAssembly = enum.auto()
	NoseCone = enum.auto()
	NoseConeAnglularSym = enum.auto()
	BasePlate = enum.auto()
	EdgeCantileverDemo = enum.auto()
	BridgeDemo = enum.auto()
	LongBeamDemo = enum.auto()
	SimpleBracketDemo = enum.auto()
	LongBeamTopBottomLoadDemo = enum.auto()
	BasePlateAssembly = enum.auto()
     
     

def find_elements_with_nodes(mesh: mesher.Mesher, nodes_to_find: np.ndarray) -> np.ndarray:
	"""Find all elements that have nodes with fixed degrees of freedom.
	
	Args:
		mesh: The mesh object.
		nodes_to_find: array of nodes to find their corresponding elements..
	
	Returns:
		Array of element indices that have nodes with fixed degrees of freedom.
	"""
	elements_with_fixed_dofs = []

	for elem in range(mesh.num_elems):
		nodes = mesh.elemArray[elem]
		if any(node in nodes_to_find for node in nodes):
			elements_with_fixed_dofs.append(elem)

	return np.array(elements_with_fixed_dofs)

     
def getStructuralTOProblem(to_problem: StructuralTOExamplesDemo, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    if to_problem == StructuralTOExamplesDemo.KnuckleAssembly:
        structural_problem = StructuralExamplesDemo.KnuckleAssembly
        to_params.Comment = "Assembly TO"
        to_params.XSymmetry = True
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.5
    elif to_problem == StructuralTOExamplesDemo.NoseCone:
        structural_problem = StructuralExamplesDemo.NoseCone
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.93
    elif to_problem == StructuralTOExamplesDemo.NoseConeAnglularSym:
        structural_problem = StructuralExamplesDemo.NoseConeAnglularSym
        to_params.Comment  = "Large DOF"
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        #to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.935
    elif to_problem == StructuralTOExamplesDemo.BasePlate:
        structural_problem = StructuralExamplesDemo.BasePlate
        to_params.Comment  = "Large DOF"
        to_params.YSymmetry = True
        #to_params.ZSymmetry = True
        #to_params.KeepFixedElems = True
        #to_params.RemoveHangingElems = True
        #to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.70
    elif to_problem == StructuralTOExamplesDemo.BasePlateAssembly:
        structural_problem = StructuralExamplesDemo.BasePlateAssembly
        to_params.Comment  = "Large DOF"
        to_params.YSymmetry = True
        #to_params.ZSymmetry = True
        #to_params.KeepFixedElems = True
        #to_params.RemoveHangingElems = True
        #to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.70
    elif to_problem == StructuralTOExamplesDemo.EdgeCantileverDemo:
        structural_problem = StructuralExamplesDemo.EdgeCantileverDemo
        to_params.Comment = "Benchmark TO Problem for Design Trial"
        to_params.ZSymmetry = True
        to_params.AMBuildConstraint = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.5
    elif to_problem == StructuralTOExamplesDemo.BridgeDemo:
        structural_problem = StructuralExamplesDemo.BridgeDemo
        to_params.Comment = "TO Problem for Design Trial"
        to_params.XSymmetry = True
        to_params.AMBuildConstraint = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.3
    elif to_problem == StructuralTOExamplesDemo.LongBeamDemo:
        structural_problem = StructuralExamplesDemo.LongBeamDemo
        to_params.Comment = "TO Problem for Design Trial"
        to_params.XSymmetry = True
        to_params.KeepFixedElems = True
        #to_params.RelativeFilterRadius = 3.5 #relative to the element size
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.45
    elif to_problem == StructuralTOExamplesDemo.SimpleBracketDemo:
        structural_problem = StructuralExamplesDemo.SimpleBracketDemo
        to_params.Comment = "TO Problem for Design Trial"
        to_params.KeepFixedElems = True
        to_params.RelativeFilterRadius = 4
        to_params.nDOFDesired = 75000
        to_params.DesiredVolFraction = 0.7    
    elif to_problem == StructuralTOExamplesDemo.LongBeamTopBottomLoadDemo:
        structural_problem = StructuralExamplesDemo.LongBeamTopBottomLoadDemo
        to_params.Comment = "TO Problem for Design Trial"
        to_params.XSymmetry = True
        #to_params.AMBuildConstraint = True
        #to_params.KeepFixedElems = True
        to_params.RelativeFilterRadius = 4 #relative to the element size the Box is 0.012 x 0.04 x 0.02 and with 100000 dofs we get 64 x 32 x 16 mesh. so we can do 8 times element size
        to_params.nDOFDesired = 150000
        to_params.DesiredVolFraction = 0.65
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    
    mesh, mat_prop, bc, elem_body_force = getStructuralProblem(structural_problem,nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc)

    if to_problem == StructuralTOExamplesDemo.LongBeamDemo:
        if (to_params.KeepFixedElems):
            nodesX = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
            nodesY = mesh.getNodesOnBoundingBoxPlane(1,False) # y = yMax plane 
            nodes_to_find = np.union1d(nodesX, nodesY)
            to_params.ElemsToKeep = find_elements_with_nodes(mesh, nodes_to_find)
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