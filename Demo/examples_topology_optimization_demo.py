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
        to_params.KeepFixedElems = True
        to_params.RemoveHangingElems = False
        #to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 100000
        to_params.DesiredVolFraction = 0.75
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
        to_params.AMBuildConstraint = True
        to_params.nDOFDesired = 50000
        to_params.DesiredVolFraction = 0.25
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