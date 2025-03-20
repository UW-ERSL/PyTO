import enum
from examples_structural import *


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
	#BliskQuarter = enum.auto()

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

    elif to_problem == StructuralTOExamples.BliskQuarter:
        structural_problem = StructuralExamples.BliskQuarter
        to_constraints.XSymmetry = True
        to_params.nDOFDesired = 500000
        to_params.desiredVolFraction = 0.25  

    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getStructuralProblem(structural_problem,nDOFDesired = to_params.nDOFDesired, **kwargs)
    return mesh, mat_prop, bc, elem_body_force, to_constraints, to_params