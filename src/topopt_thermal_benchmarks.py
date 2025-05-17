import enum
from hex_thermal_examples import *
from topopt_common import *

class ThermalTOExamples(enum.Enum):
	HeatPlate = enum.auto()
	FourCornersThermal = enum.auto()
	BridgeThermal = enum.auto()
	

def getThermalTOProblem(to_problem: ThermalTOExamples,nDOFDesired = None, **kwargs):
    """
    """
    
    to_params = TOParams()
    if to_problem == ThermalTOExamples.HeatPlate:
        thermal_problem = HexThermalExamples.HeatPlate
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
    elif to_problem == ThermalTOExamples.FourCornersThermal:
        thermal_problem = HexThermalExamples.FourCornersThermal
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.75)]
    elif to_problem == ThermalTOExamples.BridgeThermal:
        thermal_problem = HexThermalExamples.BridgeThermal
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getThermalProblem(thermal_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value


    return mesh, mat_prop, bc, elem_body_force, to_params