import enum
from hex_thermostructural_examples import *
from topopt_common import *


# The actual implementations are in topopt_structural_benchmarks.py and topopt_thermal_benchmarks.py
class ThermoStructuralTOExamples(enum.Enum):
	BiClamp = enum.auto()


def getThermoStructuralTOProblem(to_problem: ThermoStructuralTOExamples, **kwargs):
	to_params = TOParams()
	if to_problem == ThermoStructuralTOExamples.BiClamp:
		print("Creating Thermo-structural BiClamp problem...")
		thermostructural_problem = ThermoStructuralExamples.BiClamp 
		kwargs['structural_load'] = 1e5
		kwargs['TWall'] = 24 # 23 is the reference temperature
		to_params.Comment = "Thermo-structural BiClamp example"
		to_params.XSymmetry = True
		to_params.ExtrudeZ = True
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.2)]

	mesh, mat_prop, bcStructural,bcThermal, elem_body_force = getThermoStructuralProblem(thermostructural_problem, **kwargs)

	return mesh, mat_prop, bcStructural,bcThermal, elem_body_force, to_params