import enum
from hex_thermostructural_examples import *
from topopt_common import *


# The actual implementations are in topopt_structural_benchmarks.py and topopt_thermal_benchmarks.py
class ThermoStructuralTOExamples(enum.Enum):
	BiClamp = enum.auto()
	MBBBeam = enum.auto()


def getThermoStructuralTOProblem(to_problem: ThermoStructuralTOExamples, **kwargs):
	to_params = TOParams()
	if to_problem == ThermoStructuralTOExamples.BiClamp:
		print("Creating Thermo-structural BiClamp problem...")
		thermostructural_problem = ThermoStructuralExamples.BiClamp 
		kwargs['structural_load'] = 1e5
		kwargs['TWall'] = 26 # 23 is the reference temperature
		to_params.Comment = "Thermo-structural BiClamp example"
		to_params.XSymmetry = True
		to_params.ExtrudeZ = True
		to_params.RelativeFilterRadius = 2.5
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.2)]

	elif to_problem == ThermoStructuralTOExamples.MBBBeam:
		# See paper: "Compliance‑based topology optimization of structural components
		# subjected to thermo‑mechanical loading", by Ooms, et al., 2023
		print("Creating Thermo-structural MBB Beam problem...")
		thermostructural_problem = ThermoStructuralExamples.MBBBeam 
		kwargs['structural_load'] = 5000
		kwargs['Ta'] = 23  # Ambient temperature
		kwargs['Tf'] = 33 # Base temperature
		to_params.Comment = "Thermo-structural MBB Beam example"
		to_params.ExtrudeZ = True
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
	else:
		raise ValueError("Invalid Thermo-structural Topology Optimization problem specified.")
	mesh, mat_prop, bcStructural,bcThermal, elem_body_force = getThermoStructuralProblem(thermostructural_problem, **kwargs)

	return mesh, mat_prop, bcStructural,bcThermal, elem_body_force, to_params