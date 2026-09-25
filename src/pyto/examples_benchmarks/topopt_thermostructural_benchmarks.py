import enum
from pyto.examples_benchmarks.hex_thermostructural_examples import *
from pyto.topopt.common import *


# The actual implementations are in topopt_structural_benchmarks.py and topopt_thermal_benchmarks.py
class ThermoStructuralTOExamples(enum.Enum):
	BiClamp = enum.auto()      # Rodrigues & Fernandes Example A with dT = 1 K
	BiClampDT0 = enum.auto()   # same, dT = 0 (purely mechanical)
	BiClampDT4 = enum.auto()   # same, dT = 4 K
	BiClampDT10 = enum.auto()  # same, dT = 10 K
	MBBBeam = enum.auto()      # Ooms et al., steady-state thermal field (their Fig. 6i)
	MBBBeamNoHeat = enum.auto() # same, bottom kept at 20 C: purely mechanical (their Fig. 6a, t_f = 0)


def getThermoStructuralTOProblem(to_problem: ThermoStructuralTOExamples, **kwargs):
	to_params = TOParams()
	biclamp_dT = {ThermoStructuralTOExamples.BiClamp: 1, ThermoStructuralTOExamples.BiClampDT0: 0,
	              ThermoStructuralTOExamples.BiClampDT4: 4, ThermoStructuralTOExamples.BiClampDT10: 10}
	if to_problem in biclamp_dT:
		# See paper: "A material based model for topology optimization of thermoelastic structures",
		# Rodrigues and Fernandes, IJNME 38 (1995), Example A: 72 x 47.7 x 1 cm plate clamped left and right
		# (the clamped columns are non-design, full material), 1000 kgf at the bottom centre, volume 40 %,
		# uniform temperature rise dT of 0, 1, 4 or 10 K. Here the clamped walls are held at T_ref + dT and
		# heat conduction spreads it: with no other thermal boundary condition a connected design sits at the
		# uniform temperature T_ref + dT, as in the paper.
		dT = biclamp_dT[to_problem]
		print(f"Creating Thermo-structural BiClamp problem (dT = {dT} K)...")
		thermostructural_problem = ThermoStructuralExamples.BiClamp 
		kwargs['structural_load'] = 1000 * 9.80665 # 1000 kgf
		to_params.ThermalReferenceTemperature = 23.0
		kwargs['TWall'] = to_params.ThermalReferenceTemperature + dT
		to_params.Comment = f"Thermo-structural BiClamp (Rodrigues & Fernandes Example A), dT = {dT} K"
		to_params.XSymmetry = True
		to_params.ExtrudeZ = True
		to_params.KeepFixedElems = True
		to_params.RelativeFilterRadius = 1.5
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]

	elif to_problem in (ThermoStructuralTOExamples.MBBBeam, ThermoStructuralTOExamples.MBBBeamNoHeat):
		# See paper: "Thermoelastic topology optimization of structural components at elevated temperatures
		# considering transient heat conduction", by Ooms, et al., 2023. Steady-state case (t_f = infinity,
		# their Fig. 6i): concrete, 20 C on top, 800 C on the bottom, F_m = 10 kN on a 10 mm thick half model,
		# volume 40 %. Void is insulation (conductivity 0.03 of solid); conductivity, stiffness and thermal
		# stress coefficient all use SIMP with exponent 3.
		print("Creating Thermo-structural MBB Beam problem...")
		thermostructural_problem = ThermoStructuralExamples.MBBBeam 
		kwargs['Ta'] = 20  # Top temperature
		kwargs['Tf'] = 20 if to_problem == ThermoStructuralTOExamples.MBBBeamNoHeat else 800 # Bottom (fire) temperature
		to_params.ThermalReferenceTemperature = 20.0
		to_params.ConductivityPenalty = 3.0
		to_params.ConductivityVoidRatio = 0.03
		# Their filter radius is 0.04 x the number of elements along the longest side = 9.6 elements of 5 mm = 48 mm
		# (our elements are ~10.8 mm), with their Heaviside projection continuation.
		to_params.RelativeFilterRadius = 4.4
		to_params.HeavisideProjection = True # beta 1 -> 32, doubling every 50 iterations (their Sect. 3.4)
		to_params.MaxIterations = 350
		to_params.Comment = "Thermo-structural MBB Beam (Ooms et al.), " + (
			"no heating (t_f = 0)" if to_problem == ThermoStructuralTOExamples.MBBBeamNoHeat else "steady state")
		to_params.ExtrudeZ = True
		to_params.nDOFDesired = 25000
		to_params.Objective = (TO_QOI.COMPLIANCE, None)
		to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
	else:
		raise ValueError("Invalid Thermo-structural Topology Optimization problem specified.")
	mesh, mat_prop, bcStructural,bcThermal, elem_body_force = getThermoStructuralProblem(thermostructural_problem, **kwargs)

	to_params.ElemsToKeep = None
	if to_params.KeepFixedElems:  # the clamped columns are non-design (full material)
		to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bcStructural, nDOFPerNode=3)

	return mesh, mat_prop, bcStructural,bcThermal, elem_body_force, to_params
