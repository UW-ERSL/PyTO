"""Optimization routines for topology optimization."""
import enum
import numpy as np

SIMP_PENALTY_MIN = 1 # Min Penalization factor for SIMP method
SIMP_PENALTY_MAX = 6.0 # Max Penalization factor for SIMP method

SIMP_PENALTY = 3.0  # Default penalization factor for SIMP method
# For large DOF problems, we encounters numerical issues for smaller values of EVOID_RELATIVE

EVOID_RELATIVE = 1e-8  # Minimum Young's modulus for void elements
KVOID_RELATIVE = 1e-8  # Minimum Conductivity for void elements

RAMP_PENALTY = 5

MASS_PENALTY = 1  # Penalization factor for mass density
MASS_LOW = 0  # Low density threshold for mass penalization


class MaterialModel(enum.Enum):
	SIMP = enum.auto()
	RAMP = enum.auto()
	SIMPPLUS = enum.auto()


def set_SIMP_PENALTY_MAX(value: float) -> None:
	"""Set the maximum SIMP penalty value."""
	global SIMP_PENALTY_MAX
	SIMP_PENALTY_MAX = value

def update_SIMP_PENALTY_for_body_force(fraction_grey: float) -> None:
	"""Update the SIMP penalty value. This is heuristic for problems with body force"""
	global SIMP_PENALTY
	SIMP_PENALTY = SIMP_PENALTY_MIN*(1 - fraction_grey) + SIMP_PENALTY_MAX*fraction_grey


def increment_SIMP_PENALTY(value) -> None:
	"""Update the SIMP penalty value."""
	global SIMP_PENALTY
	SIMP_PENALTY += value
	SIMP_PENALTY = max(min(SIMP_PENALTY, SIMP_PENALTY_MAX),SIMP_PENALTY_MIN)  # Ensure it does not exceed the maximum value

def initialize_SIMP_PENALTY(value = None) -> None:
	"""Initialize the SIMP penalty value."""
	global SIMP_PENALTY
	if value is not None:
		SIMP_PENALTY = value
	else:
		SIMP_PENALTY = 3.0

def get_structural_material_model_scaling(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the Young's modulus based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	Returns: Array of Young's moduli for each element.
	"""
	if material_model is None:
		return x
	elif material_model == MaterialModel.SIMP:
		return (EVOID_RELATIVE + (x**SIMP_PENALTY)*(1 - EVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return (EVOID_RELATIVE + x*(1-EVOID_RELATIVE) / (1 + (1 - x) * RAMP_PENALTY))  # RAMP model
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (EVOID_RELATIVE + (x**SIMP_PENALTY)*(1 - EVOID_RELATIVE))
		y2 = EVOID_RELATIVE*x
		return (y1+y2)/2
	raise ValueError("Invalid material model specified.")	


def get_thermal_material_model_scaling(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the thermal based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	Returns: Array of Young's moduli for each element.
	"""
	if material_model is None:
		return x
	elif material_model == MaterialModel.SIMP:
		return (KVOID_RELATIVE + (x**SIMP_PENALTY)*(1 - KVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return (KVOID_RELATIVE + x*(1-KVOID_RELATIVE) / (1 + (1 - x) * RAMP_PENALTY))  # RAMP model
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (KVOID_RELATIVE + (x**SIMP_PENALTY)*(1 - KVOID_RELATIVE))
		y2 = KVOID_RELATIVE*x
		return (y1+y2)/2
	raise ValueError("Invalid material model specified.")	
	
def get_structural_material_model_sensitivity(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the Young's modulus based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	Returns: Array of Young's moduli for each element.
	"""
	if material_model is None:
		return np.ones_like(x)
	elif material_model == MaterialModel.SIMP:
		return (SIMP_PENALTY*(x**(SIMP_PENALTY-1))*(1 - EVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return ((1-EVOID_RELATIVE) / (1 + (1 - x) * RAMP_PENALTY)**2) * (1 + RAMP_PENALTY * (1 - x))
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (SIMP_PENALTY*(x**(SIMP_PENALTY-1))*(1 - EVOID_RELATIVE))
		y2 = EVOID_RELATIVE
		return (y1+y2)/2
	else:			
		raise ValueError("Invalid material model specified.")	
	
	
def get_thermal_material_model_sensitivity(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the conductivity sensitivity based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	.
	"""
	if material_model is None:
		return np.ones_like(x)
	elif material_model == MaterialModel.SIMP:
		return (SIMP_PENALTY*(x**(SIMP_PENALTY-1))*(1 - KVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return ((1-KVOID_RELATIVE) / (1 + (1 - x) * RAMP_PENALTY)**2) * (1 + RAMP_PENALTY * (1 - x))
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (SIMP_PENALTY*(x**(SIMP_PENALTY-1))*(1 - KVOID_RELATIVE))
		y2 = KVOID_RELATIVE
		return (y1+y2)/2
	else:			
		raise ValueError("Invalid material model specified.")	
	

def get_material_model_rho_scaling(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the Young's modulus based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	Returns: Array of Young's moduli for each element.
	"""
	if material_model is None:
		return x
	elif material_model == MaterialModel.SIMP:
		return (x>MASS_LOW) * (x) + (x <= MASS_LOW) * (x**MASS_PENALTY)/(MASS_LOW**(MASS_PENALTY-1))  # SIMP model  
	raise ValueError("Invalid material model specified.")	
	
def get_material_model_rho_sensitivity(x: np.ndarray, material_model: MaterialModel) -> np.ndarray:
	"""Compute the Young's modulus based on the material model.

	Args:
		x: Array of (num_elems,) containing the element densities.
		material_model: The material model to use (SIMP or SIMP+).

	Returns: Array of Young's moduli for each element.
	"""
	if material_model is None:
		return np.ones_like(x)
	elif material_model == MaterialModel.SIMP:
		return (x > MASS_LOW) + (x <=MASS_LOW) * (MASS_PENALTY*x**(MASS_PENALTY-1))/(MASS_LOW**(MASS_PENALTY-1)) 
	else:			
		raise ValueError("Invalid material model specified.")	
	
