"""Optimization routines for topology optimization."""
import enum
import numpy as np
import torch
from torch import Tensor

# stay CPU-first; flip to "cuda" later if/when you like
_tdev = torch.device("cpu")
torch.set_default_dtype(torch.float64)


# Make these variables private to the module
_SIMP_STRUCTURAL_PENALTY_MIN = 1 # Min Penalization factor for SIMP method
_SIMP_STRUCTURAL_PENALTY_MAX = 6.0 # Max Penalization factor for SIMP method

_SIMP_THERMAL_PENALTY_MIN = 1  # Min Penalization factor
_SIMP_THERMAL_PENALTY_MAX = 3  # Max Penalization factor for SIMP method - thermal


_SIMP_STRUCTURAL_PENALTY = 3.0  # Default penalization factor for SIMP method 
_SIMP_THERMAL_PENALTY = 1  # Default penalization factor for SIMP method - thermal
_SIMP_STRESS_RELAXATION = 0.5  # Relative stress limit for stress constraint
# For large DOF problems, we encounters numerical issues for smaller values of EVOID_RELATIVE

_EVOID_RELATIVE = 1e-8  # Minimum Young's modulus for void elements
_KVOID_RELATIVE = 1e-8  # Minimum Conductivity for void elements

_RAMP_PENALTY = 5 

_MASS_PENALTY = 1  # Penalization factor for mass density
_MASS_LOW = 0  # Low density threshold for mass penalization

_PNORM_EXPONENT = 6  # Exponent for p-norm stress constraint

class MaterialModel(enum.Enum):
	SIMP = enum.auto()
	RAMP = enum.auto()
	SIMPPLUS = enum.auto()



def set_SIMP_STRUCTURAL_PENALTY(value: float) -> None:
	"""Set the maximum SIMP penalty value."""
	global _SIMP_STRUCTURAL_PENALTY
	_SIMP_STRUCTURAL_PENALTY = value

def update_SIMP_STUCTURAL_PENALTY_for_body_force(fraction_grey: float) -> None:
	"""Update the SIMP penalty value. This is heuristic for problems with body force"""
	global _SIMP_STRUCTURAL_PENALTY
	_SIMP_STRUCTURAL_PENALTY = _SIMP_STRUCTURAL_PENALTY_MIN*(1 - fraction_grey) + _SIMP_STRUCTURAL_PENALTY_MAX*fraction_grey


def increment_SIMP_STRUCTURAL_PENALTY(value) -> None:
	"""Update the SIMP penalty value."""
	global _SIMP_STRUCTURAL_PENALTY
	_SIMP_STRUCTURAL_PENALTY += value
	_SIMP_STRUCTURAL_PENALTY = max(min(_SIMP_STRUCTURAL_PENALTY, _SIMP_STRUCTURAL_PENALTY_MAX),_SIMP_STRUCTURAL_PENALTY_MIN)  # Ensure it does not exceed the maximum value
	return _SIMP_STRUCTURAL_PENALTY

def increment_SIMP_THERMAL_PENALTY(value) -> None:
	"""Update the SIMP penalty value."""
	global _SIMP_THERMAL_PENALTY
	_SIMP_THERMAL_PENALTY += value
	_SIMP_THERMAL_PENALTY = max(min(_SIMP_THERMAL_PENALTY, _SIMP_THERMAL_PENALTY_MAX),_SIMP_THERMAL_PENALTY_MIN)  # Ensure it does not exceed the maximum value
	return _SIMP_THERMAL_PENALTY

def initialize_SIMP_STRUCTURAL_PENALTY(value = None) -> None:
	"""Initialize the SIMP penalty value."""
	global _SIMP_STRUCTURAL_PENALTY
	if value is not None:
		_SIMP_STRUCTURAL_PENALTY = value
	else:
		_SIMP_STRUCTURAL_PENALTY = 3.0
	
def initialize_SIMP_THERMAL_PENALTY(value = None) -> None:
	"""Initialize the SIMP penalty value."""
	global _SIMP_THERMAL_PENALTY
	if value is not None:
		_SIMP_THERMAL_PENALTY = value
	else:
		_SIMP_THERMAL_PENALTY = 1.0
		
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
		return (_EVOID_RELATIVE + (x**_SIMP_STRUCTURAL_PENALTY)*(1 - _EVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return (_EVOID_RELATIVE + x*(1-_EVOID_RELATIVE) / (1 + (1 - x) * _RAMP_PENALTY))  # RAMP model
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (_EVOID_RELATIVE + (x**_SIMP_STRUCTURAL_PENALTY)*(1 - _EVOID_RELATIVE))
		y2 = _EVOID_RELATIVE*x
		return (y1+y2)/2
	raise ValueError("Invalid material model specified.")	

def get_structural_material_model_scaling_torch(x: Tensor,
                                                material_model: MaterialModel | None) -> Tensor:
    """Compute the Young's modulus based on the material model.

    Args:
      x: Array of (num_elems,) containing the element densities.
      material_model: The material model to use (SIMP or SIMP+).

    Returns: Array of Young's moduli for each element.
    """
    Evoid = torch.tensor(_EVOID_RELATIVE, device=_tdev, dtype=torch.float64)
    p     = torch.tensor(_SIMP_STRUCTURAL_PENALTY, device=_tdev, dtype=torch.float64)
    ramp  = torch.tensor(_RAMP_PENALTY,     device=_tdev, dtype=torch.float64)

    if material_model is None:
        return x

    if material_model == MaterialModel.SIMP:
        return Evoid + (x**p) * (1.0 - Evoid)

    if material_model == MaterialModel.RAMP:
        return Evoid + x * (1.0 - Evoid) / (1.0 + (1.0 - x) * ramp)

    if material_model == MaterialModel.SIMPPLUS:
        y1 = Evoid + (x**p) * (1.0 - Evoid)
        y2 = Evoid * x
        return 0.5 * (y1 + y2)

    raise ValueError("Invalid material model specified.")


def get_thermal_material_model_scaling_torch(x: Tensor,
                                             material_model: MaterialModel | None,
                                             penalty: float | None = None,
                                             void_ratio: float | None = None) -> Tensor:
    """Compute the thermal conductivity scaling based on the material model.

    Torch-differentiable twin of get_thermal_material_model_scaling below,
    mirroring get_structural_material_model_scaling_torch above (same
    pattern, thermal's own void/penalty constants).

    Args:
      x: Array of (num_elems,) containing the element densities.
      material_model: The material model to use (SIMP or SIMP+).
      penalty: SIMP exponent of the conductivity (default _SIMP_THERMAL_PENALTY). A problem may set its own,
        e.g. Ooms et al. use 3.
      void_ratio: conductivity of void relative to solid (default _KVOID_RELATIVE). A problem may set its own,
        e.g. Ooms et al. model insulation as void with 0.03 of the solid conductivity.

    Returns: Array of conductivity scalings for each element.
    """
    Kvoid = torch.tensor(_KVOID_RELATIVE if void_ratio is None else void_ratio, device=_tdev, dtype=torch.float64)
    p     = torch.tensor(_SIMP_THERMAL_PENALTY if penalty is None else penalty, device=_tdev, dtype=torch.float64)
    ramp  = torch.tensor(_RAMP_PENALTY,     device=_tdev, dtype=torch.float64)

    if material_model is None:
        return x

    if material_model == MaterialModel.SIMP:
        return Kvoid + (x**p) * (1.0 - Kvoid)

    if material_model == MaterialModel.RAMP:
        return Kvoid + x * (1.0 - Kvoid) / (1.0 + (1.0 - x) * ramp)

    if material_model == MaterialModel.SIMPPLUS:
        y1 = Kvoid + (x**p) * (1.0 - Kvoid)
        y2 = Kvoid * x
        return 0.5 * (y1 + y2)

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
		return (_KVOID_RELATIVE + (x**_SIMP_THERMAL_PENALTY)*(1 - _KVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return (_KVOID_RELATIVE + x*(1-_KVOID_RELATIVE) / (1 + (1 - x) * _RAMP_PENALTY))  # RAMP model
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (_KVOID_RELATIVE + (x**_SIMP_THERMAL_PENALTY)*(1 - _KVOID_RELATIVE))
		y2 = _KVOID_RELATIVE*x
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
		return (_SIMP_STRUCTURAL_PENALTY*(x**(_SIMP_STRUCTURAL_PENALTY-1))*(1 - _EVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return ((1-_EVOID_RELATIVE) / (1 + (1 - x) * _RAMP_PENALTY)**2) * (1 + _RAMP_PENALTY * (1 - x))
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (_SIMP_STRUCTURAL_PENALTY*(x**(_SIMP_STRUCTURAL_PENALTY-1))*(1 - _EVOID_RELATIVE))
		y2 = _EVOID_RELATIVE
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
	if material_model == MaterialModel.SIMP:
		return (_SIMP_THERMAL_PENALTY*(x**(_SIMP_THERMAL_PENALTY-1))*(1 - _KVOID_RELATIVE))  # SIMP model
	elif material_model == MaterialModel.RAMP:
		return ((1-_KVOID_RELATIVE) / (1 + (1 - x) * _RAMP_PENALTY)**2) * (1 + _RAMP_PENALTY * (1 - x))
	elif material_model == MaterialModel.SIMPPLUS:
		y1 = (_SIMP_THERMAL_PENALTY*(x**(_SIMP_THERMAL_PENALTY-1))*(1 - _KVOID_RELATIVE))
		y2 = _KVOID_RELATIVE
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
		return (x>_MASS_LOW) * (x) + (x <= _MASS_LOW) * (x**_MASS_PENALTY)/(_MASS_LOW**(_MASS_PENALTY-1))  # SIMP model  
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
		return (x > _MASS_LOW) + (x <=_MASS_LOW) * (_MASS_PENALTY*x**(_MASS_PENALTY-1))/(_MASS_LOW**(_MASS_PENALTY-1)) 
	else:			
		raise ValueError("Invalid material model specified.")	
	
def get_stress_relaxation_factor() -> float:
	"""Get the stress relaxation factor."""
	return _SIMP_STRESS_RELAXATION


def get_stress_relaxation_correction(x) -> float:
	"""Get the stress relaxation sensitivity."""
	return ( (x**_SIMP_STRESS_RELAXATION)).reshape((-1,1))

def get_stress_relaxation_factor_sensitivity(x) -> float:
	"""Get the stress relaxation sensitivity."""
	return  ( _SIMP_STRESS_RELAXATION* (x**(_SIMP_STRESS_RELAXATION-1))).reshape((-1,1))


def get_pNorm_exponent() -> float:
	"""Get the p-norm exponent."""
	return _PNORM_EXPONENT
