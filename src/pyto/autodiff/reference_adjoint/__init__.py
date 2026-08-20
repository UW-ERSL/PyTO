"""Legacy NumPy discrete-adjoint sensitivities -- test-oracle only.

See legacy_sensitivities.py for the "why" of this package's existence.
"""
from pyto.autodiff.reference_adjoint.legacy_sensitivities import (
    compute_volume_constraint_and_gradient,
    compute_compliance,
    compute_compliance_and_gradient,
    compute_pnorm_stress_and_sensitivity,
    compute_solution_dotproduct_and_gradient,
)

__all__ = [
    "compute_volume_constraint_and_gradient",
    "compute_compliance",
    "compute_compliance_and_gradient",
    "compute_pnorm_stress_and_sensitivity",
    "compute_solution_dotproduct_and_gradient",
]
