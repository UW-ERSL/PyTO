"""Compatibility shim: topopt_obj_cons_sensitivities' content split in Phase 4/5.

Torch-native QOI functions and the objective/constraint dispatchers now
live at pyto.autodiff.qoi (the one production sensitivity path). The
hand-derived legacy NumPy adjoint formulas now live at
pyto.autodiff.reference_adjoint (test-oracle only -- no driver imports
them anymore). Topological-sensitivity formulas (Pareto's actual
production algorithm, a different mathematical object from a gradient --
see pyto/topopt/topological_sensitivity.py's docstring) now live at
pyto.topopt.topological_sensitivity. See plan Phase 4/5 for why this file
used to mix all of these.

Kept so the flat `from topopt_obj_cons_sensitivities import *` call sites
in topopt_mma.py / topopt_ocm.py / topopt_pareto.py keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.autodiff.qoi import *
from pyto.autodiff.qoi import (
    compute_compliance_torch,
    compute_volume_constraint_torch,
    compute_mass_torch,
    compute_gvector_torch,
    compute_pnorm_stress_autograd,
    compute_objective_and_gradient,
    compute_constraint_and_gradient,
)
from pyto.autodiff.reference_adjoint import *
from pyto.autodiff.reference_adjoint import (
    compute_volume_constraint_and_gradient,
    compute_compliance,
    compute_compliance_and_gradient,
    compute_pnorm_stress_and_sensitivity,
    compute_solution_dotproduct_and_gradient,
)
from pyto.topopt.topological_sensitivity import (
    compute_objective_topological_sensitivity_compliance,
    computeStructuralTopologicalSensitivity,
    compute_pnorm_stress_and_TS,
    computeThermalTopologicalSensitivity,
)
