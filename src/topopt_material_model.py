"""Compatibility shim: topopt_material_model now lives at pyto.autodiff.material_model.

Kept so the many flat `from topopt_material_model import *` call sites
elsewhere in src/ keep working unmodified during the restructuring (see
plan Phase 2-7). Remove once nothing imports this flat path anymore
(Phase 7).
"""
from pyto.autodiff.material_model import *
# import * does not bring in underscore-prefixed names; re-export explicitly
# since topopt_obj_cons_sensitivities.py imports these by name directly.
from pyto.autodiff.material_model import _EVOID_RELATIVE, _KVOID_RELATIVE, _PNORM_EXPONENT
