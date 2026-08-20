"""Compatibility shim: bound_cond now lives at pyto.core.bc.

Kept so the ~50 flat `import bound_cond` call sites elsewhere in src/ keep
working unmodified during the restructuring (see plan Phase 2-7). Remove
once nothing imports this flat path anymore (Phase 7).
"""
from pyto.core.bc import *
from pyto.core.bc import BC, impose_dirichlet_bc, apply_dirichlet_bc_torch
