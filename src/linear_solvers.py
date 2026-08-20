"""Compatibility shim: linear_solvers now lives at pyto.solve.numpy_backend.

Solvers is re-exported from pyto.solve.solvers (the single, unified enum
shared with pyto.autodiff.sparse_solve -- see plan Phase 3) rather than
being defined here.

Kept so the ~17 flat `import linear_solvers` call sites elsewhere in src/
keep working unmodified during the restructuring (see plan Phase 2-7).
Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.solve.numpy_backend import *
from pyto.solve.numpy_backend import Solvers, Preconditioners, solve, get_preconditioner
