"""Compatibility shim: torch_spsolve now lives at pyto.autodiff.sparse_solve.

Solvers is re-exported from pyto.solve.solvers (the single, unified enum
shared with pyto.solve.numpy_backend -- see plan Phase 3) rather than
being defined here.

Kept so the flat `import torch_spsolve` / `from torch_spsolve import ...`
call sites elsewhere in src/ keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.autodiff.sparse_solve import *
from pyto.autodiff.sparse_solve import SparseLinearSolve, Solvers, solve
