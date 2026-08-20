"""The single Solvers enum shared by the NumPy backend (pyto.solve.numpy_backend)
and the torch-autodiff backend (pyto.autodiff.sparse_solve).

Before this module existed, linear_solvers.py and torch_spsolve.py each
defined their own Solvers enum. Same member names, but two distinct classes
-- a value compared or passed between the two backends silently failed
`is`/`==` checks against the "same" solver from the other module. This is
now the one canonical definition; both backends import it from here.
"""
import enum


class Solvers(enum.Enum):
    SPSOLVE = enum.auto()
    PCG = enum.auto()
    PYAMG = enum.auto()
    DPCG = enum.auto()
    PARDISO = enum.auto()
    SPLU = enum.auto()
    PETSC = enum.auto()
