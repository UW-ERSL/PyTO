"""Compatibility shim: deflation now lives at pyto.solve.deflation.

Kept so the existing `import deflation` call sites elsewhere in src/ keep
working unmodified during the restructuring (see plan Phase 2-7). Remove
once nothing imports this flat path anymore (Phase 7).
"""
from pyto.solve.deflation import *
from pyto.solve.deflation import DeflationSolver
