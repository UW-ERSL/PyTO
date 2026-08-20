"""Compatibility shim: topopt_mma now lives at pyto.topopt.drivers.mma.

Kept so the flat `from topopt_mma import topopt_mma` call sites (PyTOGUI.py,
PyTO_demos.py, other drivers/benchmarks) keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.topopt.drivers.mma import *
from pyto.topopt.drivers.mma import topopt_mma
