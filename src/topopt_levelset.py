"""Compatibility shim: topopt_levelset now lives at pyto.topopt.drivers.levelset.

Kept so the flat `from topopt_levelset import topopt_levelset` call sites
(PyTOGUI.py, PyTO_demos.py) keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.topopt.drivers.levelset import *
from pyto.topopt.drivers.levelset import topopt_levelset, run_topopt_levelset
