"""Compatibility shim: topopt_ocm now lives at pyto.topopt.drivers.oc.

Kept so the flat `from topopt_ocm import topopt_optimality_criteria` call
sites (PyTOGUI.py, PyTO_demos.py, other benchmarks) keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.topopt.drivers.oc import *
from pyto.topopt.drivers.oc import topopt_optimality_criteria, run_topopt_ocm
