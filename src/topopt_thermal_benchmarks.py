"""Compatibility shim: topopt_thermal_benchmarks now lives at
pyto.examples_benchmarks.topopt_thermal_benchmarks.

Kept so the flat `from topopt_thermal_benchmarks import *` call sites
keep working unmodified during the restructuring (see plan Phase 2-7).
Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.examples_benchmarks.topopt_thermal_benchmarks import *
