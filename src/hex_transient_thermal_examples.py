"""Compatibility shim: hex_transient_thermal_examples now lives at
pyto.examples_benchmarks.hex_transient_thermal_examples.

Kept so the flat `from hex_transient_thermal_examples import *` call
sites keep working unmodified during the restructuring (see plan
Phase 2-7). Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.examples_benchmarks.hex_transient_thermal_examples import *
