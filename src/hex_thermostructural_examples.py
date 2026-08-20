"""Compatibility shim: hex_thermostructural_examples now lives at
pyto.examples_benchmarks.hex_thermostructural_examples.

Kept so the flat `from hex_thermostructural_examples import *` call sites
keep working unmodified during the restructuring (see plan Phase 2-7).
Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.examples_benchmarks.hex_thermostructural_examples import *
