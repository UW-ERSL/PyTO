"""Compatibility shim: mmaWrapper now lives at pyto.topopt.drivers.mmaWrapper.

Kept so the flat `from mmaWrapper import runMMA` call sites keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.topopt.drivers.mmaWrapper import *
from pyto.topopt.drivers.mmaWrapper import runMMA
