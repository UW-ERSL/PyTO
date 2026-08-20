"""Compatibility shim: topopt_filters now lives at pyto.topopt.filters.

Kept so the flat `from topopt_filters import *` call sites elsewhere in
src/ keep working unmodified during the restructuring (see plan
Phase 2-7). Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.topopt.filters import *
