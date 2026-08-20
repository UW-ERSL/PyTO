"""Compatibility shim: topopt_common now lives at pyto.topopt.common.

Kept so the many flat `from topopt_common import *` call sites elsewhere
in src/ keep working unmodified during the restructuring (see plan
Phase 2-7). Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.topopt.common import *
