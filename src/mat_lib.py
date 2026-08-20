"""Compatibility shim: mat_lib now lives at pyto.core.mat_lib.

Kept so the ~50 flat `import mat_lib` call sites elsewhere in src/ keep
working unmodified during the restructuring (see plan Phase 2-7). Remove
once nothing imports this flat path anymore (Phase 7).
"""
from pyto.core.mat_lib import *
from pyto.core.mat_lib import Material, materials, get_material, create_material_with_defaults
