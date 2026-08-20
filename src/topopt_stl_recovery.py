"""Compatibility shim: topopt_stl_recovery now lives at pyto.io.topopt_stl_recovery.

Kept so the flat `from topopt_stl_recovery import extract_isosurface_cnn,
subtract_voids_from_stl` call sites keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.io.topopt_stl_recovery import *
from pyto.io.topopt_stl_recovery import (
    CNN3D,
    extract_isosurface_cnn,
    visualize,
    to_trimesh,
    subtract_voids_from_stl,
)

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/io/topopt_stl_recovery.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.io.topopt_stl_recovery" there.
    import runpy
    runpy.run_module("pyto.io.topopt_stl_recovery", run_name="__main__")
