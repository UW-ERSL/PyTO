"""Compatibility shim: stl_reader now lives at pyto.io.stl_reader.

Kept so the flat `from stl_reader import STLGeom` call sites keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.io.stl_reader import *
from pyto.io.stl_reader import STLGeom

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/io/stl_reader.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.io.stl_reader" there.
    import runpy
    runpy.run_module("pyto.io.stl_reader", run_name="__main__")
