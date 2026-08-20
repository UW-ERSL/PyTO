"""Compatibility shim: solidworks_interface now lives at pyto.io.solidworks_interface.

Kept so the flat `from solidworks_interface import SolidWorksInterface`
call sites keep working unmodified during the restructuring (see plan
Phase 2-7). Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.io.solidworks_interface import *
from pyto.io.solidworks_interface import SolidWorksInterface

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/io/solidworks_interface.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.io.solidworks_interface" there.
    import runpy
    runpy.run_module("pyto.io.solidworks_interface", run_name="__main__")
