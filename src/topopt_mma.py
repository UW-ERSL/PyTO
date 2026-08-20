"""Compatibility shim: topopt_mma now lives at pyto.topopt.drivers.mma.

Kept so the flat `from topopt_mma import topopt_mma` call sites (PyTOGUI.py,
PyTO_demos.py, other drivers/benchmarks) keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.topopt.drivers.mma import *
from pyto.topopt.drivers.mma import topopt_mma

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/topopt/drivers/mma.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.topopt.drivers.mma" there.
    import runpy
    runpy.run_module("pyto.topopt.drivers.mma", run_name="__main__")
