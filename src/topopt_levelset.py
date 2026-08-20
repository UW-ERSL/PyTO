"""Compatibility shim: topopt_levelset now lives at pyto.topopt.drivers.levelset.

Kept so the flat `from topopt_levelset import topopt_levelset` call sites
(PyTOGUI.py, PyTO_demos.py) keep working unmodified during the
restructuring (see plan Phase 2-7). Remove once nothing imports this flat
path anymore (Phase 7).
"""
from pyto.topopt.drivers.levelset import *
from pyto.topopt.drivers.levelset import topopt_levelset, run_topopt_levelset

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/topopt/drivers/levelset.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.topopt.drivers.levelset" there.
    import runpy
    runpy.run_module("pyto.topopt.drivers.levelset", run_name="__main__")
