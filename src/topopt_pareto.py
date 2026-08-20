"""Compatibility shim: topopt_pareto now lives at pyto.topopt.drivers.pareto.

Kept so the flat `from topopt_pareto import topopt_pareto` call sites
(PyTOGUI.py, PyTO_demos.py, topopt_run_benchmarks.py) keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.topopt.drivers.pareto import *
from pyto.topopt.drivers.pareto import topopt_pareto, run_pareto_topopt

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/topopt/drivers/pareto.py's own
    # `if __name__ == "__main__":` demo block actually runs -- a plain
    # import would leave __name__ as "pyto.topopt.drivers.pareto" there.
    import runpy
    runpy.run_module("pyto.topopt.drivers.pareto", run_name="__main__")
