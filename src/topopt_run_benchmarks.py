"""Compatibility shim: topopt_run_benchmarks now lives at
pyto.examples_benchmarks.topopt_run_benchmarks.

Kept so the flat `from topopt_run_benchmarks import *` call sites keep
working unmodified during the restructuring (see plan Phase 2-7). Remove
once nothing imports this flat path anymore (Phase 7).
"""
from pyto.examples_benchmarks.topopt_run_benchmarks import *

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/examples_benchmarks/topopt_run_benchmarks.py's
    # own `if __name__ == "__main__":` sweep actually runs -- a plain import
    # would leave __name__ as "pyto.examples_benchmarks.topopt_run_benchmarks" there.
    import runpy
    runpy.run_module("pyto.examples_benchmarks.topopt_run_benchmarks", run_name="__main__")
