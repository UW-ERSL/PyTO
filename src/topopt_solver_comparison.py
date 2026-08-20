"""Compatibility shim: topopt_solver_comparison now lives at
pyto.examples_benchmarks.topopt_solver_comparison.

This is a benchmark script that runs a multi-minute solver-timing sweep at
import time -- excluded from the pytest suite via _SKIP in
tests/test_imports_smoke.py, unaffected by this move. Kept so running the
flat path still works. Remove once nothing references this flat path
anymore (Phase 7).
"""
import pyto.examples_benchmarks.topopt_solver_comparison  # noqa: F401 (imported for its side effects)
