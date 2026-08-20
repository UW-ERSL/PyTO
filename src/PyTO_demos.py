"""Compatibility shim: PyTO_demos now lives at pyto.gui.PyTO_demos.

This runs an interactive input() prompt loop unconditionally at import
time (no `if __name__ == "__main__":` guard) -- excluded from the pytest
suite via _SKIP in tests/test_imports_smoke.py, unaffected by this move.
Kept so running the flat path still works, since the moved module runs
its side effects the same way either way. Remove once nothing references
this flat path anymore (Phase 7).
"""
import pyto.gui.PyTO_demos  # noqa: F401 (imported for its side effects)
