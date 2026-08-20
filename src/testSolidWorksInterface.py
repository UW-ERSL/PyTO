"""Compatibility shim: testSolidWorksInterface now lives at pyto.io.testSolidWorksInterface.

This is a manual, interactive smoke script (requires a live SolidWorks COM
connection on Windows) -- excluded from the pytest suite via _SKIP in
tests/test_imports_smoke.py, unaffected by this move. Kept so running the
flat path still works, since the moved module runs its side effects
(SolidWorks COM calls) at import time either way. Remove once nothing
references this flat path anymore (Phase 7).
"""
import pyto.io.testSolidWorksInterface  # noqa: F401 (imported for its side effects)
