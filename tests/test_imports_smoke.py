"""Every module in src/ should import cleanly.

This turns an ad-hoc "does it even import" check into a permanent
regression test -- catches merge fallout (renamed constants, changed
attribute names) and dependency-availability problems immediately, rather
than several call-frames deep inside an optimizer run.
"""
import glob
import importlib
import os

import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Scripts with heavy top-level side effects (open a Qt window, run a
# multi-minute benchmark sweep, expect argv, etc.) are exercised elsewhere
# (or manually) rather than via a bare import here.
_SKIP = {
    "PyTOGUI",  # opens a Qt/VTK window at import time via module-level setup
    "PyTO_demos",  # runs an interactive input() prompt loop at import time
    "testSolidWorksInterface",  # requires a live SolidWorks COM connection
    "solidworks_interface",  # Windows-only: imports win32com.client
    "topopt_solver_comparison",  # runs a multi-minute benchmark sweep at import time
}

# Modules with a known, currently-broken import unrelated to the autodiff
# work this suite otherwise targets (see plan for scope). Documented here so
# a real regression elsewhere isn't masked by a blanket skip.
_XFAIL_IMPORTS = {
    "hex_transient_thermal": "mat_lib.ThermalMaterial does not exist (merge fallout, transient-thermal path, out of scope for the autodiff plan)",
}

_MODULE_NAMES = sorted(
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(SRC_DIR, "*.py"))
    if os.path.basename(f) != "__init__.py"
)


@pytest.mark.parametrize("module_name", [m for m in _MODULE_NAMES if m not in _SKIP])
def test_module_imports(module_name):
    if module_name in _XFAIL_IMPORTS:
        pytest.xfail(_XFAIL_IMPORTS[module_name])
    importlib.import_module(module_name)
