"""Every script in docs/examples runs to completion, so the documentation's code stays correct."""
import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "docs", "examples", "*.py")))


@pytest.mark.parametrize("script", EXAMPLES, ids=os.path.basename)
def test_docs_example_runs(script, tmp_path):
    env = dict(os.environ, MPLBACKEND="Agg", PYVISTA_OFF_SCREEN="true", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, script], cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
