"""Start the PyTO GUI.

Run from anywhere, with the PyToLib environment:

    conda activate PyToLib
    python run_gui.py

(same as `PYTHONPATH=src python -m pyto.gui.PyTOGUI`, without having to set PYTHONPATH). For scripted runs and
benchmarks, use main.ipynb instead.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

for var in ("PYVISTA_OFF_SCREEN", "MPLBACKEND"):     # headless settings from test/benchmark shells would hide the GUI
    if os.environ.get(var) in ("true", "True", "1", "Agg"):
        os.environ.pop(var)

from pyto.gui.PyTOGUI import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
