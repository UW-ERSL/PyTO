"""Start the PyTO GUI.

    python run_gui.py            start the GUI
    python run_gui.py --check    only report which Python/environment would be used, then exit

Any Python can start it: if the one running this file lacks the GUI's modules (PyQt5, pyvista, torch, ...), the
script restarts itself in the conda environment PyToLib through `conda run`, which activates that environment
properly. Set PYTO_CONDA_ENV to use another environment name. It adds src/ to the import path itself, so
PYTHONPATH is not needed. For scripted runs and benchmarks, use main.ipynb instead.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ENV_NAME = os.environ.get("PYTO_CONDA_ENV", "PyToLib")
REQUIRED = ("PyQt5", "pyvistaqt", "pyvista", "vtk", "torch", "scipy", "numpy", "matplotlib")
HERE = os.path.dirname(os.path.abspath(__file__))


def missing_modules():
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def find_conda():
    """The conda executable: $CONDA_EXE (set by conda's shell hook), conda on PATH, or the usual install places."""
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")] + [
        os.path.expanduser(f"~/{d}/bin/conda") for d in ("anaconda3", "miniconda3", "miniforge3", "mambaforge")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def env_prefix(conda, name):
    """Directory of the named conda environment, or None."""
    try:
        envs = json.loads(subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True,
                                         timeout=60).stdout)["envs"]
    except Exception:
        return None
    return next((e for e in envs if os.path.basename(e) == name), None)


def relaunch_in_env(argv):
    """Replace this process by `conda run -n ENV python run_gui.py ...` (activates the environment)."""
    conda = find_conda()
    prefix = env_prefix(conda, ENV_NAME) if conda else None
    if conda is None or prefix is None:              # also tells type checkers both are str below
        sys.exit(f"The GUI needs {', '.join(missing_modules())}, which this Python ({sys.executable}) does not have, "
                 f"and the conda environment '{ENV_NAME}' was not found"
                 f"{'' if conda else ' (conda not found either)'}.\n"
                 f"Create it (see requirements.txt), or run with the right interpreter, or set PYTO_CONDA_ENV.")
    os.environ["PYTO_GUI_RELAUNCHED"] = "1"          # loop guard
    # The environment's own python by full path: inside `conda run`, a bare "python" can still resolve to the
    # python of the environment that is active in the calling shell (seen with conda 22.9).
    python = os.path.join(prefix, "bin", "python")
    cmd = [conda, "run", "--no-capture-output", "-n", ENV_NAME, python, os.path.abspath(__file__)] + argv
    print(f"Starting the GUI in the conda environment '{ENV_NAME}'...", flush=True)
    os.execv(conda, cmd)


def main(argv):
    missing = missing_modules()
    if missing:
        if os.environ.get("PYTO_GUI_RELAUNCHED"):
            sys.exit(f"Even in '{ENV_NAME}' these modules are missing: {', '.join(missing)}. "
                     f"Install them there (see requirements.txt).")
        relaunch_in_env(argv)                        # does not return
    if "--check" in argv:
        print(f"OK: {sys.executable} (environment {os.path.basename(sys.prefix)}) has all GUI modules.")
        return 0
    sys.path.insert(0, os.path.join(HERE, "src"))
    for var in ("PYVISTA_OFF_SCREEN", "MPLBACKEND"):  # headless settings from test/benchmark shells would hide the GUI
        if os.environ.get(var) in ("true", "True", "1", "Agg"):
            os.environ.pop(var)
    from pyto.gui.PyTOGUI import main as gui_main # type: ignore
    return gui_main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
