"""Compatibility shim: PyTOGUI now lives at pyto.gui.PyTOGUI.

Kept so `python PyTOGUI.py` and any flat `import PyTOGUI` call sites keep
working unmodified during the restructuring (see plan Phase 2-7). Remove
once nothing references this flat path anymore (Phase 7).

Uses runpy (not a plain `import`) for the __main__ case so
pyto/gui/PyTOGUI.py's own `if __name__ == "__main__":` app-launch block
actually runs -- a plain import would leave __name__ as
"pyto.gui.PyTOGUI" there, never matching "__main__".
"""
from pyto.gui.PyTOGUI import *

if __name__ == "__main__":
    import runpy
    runpy.run_module("pyto.gui.PyTOGUI", run_name="__main__")
