"""Compatibility shim: hex_plotter now lives at pyto.gui.hex_plotter.

Kept so the flat `from hex_plotter import HexFEAPlotter` call sites
(hex_structural_fea.py, hex_thermal_fea.py, hex_mesher.py, hex_modal_fea.py)
keep working unmodified during the restructuring (see plan Phase 2-7).
Remove once nothing imports this flat path anymore (Phase 7).
"""
from pyto.gui.hex_plotter import *
from pyto.gui.hex_plotter import HexFEAPlotter, format_value
