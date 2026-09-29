"""The GUI and demo scripts must call the FE solvers and optimization drivers with their current signatures.

The GUI cannot be exercised headlessly (VTK needs a display), and these call sites broke silently when the drivers
moved to the single-solver torch API. This parses the GUI sources and binds every call's arguments to the current
signature, so a future API change fails here instead of when a user clicks a button.
"""
import ast
import inspect
import os

import pytest
import torch

GUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pyto", "gui")


def _callables():
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.topopt.drivers.levelset import topopt_levelset
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.drivers.oc import topopt_optimality_criteria
    from pyto.topopt.drivers.pareto import topopt_pareto
    return {f.__name__: f for f in (topopt_mma, topopt_optimality_criteria, topopt_pareto, topopt_levelset,
                                    HexStructuralFEA, HexThermalFEA)}


@pytest.mark.parametrize("source", ["PyTOGUI.py", "PyTO_demos.py"])
def test_gui_calls_match_current_signatures(source):
    funcs = _callables()
    tree = ast.parse(open(os.path.join(GUI_DIR, source)).read())
    mismatches, checked = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name not in funcs:
            continue
        checked += 1
        try:
            inspect.signature(funcs[name]).bind(*[None] * len(node.args),
                                                **{k.arg: None for k in node.keywords if k.arg})
        except TypeError as e:
            mismatches.append(f"{source}:{node.lineno} {name}: {e}")
    assert checked > 0
    assert not mismatches, "\n".join(mismatches)


def test_structural_solve_without_design_is_a_solid_analysis(structural_fe_solver):
    # The GUI's "run structural analysis" calls solve() with no design (as HexThermalFEA.solve allows).
    fe = structural_fe_solver
    u_default = fe.solve()
    u_solid = fe.solve(torch.ones(fe.mesh.num_elems, dtype=torch.float64))
    assert torch.allclose(u_default, u_solid)


def test_plot_temperature_accepts_grad_tracked_solution(thermal_fe_solver, tmp_path):
    # After an optimization the thermal solution carries autograd history; plotting must still work.
    from pyto.autodiff.material_model import MaterialModel
    fe = thermal_fe_solver
    assert fe.plot_temperature(save_path=str(tmp_path / "none.png")) is None   # nothing solved yet
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64, requires_grad=True)
    fe.solve(x, MaterialModel.SIMP)
    fe.plot_temperature(save_path=str(tmp_path / "T.png"))
    assert (tmp_path / "T.png").exists()
