"""GUI optimization formulation (plan S5-S7): plain logic, and the Qt panel under the offscreen platform."""
import os

import numpy as np
import pytest

from pyto.gui import optimization_setup as gs
from pyto.topopt.common import TO_QOI
from pyto.topopt.spec import ConstraintSpec, MethodSpec, ObjectiveSpec, OptimizationSpec


def test_builtin_expressions_and_labels():
    assert gs.builtin_expression("Compliance") == "Compliance()"
    assert gs.builtin_expression("Displacement", "Load1", "y", "pnorm") == "Displacement(Load1, y, pnorm)"
    assert gs.builtin_expression("Temperature", "FixedTemp1", aggregate="mean") == "Temperature(FixedTemp1, mean)"
    assert gs.builtin_expression("Reaction force", "Support1", "magnitude") == "ReactionForce(Support1, y)"
    assert gs.builtin_expression("Volume fraction", "All") == "VolumeFraction()"
    assert "Temperature" not in gs.response_labels("structural")
    assert "Displacement" not in gs.response_labels("thermal") and "Temperature" in gs.response_labels("thermal")
    assert {"Stress (p-norm)", "Mechanical compliance", "Thermal compliance"} <= set(gs.response_labels("thermo-structural"))


OLD_OPTIONS = {"manufacturing": {"extrude": {"enabled": True, "value": "ZDir"},
                                 "cyclic_symmetry": {"enabled": True, "value": "(4) 90 deg"}},
               "performance": {"stress_safety": {"enabled": True, "value": 2.0},
                               "max_displacement": {"enabled": True, "value": 1e-3},
                               "max_temperature": {"enabled": False, "value": 2000.0}},
               "symmetry": {"x_symmetry": True, "y_symmetry": False, "z_symmetry": False},
               "other": {"keep_fixed_faces": True}}


def test_old_performance_options_migrate_as_disabled_constraints():
    rows = gs.migrate_performance_options(OLD_OPTIONS)
    assert [(r.expression, r.op, r.bound, r.enabled) for r in rows] == [
        ("StressFailureFactor()", "<=", 0.5, False), ("Displacement(All, magnitude, pnorm)", "<=", 1e-3, False)]
    assert gs.migrate_performance_options(None) == []


def test_manufacturing_from_options():
    m = gs.manufacturing_from_topopt_options(OLD_OPTIONS)
    assert (m.extrude, m.symmetry, m.cyclic_z, m.keep_selection) == ("Z", ["X"], 4, "FixedFaces")
    assert gs.manufacturing_from_topopt_options(None).extrude == ""


def test_prepare_run():
    sel = {"Load1": {"nodes": np.array([1, 2])}}
    plan = gs.prepare_run(gs.default_spec(0.4), "structural", sel)
    assert plan.to_params is not None and not plan.errors
    assert plan.to_params.Objective == (TO_QOI.COMPLIANCE, None) and "autograd" in plan.gradient
    assert plan.cost == "Per iteration: 1 forward FE solve + 1 adjoint solve."
    # OC chosen for a stress-constrained problem: refused, with the reason
    spec = gs.default_spec(0.4)
    spec.constraints.append(ConstraintSpec("StressFailureFactor()", "<=", 0.5))
    spec.method = MethodSpec("OC")
    plan = gs.prepare_run(spec, "structural", sel)
    assert plan.to_params is None and any("use MMA" in i.message for i in plan.errors)
    # a Python-file objective is refused without consent
    spec = OptimizationSpec(objective=ObjectiveSpec("python:/nonexistent.py::f"))
    plan = gs.prepare_run(spec, "structural", sel)
    assert plan.to_params is None and any("confirmation" in i.message or "Python" in i.message for i in plan.issues)


def test_python_function_objective_with_consent(tmp_path, structural_fe_solver):
    import torch
    from pyto.autodiff.material_model import MaterialModel
    f = tmp_path / "my_objective.py"
    f.write_text("def tip(sol, x, fe_solver):\n    return (sol ** 2).sum() * 1e9 + x.mean()\n")
    spec = OptimizationSpec(objective=ObjectiveSpec(f"python:{f}::tip"))
    plan = gs.prepare_run(spec, "structural", {}, allow_code=True)
    assert plan.to_params is not None, plan.issues
    fe = structural_fe_solver
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64)
    u = fe.solve(x, MaterialModel.SIMP)
    assert plan.to_params.Objective[1](u, x, fe).item() == pytest.approx(((u ** 2).sum() * 1e9 + 0.5).item())
    bad = OptimizationSpec(objective=ObjectiveSpec(f"python:{f}::missing"))
    assert any("no function" in i.message for i in gs.prepare_run(bad, "structural", {}, allow_code=True).errors)


def test_display_objective_flips_sign_for_maximize():
    spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(All, y)", "maximize"))
    assert gs.display_objective(spec, [np.array([2.0]), -1.5]) == [-2.0, 1.5]
    assert gs.display_objective(gs.default_spec(), [3.0]) == [3.0]


def test_project_round_trip_and_migration():
    spec = gs.default_spec(0.3)
    spec.constraints.append(ConstraintSpec("Displacement(Load1, y, pnorm)", "<=", 2e-3))
    data = {**gs.project_entries(spec, {"X": 0.0, "Y": -9.81, "Z": 0.0})}
    restored, bf = gs.restore_from_project(data)
    assert restored == spec and bf == {"X": 0.0, "Y": -9.81, "Z": 0.0}
    old, bf = gs.restore_from_project({"topopt_options": OLD_OPTIONS})          # project from before this change
    assert old.objective.expression == "Compliance()" and [c.enabled for c in old.constraints] == [True, False, False]
    assert gs.restore_from_project({}) == (None, None)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_formulation_panel_round_trip_and_method_enabling(qapp):
    panel = gs.FormulationPanel()
    sel = {"Load1": {"nodes": [1]}, "Support1": {"nodes": [2]}}
    panel.set_context("structural", sel)
    spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(Load1, y, mean) / 1e-3", "maximize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.35),
                                         ConstraintSpec("StressFailureFactor()", "<=", 0.5, enabled=False)],
                            method=MethodSpec("MMA", 77, 0.15))
    panel.set_spec(spec)
    out = panel.get_spec()
    assert out.objective == spec.objective and out.constraints == spec.constraints
    assert (out.method.name, out.method.max_iterations, round(out.method.move_limit, 2)) == ("MMA", 77, 0.15)
    # a non-compliance objective disables OC/Pareto/LevelSet in the method list
    panel.refresh_method_info()
    model = panel.method_combo.model()
    enabled = {panel.method_combo.itemText(i): model.item(i).isEnabled() for i in range(panel.method_combo.count())}
    assert enabled == {"MMA": True, "OC": False, "PARETO": False, "LEVELSET": False}
    assert "autograd" in panel.gradient_label.text()
    # the classical formulation enables them again, and choosing a built-in fills in the expression
    panel.set_spec(gs.default_spec())
    panel.refresh_method_info()
    assert all(model.item(i).isEnabled() for i in range(panel.method_combo.count()))
    panel.type_combo.setCurrentText("Displacement")
    panel.selection_combo.setCurrentText("Load1")
    panel.component_combo.setCurrentText("y")
    assert panel.expression_edit.text() == "Displacement(Load1, y, mean)"
    # thermal context: the quantities offered change
    panel.set_context("thermal", {"FixedTemp1": {"nodes": [0]}})
    items = [panel.type_combo.itemText(i) for i in range(panel.type_combo.count())]
    assert "Temperature" in items and "Displacement" not in items


def test_gradient_choice(qapp):
    """AD by default; Manual only where a hand-derived gradient exists; Pareto/LevelSet forced to Manual."""
    panel = gs.FormulationPanel()
    panel.set_context("structural", {"Load1": {"nodes": [1]}})
    panel.set_spec(gs.default_spec())
    panel.refresh_method_info()
    manual_item = panel.gradient_combo.model().item(1)
    assert panel.get_spec().method.gradient == "autodiff" and manual_item.isEnabled()
    panel.gradient_combo.setCurrentIndex(1)
    assert panel.get_spec().method.gradient == "manual" and "Hand-derived" in panel.gradient_label.text()
    plan = gs.prepare_run(panel.get_spec(), "structural", {"Load1": {"nodes": [1]}})
    assert not plan.errors and plan.to_params.Gradient == "manual" and "no adjoint" in plan.cost
    # a user expression has no hand-derived gradient: Manual greyed out, back to AD
    spec = panel.get_spec()
    spec.objective = ObjectiveSpec("Displacement(Load1, y, mean)", "maximize")
    panel.set_spec(spec)
    panel.refresh_method_info()
    assert not manual_item.isEnabled() and panel.get_spec().method.gradient == "autodiff"
    spec.method.gradient = "manual"
    assert any("Manual gradient not available" in str(i) for i in gs.prepare_run(spec, "structural",
                                                                                    {"Load1": {"nodes": [1]}}).errors)
    # Pareto: forced to Manual, combo locked
    panel.set_spec(gs.default_spec())
    panel.method_combo.setCurrentText("PARETO")
    assert not panel.gradient_combo.isEnabled() and panel.get_spec().method.gradient == "manual"
    panel.method_combo.setCurrentText("MMA")
    assert panel.gradient_combo.isEnabled() and panel.get_spec().method.gradient == "autodiff"
    # old saved specs without the field load as AD
    assert OptimizationSpec.from_dict({"method": {"name": "MMA"}}).method.gradient == "autodiff"
