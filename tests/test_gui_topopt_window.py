"""The real TopOpt Execute window, driven headlessly (offscreen Qt): picked faces -> selections -> FE problem ->
formulation -> optimization -> live history (plan S5/S6)."""
import contextlib
import io
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def window_parent():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyto.core.hex_mesher import HexMesher
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesher = HexMesher()
    with contextlib.redirect_stdout(io.StringIO()):
        mesher.createMeshFromSTLFile(os.path.join(root, "Models", "Cantilever", "Cantilever.STL"), 2000)
        mesher.createEdofMatStructural()
        mesher.createEdofMatThermal()
    cen = mesher.stlGeom.mesh.vectors.mean(axis=1)
    x0 = [int(i) for i in np.where(np.abs(cen[:, 0] - cen[:, 0].min()) < 1e-9)[0]]
    x1 = [int(i) for i in np.where(np.abs(cen[:, 0] - cen[:, 0].max()) < 1e-9)[0]]
    parent = QtWidgets.QWidget()
    parent.message_text = QtWidgets.QTextEdit()
    parent.fe_solver, parent.hex_mesh, parent.stl_geom = None, mesher, mesher.stlGeom
    parent.applied_material = {"properties": {"Young's Modulus": 2.1e11, "Poisson's Ratio": 0.3, "Density": 7850,
                                              "Thermal Conductivity": 50.0, "Specific Heat Capacity": 460.0,
                                              "Thermal Expansion": 12e-6, "Price": 1.0, "Yield Strength": 2.5e8}}
    parent.constrained_triangles, parent.topopt_options, parent.optimization_spec = set(x0), {}, None
    parent.faces = (x0, x1)
    return app, parent


def _run(app, parent, spec, structural=True, thermal=None):
    import pyto.gui.PyTOGUI as G
    from pyto.gui.optimization_setup import display_objective, prepare_run
    x0, x1 = parent.faces
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": x0}] if structural else []
    parent.force_data = ([{"type": "force", "triangles": x1, "force_x": 0.0, "force_y": -1e4, "force_z": 0.0}]
                         if structural else [])
    parent.thermal_loads_window = types.SimpleNamespace(thermal_loads=thermal) if thermal else None
    w = G.StructuralTopOptWindow(parent)
    w.optimization_done.disconnect()
    done = {}
    w.optimization_done.connect(lambda ok, msg, h, u, fe: done.update(ok=ok, msg=msg, h=h))
    w.formulation.set_spec(spec)
    fe, physics, selections = w.build_problem()
    w.formulation.set_context(physics, selections)
    plan = prepare_run(w.formulation.get_spec(), physics, selections)
    assert not plan.errors, plan.issues
    w.spec, w.physics = w.formulation.get_spec(), physics
    with contextlib.redirect_stdout(io.StringIO()):
        w.run_optimization(plan, fe, w.spec)
    app.processEvents()                                   # deliver the queued live-plot updates
    return w, fe, physics, selections, done, display_objective(w.spec, done["h"]["objective"])


def test_structural_user_objective_through_the_window(window_parent):
    from pyto.topopt.spec import ConstraintSpec, MethodSpec, ObjectiveSpec, OptimizationSpec
    app, parent = window_parent
    spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(Load1, y, mean)", "maximize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.5),
                                         ConstraintSpec("StressFailureFactor()", "<=", 2.0)],
                            method=MethodSpec("MMA", 4))
    w, fe, physics, sel, done, J = _run(app, parent, spec)
    assert physics == "structural" and type(fe).__name__ == "HexStructuralFEA"
    assert {"Support1", "Supports", "Load1", "Loads"} <= set(sel) and len(sel["Load1"]["nodes"]) > 0
    assert done["ok"] and J[-1] > J[0]                    # maximized (negative) deflection increased
    # one live point per iteration; the final-design point is added by optimization_completed (disconnected here)
    assert len(w.history_figure.axes[0].lines[0].get_xdata()) == len(J) - 1


def test_thermal_problem_through_the_window(window_parent):
    from pyto.topopt.spec import MethodSpec, ObjectiveSpec, OptimizationSpec
    app, parent = window_parent
    x0, x1 = parent.faces
    thermal = {"fixed_temps": [{"triangles": x0, "temperature": 300.0}],
               "total_heat_sources": [{"triangles": x1, "total_heat": 50.0}]}
    spec = OptimizationSpec(objective=ObjectiveSpec("Temperature(All, pnorm)"), method=MethodSpec("MMA", 4))
    w, fe, physics, sel, done, J = _run(app, parent, spec, structural=False, thermal=thermal)
    assert physics == "thermal" and type(fe).__name__ == "HexThermalFEA" and set(sel) == {"FixedTemp1", "HeatSource1"}
    assert done["ok"] and J[-1] < J[0]
