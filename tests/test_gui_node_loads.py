"""Node (point) loads and supports in the GUI: picking, applying, analysis, optimization selections, project file."""
import contextlib
import io
import json
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def gui(tmp_path_factory):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import pyvista as pv
    import pyto.gui.PyTOGUI as G
    from pyto.core.hex_mesher import HexMesher

    class ViewStub(pv.Plotter):              # the GUI's QtInteractor: show() just redraws
        def show(self, *a, **k):
            self.render()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesher = HexMesher()
    with contextlib.redirect_stdout(io.StringIO()):
        mesher.createMeshFromSTLFile(os.path.join(root, "Models", "Cantilever", "Cantilever.STL"), 2000)
        mesher.createEdofMatStructural()
        mesher.createEdofMatThermal()
    parent = QtWidgets.QWidget()
    parent.message_text = QtWidgets.QTextEdit()
    parent.plotter = ViewStub(off_screen=True)
    parent.settings = G.Settings()
    parent.stl_geom, parent.hex_mesh = mesher.stlGeom, mesher
    parent.applied_material = {"properties": {"Young's Modulus": 2.1e11, "Poisson's Ratio": 0.3, "Density": 7850,
                                              "Thermal Conductivity": 50.0, "Specific Heat Capacity": 460.0,
                                              "Thermal Expansion": 12e-6, "Price": 1.0, "Yield Strength": 2.5e8}}
    parent.update_LivVar = lambda *a, **k: None
    parent.LivVar = {'structural_loads': {'forces_applied': False, 'fixed_constraints': False, 'applied': False}}
    parent.notify_display_options_update = lambda *a, **k: None
    parent.update_highlights = lambda *a, **k: None
    for name in ("update_point_markers", "on_left_button_press", "on_right_button_press", "select_triangles"):
        setattr(parent, name, types.MethodType(getattr(G.MainWindow, name), parent))
    parent.highlight_mode, parent.selected_points, parent.highlight_actor = "coarse", [], None
    parent.constraint_data, parent.force_data, parent.constrained_triangles = [], [], set()
    parent.force_actors, parent.constraint_actors, parent.torque_actors = [], [], []
    parent.thermal_loads_window, parent.body_force, parent.topopt_options = None, None, {}
    parent.fe_solver = parent.thermal_fe_solver = parent.topopt_results = parent.optimization_spec = None
    verts = mesher.stlGeom.mesh.vectors.reshape(-1, 3)
    x0 = np.unique(verts[np.abs(verts[:, 0] - verts[:, 0].min()) < 1e-9], axis=0)      # clamped-end corners
    xmax = verts[:, 0].max()
    tip = verts[np.abs(verts[:, 0] - xmax) < 1e-9]
    tip = tip[np.argmin(tip[:, 1])]                                                    # one tip corner
    yield app, G, parent, x0, tip
    parent.plotter.close()


class _Picker:
    def __init__(self, cell):
        self.cell = cell

    def GetCellId(self):
        return self.cell

    def GetActor(self):
        return None


def _pick(parent, point):
    """Click the STL triangle that has `point` as a vertex, slightly off the vertex."""
    tri = int(np.argmin(np.linalg.norm(parent.stl_geom.mesh.vectors - np.asarray(point), axis=2).min(axis=1)))
    parent.on_left_button_press(np.asarray(point) + 1e-5, _Picker(tri))


def test_helpers(gui):
    app, G, parent, x0, tip = gui
    nodes, dist = G.nearest_surface_nodes([[0, 0, 0], [1, 1, 1]], np.array([7, 9]), np.array([[0, 0, 0.1], [1, 1, 1]]))
    assert list(nodes) == [7, 9] and np.allclose(dist, [0.1, 0.0])
    v = parent.stl_geom.mesh.vectors[0]
    assert np.allclose(G.snap_to_triangle_vertex(parent.stl_geom, 0, v[2] + 1e-6), v[2])


def test_node_picking_snaps_dedups_and_clears(gui):
    app, G, parent, x0, tip = gui
    parent.highlight_mode, parent.selected_points = "node", []
    _pick(parent, tip)
    assert len(parent.selected_points) == 1 and np.allclose(parent.selected_points[0], tip)   # snapped to the vertex
    _pick(parent, tip)
    assert len(parent.selected_points) == 1                                                      # duplicate refused
    assert "selected_points" in parent.plotter.actors
    parent.on_right_button_press(None, None)
    assert parent.selected_points == [] and "selected_points" not in parent.plotter.actors


def test_point_supports_and_load_through_the_windows(gui, tmp_path):
    app, G, parent, x0, tip = gui
    loads = G.StructuralLoadsWindow(parent)
    loads.selection_combo.setCurrentText("Node")
    assert parent.highlight_mode == "node"
    for p in x0:                                            # the clamped end's corner nodes
        _pick(parent, p)
    loads.load_type.setCurrentText("Fixed XYZ")
    loads.on_apply_clicked()
    _pick(parent, tip)
    loads.load_type.setCurrentText("Force")
    loads.force_spinboxes["Y"].setValue(-1000.0)
    loads.on_apply_clicked()
    assert [c["type"] for c in parent.constraint_data] == ["Fixed XYZ"] and len(parent.constraint_data[0]["points"]) == len(x0)
    assert parent.force_data[0]["points"] == [tip.tolist()] and parent.force_data[0]["force_y"] == -1000.0
    assert "point_supports" in parent.plotter.actors and parent.force_actors
    # torque on nodes is refused
    warned = []
    saved = G.QtWidgets.QMessageBox.warning
    G.QtWidgets.QMessageBox.warning = lambda *a, **k: warned.append(a[2])
    try:
        _pick(parent, tip)
        loads.load_type.setCurrentText("Torque")
        loads.on_apply_clicked()
    finally:
        G.QtWidgets.QMessageBox.warning = saved
    assert warned and "face" in warned[0] and len(parent.force_data) == 1
    parent.on_right_button_press(None, None)

    # structural analysis: the picked vertices become fixed nodes, the load sits on the node nearest the tip vertex
    analysis = G.AnalysisWindow(parent)
    analysis.solver_combo.setCurrentText("SPSOLVE")
    with contextlib.redirect_stdout(io.StringIO()):
        analysis.run_structural_analysis()
    fe = parent.fe_solver
    xyz = fe.mesh.node_xyz
    fixed_nodes = np.unique(np.asarray(fe.bc.fixed_dofs) // 3)
    expected_fixed = {int(np.argmin(np.linalg.norm(xyz - p, axis=1))) for p in x0}
    assert set(fixed_nodes.tolist()) == expected_fixed
    loaded = np.nonzero(fe.bc.force)[0]
    tip_node = int(np.argmin(np.linalg.norm(xyz - tip, axis=1)))
    assert loaded.tolist() == [3 * tip_node + 1] and fe.bc.force[3 * tip_node + 1] == pytest.approx(-1000.0)
    assert fe.max_deformation > 0

    # optimization: the point entries are the named selections Support1 / Load1
    win = G.StructuralTopOptWindow(parent)
    _, physics, selections = win.build_problem()
    assert physics == "structural" and set(selections["Support1"]["nodes"]) == expected_fixed
    assert list(selections["Load1"]["nodes"]) == [tip_node]

    # project file: point supports and loads survive save + restore
    projects = G.ProjectsWindow(parent)
    target = tmp_path / "p.pyto"
    saved_dialog = G.QtWidgets.QFileDialog.getSaveFileName
    G.QtWidgets.QFileDialog.getSaveFileName = lambda *a, **k: (str(target), "")
    try:
        projects.save_project()
    finally:
        G.QtWidgets.QFileDialog.getSaveFileName = saved_dialog
    bc = json.loads(target.read_text())["structuralBC"]
    assert bc["point_supports"][0]["type"] == "Fixed XYZ" and len(bc["point_loads"]) == 1
    constraint_data, force_data = parent.constraint_data, parent.force_data
    parent.constraint_data, parent.force_data = [], []
    parent.structural_loads_window = loads
    projects.restore_structural_loads(bc)
    assert parent.constraint_data[-1]["points"] == constraint_data[0]["points"]
    assert parent.force_data[-1]["points"] == force_data[0]["points"] and parent.force_data[-1]["force_y"] == -1000.0
    parent.constraint_data, parent.force_data = constraint_data, force_data


def test_single_point_support_reports_under_constrained(gui):
    """One fixed node leaves rotations free: the analysis shows the solver's message instead of failing silently."""
    app, G, parent, x0, tip = gui
    saved = (parent.constraint_data, parent.force_data)
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": [], "points": [x0[0].tolist()]}]
    parent.force_data = [{"type": "force_xyz", "triangles": [], "points": [tip.tolist()], "force_x": 0.0,
                          "force_y": -1000.0, "force_z": 0.0}]
    shown = []
    real = G.QtWidgets.QMessageBox.critical
    G.QtWidgets.QMessageBox.critical = lambda *a, **k: shown.append(a[2])
    try:
        analysis = G.AnalysisWindow(parent)
        analysis.solver_combo.setCurrentText("SPSOLVE")
        with contextlib.redirect_stdout(io.StringIO()):
            analysis.structural_button.click()
    finally:
        G.QtWidgets.QMessageBox.critical = real
        parent.constraint_data, parent.force_data = saved
    assert shown and "under-constrained" in shown[0]
