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
    for name in ("update_point_markers", "on_left_button_press", "on_right_button_press", "select_triangles",
                 "add_selected_points", "show_candidate_nodes", "start_box_node_selection", "on_box_node_selection",
                 "restore_surface_picking", "node_selection_preview", "_finish_box_node_selection"):
        setattr(parent, name, types.MethodType(getattr(G.MainWindow, name), parent))
    parent.highlight_mode, parent.selected_points, parent.highlight_actor = "coarse", [], None
    parent.selected_regions = []
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
    assert len(parent.selected_points) == 1 and np.allclose(parent.selected_points[0], tip, atol=1e-6)  # mesh corner node
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


def test_candidates_are_mesh_surface_nodes_once_meshed(gui):
    app, G, parent, x0, tip = gui
    mesh = parent.hex_mesh
    cand = G.candidate_node_coordinates(parent)
    assert np.allclose(cand, mesh.node_xyz[mesh.get_boundary_nodes()])
    assert len(cand) > len(np.unique(parent.stl_geom.mesh.vectors.reshape(-1, 3), axis=0))   # far more than STL corners
    parent.hex_mesh = None                     # before meshing: a dense sample of the STL surface (incl. its corners)
    try:
        sample = G.candidate_node_coordinates(parent)
        corners = np.unique(parent.stl_geom.mesh.vectors.reshape(-1, 3), axis=0)
        assert len(sample) > 20 * len(corners)
        assert all(np.min(np.linalg.norm(sample - c, axis=1)) < 1e-9 for c in corners)
    finally:
        parent.hex_mesh = mesh
    # a click between nodes snaps to the nearest mesh surface node, not only to STL corners
    parent.highlight_mode, parent.selected_points = "node", []
    target = cand[len(cand) // 2]
    parent.on_left_button_press(target + 1e-4, _Picker(0))
    assert np.allclose(parent.selected_points[0], target)
    parent.on_right_button_press(None, None)


def test_regions_from_frustum():
    import vtk
    from pyto.gui.PyTOGUI import frustum_to_region, points_in_region
    planes = vtk.vtkPlanes()
    planes.SetBounds(0, 1, 0, 1, 0, 1)
    region = frustum_to_region(planes)
    assert len(region["normals"]) == 6 and len(region["origins"]) == 6
    assert list(points_in_region([[0.5, 0.5, 0.5], [2, 0.5, 0.5], [0.5, -1, 0.5], [1, 1, 1]], region)) == \
        [True, False, False, True]


def _end_box(parent, G, depth_elems=0.5):
    """vtkPlanes box around the free end (x > xmax - depth), like a rectangle dragged around it."""
    import vtk
    cand = parent.hex_mesh.node_xyz
    h = float(max(parent.hex_mesh.elem_size))
    lo, hi = cand.min(axis=0) - 1e-6, cand.max(axis=0) + 1e-6
    planes = vtk.vtkPlanes()
    planes.SetBounds(cand[:, 0].max() - depth_elems * h, hi[0], lo[1], hi[1], lo[2], hi[2])
    return types.SimpleNamespace(frustum=planes), cand[:, 0].max() - depth_elems * h


def test_box_selection_before_meshing_is_a_region_resolved_on_the_mesh(gui, tmp_path):
    """Box select works without a mesh: the box is stored; at analysis time every mesh surface node inside it is
    used. It survives remeshing because it is a region, not a list of nodes."""
    import vtk
    app, G, parent, x0, tip = gui
    mesh = parent.hex_mesh
    selection, xcut = _end_box(parent, G)
    saved = (parent.constraint_data, parent.force_data)
    parent.hex_mesh = None                                           # no mesh yet
    try:
        parent.selected_points, parent.selected_regions = [], []
        parent.on_box_node_selection(selection)
        app.processEvents()                                  # the box is finished right after VTK returns
        assert len(parent.selected_regions) == 1
        preview = parent.node_selection_preview()
        assert len(preview) > 10 and np.all(preview[:, 0] >= xcut - 1e-9)       # STL surface points in the box
        empty = vtk.vtkPlanes()
        empty.SetBounds(100, 101, 100, 101, 100, 101)
        parent.on_box_node_selection(types.SimpleNamespace(frustum=empty))
        app.processEvents()
        assert len(parent.selected_regions) == 1                                 # empty box not added
        loads = G.StructuralLoadsWindow(parent)
        loads.selection_combo.setCurrentText("Node")
        loads.load_type.setCurrentText("Force")
        loads.force_spinboxes["Y"].setValue(-1000.0)
        loads.on_apply_clicked()
        load = parent.force_data[-1]
        assert load["regions"] and not load["points"] and parent.selected_regions == []
    finally:
        parent.hex_mesh = mesh
    # meshed now: the analysis loads every surface node inside the box, the total split evenly
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": [], "points": [p.tolist() for p in x0]}]
    parent.force_data = [load]
    try:
        analysis = G.AnalysisWindow(parent)
        analysis.solver_combo.setCurrentText("SPSOLVE")
        with contextlib.redirect_stdout(io.StringIO()):
            analysis.run_structural_analysis()
        fe = parent.fe_solver
        boundary = mesh.get_boundary_nodes()
        expected = set(boundary[mesh.node_xyz[boundary, 0] >= xcut].tolist())
        loaded = set((np.nonzero(fe.bc.force)[0] // 3).tolist())
        assert loaded == expected and len(expected) > 4
        assert float(np.sum(fe.bc.force)) == pytest.approx(-1000.0)
        # project file keeps the box
        projects = G.ProjectsWindow(parent)
        target = tmp_path / "box.pyto"
        real = G.QtWidgets.QFileDialog.getSaveFileName
        G.QtWidgets.QFileDialog.getSaveFileName = lambda *a, **k: (str(target), "")
        try:
            projects.save_project()
        finally:
            G.QtWidgets.QFileDialog.getSaveFileName = real
        bc = json.loads(target.read_text())["structuralBC"]
        assert bc["point_loads"][0]["regions"] == load["regions"]
        parent.force_data = []
        parent.structural_loads_window = G.StructuralLoadsWindow(parent)
        projects.restore_structural_loads(bc)
        assert parent.force_data[-1]["regions"] == load["regions"]
    finally:
        parent.constraint_data, parent.force_data = saved


def test_candidate_dots_and_box_button_only_in_node_mode(gui):
    app, G, parent, x0, tip = gui
    loads = G.StructuralLoadsWindow(parent)
    loads.selection_combo.setCurrentText("Node")
    assert "candidate_nodes" in parent.plotter.actors and not loads.box_select_button.isHidden()
    loads.selection_combo.setCurrentText("Facet")
    assert "candidate_nodes" not in parent.plotter.actors and loads.box_select_button.isHidden()
    loads.selection_combo.setCurrentText("Node")
    loads.close()
    assert "candidate_nodes" not in parent.plotter.actors


def test_box_button_starts_while_click_picking_is_on(gui):
    """The GUI always has click picking on; starting the box selection must switch it off first (pyvista allows
    one picking mode), and finishing it must switch click picking back on."""
    app, G, parent, x0, tip = gui
    parent.restore_surface_picking()                    # the GUI's normal state: click picking enabled
    parent.start_box_node_selection()                   # used to raise PyVistaPickingError
    selection, _ = _end_box(parent, G)
    parent.selected_points, parent.selected_regions = [], []
    parent.on_box_node_selection(selection)             # finishing restores click picking
    app.processEvents()
    parent.start_box_node_selection()                   # and a second box can be started again
    parent.on_box_node_selection(selection)
    app.processEvents()
    assert len(parent.selected_regions) == 2
    parent.on_right_button_press(None, None)


REAL_DRAG_SCRIPT = r"""
import os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
import numpy as np, pyvista as pv
from PyQt5 import QtWidgets
app = QtWidgets.QApplication([])
import pyto.gui.PyTOGUI as G
parent = QtWidgets.QWidget()
parent.message_text = QtWidgets.QTextEdit()
parent.plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
parent.plotter.add_mesh(pv.Cube())
parent.plotter.show(auto_close=False, interactive_update=True)          # a real VTK interactor
parent.stl_geom = types.SimpleNamespace(mesh=types.SimpleNamespace(vectors=np.asarray(
    pv.Cube().triangulate().points[pv.Cube().triangulate().faces.reshape(-1, 4)[:, 1:]])))
parent.hex_mesh = None
parent.selected_points, parent.selected_regions = [], []
for name in ("add_selected_points", "update_point_markers", "node_selection_preview", "start_box_node_selection",
             "on_box_node_selection", "_finish_box_node_selection", "restore_surface_picking", "on_left_button_press"):
    setattr(parent, name, types.MethodType(getattr(G.MainWindow, name), parent))
parent.restore_surface_picking()
for n in range(3):                                   # three real rubber-band drags in a row
    parent.start_box_node_selection()
    iren = parent.plotter.iren.interactor
    iren.SetEventInformation(120, 120); iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventInformation(280, 280); iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    app.processEvents()
print("REGIONS", len(parent.selected_regions))
"""


def test_real_rubber_band_drags_do_not_crash():
    """Regression: restoring picking inside VTK's end-of-selection handler segfaulted. Real interactor, three
    drags, in a subprocess so a crash fails the test instead of killing the run."""
    import subprocess, sys
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = {**os.environ, "PYVISTA_OFF_SCREEN": "true", "QT_QPA_PLATFORM": "offscreen"}
    r = subprocess.run([sys.executable, "-c", REAL_DRAG_SCRIPT, src], capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, f"crashed (exit {r.returncode}): {r.stderr[-500:]}"
    assert "REGIONS 3" in r.stdout, r.stdout[-500:]


def test_topopt_prerequisites_by_physics():
    from pyto.gui.PyTOGUI import topopt_load_problems as P
    assert P("structural", True, True, False, False) == []
    assert P("structural", True, False, True, False) == []                  # body force is a load
    assert P("structural", False, True, False, False) == ["supports (Fixed XYZ/X/Y/Z on faces or nodes)"]
    assert P("structural", True, False, False, False) == ["a load (force, torque or body force)"]
    assert P("thermal", False, False, False, True) == []
    assert P("thermo-structural", True, False, False, True) == []           # thermal expansion alone is a load
    assert P("thermo-structural", False, False, False, True) == ["supports (Fixed XYZ/X/Y/Z on faces or nodes)"]


def test_topopt_accepts_node_supports(gui):
    """Regression: TopOpt Execute said 'Please apply both forces and constraints' with node/box supports, because
    it counted only support faces."""
    app, G, parent, x0, tip = gui
    saved = (parent.constraint_data, parent.force_data, parent.constrained_triangles, parent.topopt_options)
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": [], "points": [p.tolist() for p in x0]}]
    parent.force_data = [{"type": "force_xyz", "triangles": [], "points": [tip.tolist()], "force_x": 0.0,
                          "force_y": -1000.0, "force_z": 0.0}]
    parent.constrained_triangles, parent.topopt_options = set(), {}          # no support faces at all
    warned = []
    real = G.QtWidgets.QMessageBox.warning
    G.QtWidgets.QMessageBox.warning = lambda *a, **k: warned.append(a[2])
    try:
        window = G.StructuralTopOptWindow(parent)
        assert window.problem_physics() == "structural" and window.check_prerequisites() and not warned
        parent.constraint_data = []                                          # really no supports: refused, and says what
        assert not window.check_prerequisites() and "supports" in warned[-1]
    finally:
        G.QtWidgets.QMessageBox.warning = real
        parent.constraint_data, parent.force_data, parent.constrained_triangles, parent.topopt_options = saved
