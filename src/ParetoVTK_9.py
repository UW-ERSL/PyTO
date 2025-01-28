import sys
from PyQt5 import QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import math
from stl import mesh
from collections import defaultdict
from queue import Queue

class STLGeom:
    TOL = 1e-9

    def __init__(self, file_path):
        self.mesh = mesh.Mesh.from_file(file_path)
        self.stl_n_triangles = len(self.mesh.vectors)
        self.tri_normals = [self.compute_normal(vertices) for vertices in self.mesh.vectors]
        self.tri_areas = [self.get_area_of_triangle(i) for i in range(self.stl_n_triangles)]
        self.tri_neighbors = self.compute_neighbors()
        self.tri_highlight = [False] * self.stl_n_triangles
        self.selected_triangles = set()
        self.file_path = file_path

    def compute_normal(self, vertices):
        v1 = [vertices[1][i] - vertices[0][i] for i in range(3)]
        v2 = [vertices[2][i] - vertices[0][i] for i in range(3)]
        normal = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        norm = math.sqrt(sum(n ** 2 for n in normal)) or self.TOL
        return [n / norm for n in normal]
    
    def compute_neighbors(self):
        edge_map = defaultdict(list)  # Map of edges to triangle indices
        neighbors = [[] for _ in range(self.stl_n_triangles)]

        for i, vertices in enumerate(self.mesh.vectors):
            edges = [
                tuple(sorted((tuple(vertices[0]), tuple(vertices[1])))),
                tuple(sorted((tuple(vertices[1]), tuple(vertices[2])))),
                tuple(sorted((tuple(vertices[2]), tuple(vertices[0])))),
            ]

            for edge in edges:
                edge_map[edge].append(i)

        for edge, tri_list in edge_map.items():
            for t1 in tri_list:
                for t2 in tri_list:
                    if t1 != t2 and t2 not in neighbors[t1]:
                        neighbors[t1].append(t2)

        return neighbors

    def get_area_of_triangle(self, triangle_index):
        vertices = self.mesh.vectors[triangle_index]
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        return self.compute_area_of_triangle(x, y, z)

    def compute_area_of_triangle(self, x, y, z):
        v1 = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
        v2 = [x[2] - x[0], y[2] - y[0], z[2] - z[0]]

        cross_product = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        cross_product_norm = math.sqrt(sum(c ** 2 for c in cross_product))
        return 0.5 * cross_product_norm
    
    def highlight_triangles_recursive(self, seed_triangle, depth, cutoff_angle_degrees):
        """
        Toggle highlight state for triangles recursively based on the angle between normals.
        If the seed triangle is highlighted, recursively deselect; otherwise, highlight.
        """

        # # Clear previous queue
        # while not self.queue.empty():
        #     self.queue.get()

        cumulative_area = 0
        cos_theta = math.cos(math.radians(cutoff_angle_degrees))

        # Always set to True for left click (no more toggle)
        target_state = True

        # Initialize queue
        q = Queue()
        q.put((seed_triangle, depth))
        self.tri_highlight[seed_triangle] = target_state

        # Keep track of processed triangles to avoid cycles
        processed = {seed_triangle}

        while not q.empty():
            current_tri, current_depth = q.get()
            if current_depth == 0:
                continue

            cumulative_area += self.get_area_of_triangle(current_tri)
            n1 = self.tri_normals[current_tri]

            for neighbor_tri in self.tri_neighbors[current_tri]:
                if neighbor_tri not in processed:
                    n2 = self.tri_normals[neighbor_tri]
                if not self.tri_highlight[neighbor_tri] and self.dot_product(n1, n2) > cos_theta:
                    self.tri_highlight[neighbor_tri] = target_state
                    q.put((neighbor_tri, current_depth - 1))
                    processed.add(neighbor_tri)

        highlighted_count = sum(1 for x in self.tri_highlight if x)
        return highlighted_count, cumulative_area

    def store_selected_triangles(self):
        selected_indices = [i for i, is_highlighted in enumerate(self.tri_highlight) if is_highlighted]
        selected_triangles_data = []
        
        for idx in selected_indices:
            triangle_data = {
                'index': idx,
                'vertices': self.mesh.vectors[idx],
                'normal': self.tri_normals[idx],
                'area': self.tri_areas[idx],
                'center': self.get_triangle_center(idx)
            }
            selected_triangles_data.append(triangle_data)
   
        return selected_triangles_data

    def get_triangle_center(self, triangle_index):
        vertices = self.mesh.vectors[triangle_index]
        center = [(vertices[0][i] + vertices[1][i] + vertices[2][i])/3 for i in range(3)]
        return center

    @staticmethod
    def dot_product(v1, v2):
        """
        Compute the dot product of two 3D vectors.
        """
        return sum(a * b for a, b in zip(v1, v2))

    

class Settings:
    def __init__(self):
        self.unit_system = "MKS"
        self.temperature_unit = "Kelvin"
        self.angle_unit = "Degree"

    def update_settings(self, unit_system, temperature_unit, angle_unit):
        self.unit_system = unit_system
        self.temperature_unit = temperature_unit
        self.angle_unit = angle_unit

class GeometryWindow(QtWidgets.QDialog):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.setWindowTitle("Geometry")
        self.resize(300, 200)
        self.main_window = main_window

        layout = QtWidgets.QVBoxLayout(self)

        load_button = QtWidgets.QPushButton("Load Geometry")
        load_button.clicked.connect(self.load_geometry)
        layout.addWidget(load_button)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def load_geometry(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Select STL File", 
            "", 
            "STL Files (*.stl *.STL);;All Files (*)", 
            options=options
        )

        if file_path:
            try:
                self.main_window.load_stl_file(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load geometry: {str(e)}")

class DisplayOptionsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Display Options")
        self.resize(300, 100)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.edge_toggle = QtWidgets.QCheckBox("Show Edges")
        self.edge_toggle.stateChanged.connect(self.parent.toggle_edges)
        layout.addWidget(self.edge_toggle)


class StructuralLoadsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Structural Loads")
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Load Type Group
        load_group = QtWidgets.QGroupBox("Load Type")
        load_layout = QtWidgets.QVBoxLayout()
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems(["FixedXYZ"])
        load_layout.addWidget(self.load_type)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_load)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def apply_load(self):
        if self.load_type.currentText() == "FixedXYZ":
            if self.parent.stl_geom:
                # Clear existing constraint markers
                for actor in self.parent.constraint_actors:
                    self.parent.renderer.RemoveActor(actor)
                self.parent.constraint_actors = []
                
                # Get selected faces and add markers
                selected_faces = self.parent.stl_geom.store_selected_triangles()
                
                # Determine how many markers to show
                MAX_MARKERS = 5  # Maximum number of markers to show
                THRESHOLD = 25   # Threshold for when to start limiting markers
                
                if len(selected_faces) > THRESHOLD:
                    # Calculate step size to evenly distribute markers
                    step = len(selected_faces) // MAX_MARKERS
                    # Get evenly distributed indices
                    display_indices = range(0, len(selected_faces), step)[:MAX_MARKERS]
                    # Create a subset of faces to display markers for
                    display_faces = [selected_faces[i] for i in display_indices]
                    # Update message
                    self.parent.message_text.append(f"\nShowing {len(display_faces)} markers for {len(selected_faces)} selected triangles")
                else:
                    display_faces = selected_faces
                    
                for triangle in display_faces:
                    axes = vtk.vtkAxesActor()
                    axes.SetTotalLength(0.05, 0.05, 0.05)
                    transform = vtk.vtkTransform()
                    transform.Translate(triangle['center'])
                    normal = triangle['normal']
                    rotation_matrix = self.parent.compute_alignment_matrix(normal)
                    transform.Concatenate(rotation_matrix)
                    axes.SetUserTransform(transform)
                    self.parent.renderer.AddActor(axes)
                    self.parent.constraint_actors.append(axes)
                    
                self.parent.vtkWidget.GetRenderWindow().Render()
            self.close()  # Close the structural loads window


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.settings = Settings()
        self.stl_geom = None
        self.constraint_actors = []
        
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        # Create horizontal layout for VTK and sidebar
        self.h_layout = QtWidgets.QHBoxLayout()
        
        # VTK Setup
        self.vtk_frame = QtWidgets.QFrame()
        self.vtk_layout = QtWidgets.QVBoxLayout(self.vtk_frame)
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_layout.addWidget(self.vtkWidget)
        self.h_layout.addWidget(self.vtk_frame, stretch=3)
        
        # Sidebar
        self.setup_sidebar()
        self.h_layout.addWidget(self.sidebar, stretch=0)
        
        # Message Frame
        self.setup_message_frame()
        
        # Add layouts to main layout
        self.main_layout.addLayout(self.h_layout)
        self.main_layout.addWidget(self.message_frame)
        
        # Status Bar
        self.setup_status_bar()
        
        self.setup_vtk()

    def toggle_edges(self):
        if hasattr(self, 'stl_actor'):
            prop = self.stl_actor.GetProperty()
            prop.SetEdgeVisibility(not prop.GetEdgeVisibility())
            prop.SetEdgeColor(0, 0, 0)  # Black edges
            self.vtkWidget.GetRenderWindow().Render()

    def open_display_options(self):
        dialog = DisplayOptionsWindow(self)
        dialog.exec_()

    def setup_sidebar(self):
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setFixedWidth(200)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(1)
        
        buttons = [
            ("Units", "#0066CC", self.open_units_window),
            ("Geometry", "#0066CC", self.open_geometry_window),
            ("Material", "#FF0000", None),
            ("Structural Loads", "#0066CC", self.open_structural_loads),
            ("Thermal Loads", "#FF0000", None),
            ("Body force", "#FF0000", None),
            ("Display Options", "#0066CC", self.open_display_options),
            ("Analysis", "#FF0000", None),
            ("TopOpt Constraints", "#FF0000", None),
            ("Structural TopOpt", "#FF0000", None),
            ("Thermal TopOpt", "#FF0000", None),
            ("TopOpt Results", "#FF0000", None),
            ("Projects", "#0066CC", None),
            ("Help", "#0066CC", None)
        ]
        
        for text, color, command in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    color: {color};
                    border: 1px solid #CCCCCC;
                    text-align: left;
                    padding: 5px;
                    font-family: 'Segoe UI';
                    font-size: 9pt;
                }}
                QPushButton:hover {{
                    background-color: #F0F0F0;
                }}
            """)
            if command:
                btn.clicked.connect(command)
            else:
                btn.setEnabled(False)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()

    def setup_message_frame(self):
        self.message_frame = QtWidgets.QFrame()
        self.message_frame.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        message_layout = QtWidgets.QVBoxLayout(self.message_frame)
        
        self.message_text = QtWidgets.QTextEdit()
        self.message_text.setFixedHeight(80)
        self.message_text.setStyleSheet("""
            QTextEdit {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
        """)
        self.message_text.setReadOnly(True)
        self.message_text.setText("Welcome to Pareto!")
        
        message_layout.addWidget(self.message_text)

    def setup_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
        """)
        
        version_label = QtWidgets.QLabel("Pareto Version 2024.02")
        build_label = QtWidgets.QLabel("Build Date 4.21")
        license_label = QtWidgets.QLabel(
            "This is an academic license, and should not be used for commercial purposes."
        )
        license_label.setStyleSheet("color: red;")
        
        status_bar.addWidget(version_label)
        status_bar.addWidget(build_label)
        status_bar.addWidget(license_label)

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        # Setup picker and interaction style
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.AddObserver("InteractionEvent", self.on_interaction)
        self.interactor.SetInteractorStyle(style)
        
        # Add coordinate axes
        axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOn()

        # Add mouse observers for both left and right clicks
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.interactor.AddObserver("RightButtonPressEvent", self.on_right_button_press)

        self.renderer.ResetCamera()
        self.interactor.Initialize()
    
    def on_interaction(self, obj, event):
        if hasattr(self, 'highlight_poly_data'): 
            self.update_highlights()

    def load_stl_file(self, file_path):
        self.stl_geom = STLGeom(file_path)
        
        # Create vtkPolyData
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        for vertices in self.stl_geom.mesh.vectors:
            point_ids = []
            for v in vertices:
                point_ids.append(points.InsertNextPoint(v))
            triangle = vtk.vtkTriangle()
            for i in range(3):
                triangle.GetPointIds().SetId(i, point_ids[i])
            cells.InsertNextCell(triangle)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)
        self.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)

        # Remove existing actors and add new one
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(self.stl_actor)
        
        # Setup highlight actor
        self.highlight_poly_data = vtk.vtkPolyData()
        highlight_mapper = vtk.vtkPolyDataMapper()
        highlight_mapper.SetInputData(self.highlight_poly_data)
        
        self.highlight_actor = vtk.vtkActor()
        self.highlight_actor.SetMapper(highlight_mapper)
        self.highlight_actor.GetProperty().SetColor(1, 0, 0)
        self.highlight_actor.GetProperty().SetOpacity(0.6)
        
        self.renderer.AddActor(self.highlight_actor)
        
        # Configure depth peeling for proper transparency
        render_window = self.vtkWidget.GetRenderWindow()
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)
        self.renderer.UseDepthPeelingOn()
        self.renderer.SetMaximumNumberOfPeels(100)
        
        # Setup interactions
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        self.message_text.setText(f"Model loaded with {self.stl_geom.stl_n_triangles} triangles")
    
    def open_structural_loads(self):
        dialog = StructuralLoadsWindow(self)
        dialog.exec_()

    def update_highlights(self):
        if not self.stl_geom:
            return
            
        highlight_points = vtk.vtkPoints()
        highlight_cells = vtk.vtkCellArray()

        for i, highlight in enumerate(self.stl_geom.tri_highlight):
            if highlight or i in self.stl_geom.selected_triangles:
                vertices = self.stl_geom.mesh.vectors[i]
                point_ids = []
                for v in vertices:
                    point_ids.append(highlight_points.InsertNextPoint(v))
                triangle = vtk.vtkTriangle()
                for j in range(3):
                    triangle.GetPointIds().SetId(j, point_ids[j])
                highlight_cells.InsertNextCell(triangle)

        self.highlight_poly_data.SetPoints(highlight_points)
        self.highlight_poly_data.SetPolys(highlight_cells)
        self.vtkWidget.GetRenderWindow().Render()

    def compute_alignment_matrix(self, normal):
        matrix = vtk.vtkMatrix4x4()
        z = normal

        # Handle case where normal is parallel to z-axis
        if abs(abs(z[2]) - 1.0) < 1e-6:
            x = [1, 0, 0]
        else:
            x = [-z[1], z[0], 0]
            mag = math.sqrt(sum(v*v for v in x))
            if mag > 1e-6:  # Check for non-zero magnitude
                x = [v/mag for v in x]
            else:
                x = [1, 0, 0]  # Fallback for degenerate case

        # Calculate y using cross product
        y = [
            z[1]*x[2] - z[2]*x[1],
            z[2]*x[0] - z[0]*x[2],
            z[0]*x[1] - z[1]*x[0]
        ]

        for i in range(3):
            matrix.SetElement(i, 0, x[i])
            matrix.SetElement(i, 1, y[i])
            matrix.SetElement(i, 2, z[i])
            matrix.SetElement(i, 3, 0)
        matrix.SetElement(3, 3, 1)

        return matrix
    
    def clear_selections(self):
        # Clear all highlights
        if self.stl_geom:
            self.stl_geom.tri_highlight = [False] * self.stl_geom.stl_n_triangles
            self.stl_geom.selected_triangles.clear()
            self.update_highlights()
        
        # Remove all constraint markers
        for actor in self.constraint_actors:
            self.renderer.RemoveActor(actor)
        self.constraint_actors = []
        
        # Render the changes
        self.vtkWidget.GetRenderWindow().Render()

    def on_left_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        cell_id = self.picker.GetCellId()
        if self.stl_geom and cell_id >= 0:
            highlighted_count, area = self.stl_geom.highlight_triangles_recursive(
                seed_triangle=cell_id,
                depth=500,  
                cutoff_angle_degrees=30
            )
            self.message_text.append(f"\nSelected {highlighted_count} triangles with area: {area:.2f}")
            self.update_highlights()  # Only update the triangle highlights, not the markers
            # Make sure to call the original left button press behavior for camera control
            self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def on_right_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        cell_id = self.picker.GetCellId()
        if cell_id >= 0:  # If clicked on the model
            # Clear all selections and markers
            self.clear_selections()
            self.message_text.append("\nSelections cleared")
        
        # Make sure to call the original right button press behavior for camera control
        self.interactor.GetInteractorStyle().OnRightButtonDown()

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self, self)
        dialog.exec_()

class UnitsWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Units")
        self.resize(300, 200)
        self.settings = settings

        layout = QtWidgets.QVBoxLayout(self)

        # Add unit system, temperature, and angle controls
        for label_text, combo_name, items, current in [
            ("Unit System:", "unit_system", ["MKS", "mmKS", "IPS"], settings.unit_system),
            ("Temperature Unit:", "temperature_unit", ["Kelvin", "Celsius", "Fahrenheit"], settings.temperature_unit),
            ("Angle Unit:", "angle_unit", ["Degree", "Radian"], settings.angle_unit)
        ]:
            layout.addWidget(QtWidgets.QLabel(label_text))
            combo = QtWidgets.QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            setattr(self, combo_name, combo)
            layout.addWidget(combo)

        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(apply_button)

    def apply_settings(self):
        self.settings.update_settings(
            self.unit_system.currentText(),
            self.temperature_unit.currentText(),
            self.angle_unit.currentText()
        )
        self.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Pareto STL Viewer")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())