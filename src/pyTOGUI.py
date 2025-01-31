import sys
from PyQt5 import QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import math
from stl import mesh
from collections import defaultdict
from queue import Queue

class ProjectData:
    def __init__(self):
        self.version = "1.0"
        self.stl_file_path = None
        self.settings = None
        self.selected_triangles = []
        self.loads = []
        self.constraints = []

    def to_dict(self):
        return {
            'version': self.version,
            'stl_file_path': self.stl_file_path,
            'settings': {
                'unit_system': self.settings.unit_system,
                'temperature_unit': self.settings.temperature_unit,
                'angle_unit': self.settings.angle_unit
            } if self.settings else None,
            'selected_triangles': self.selected_triangles,
            'loads': self.loads,
            'constraints': self.constraints
        }

    @classmethod
    def from_dict(cls, data):
        project = cls()
        project.version = data.get('version', "1.0")
        project.stl_file_path = data.get('stl_file_path')
        
        if data.get('settings'):
            settings = Settings()
            settings.update_settings(
                data['settings']['unit_system'],
                data['settings']['temperature_unit'],
                data['settings']['angle_unit']
            )
            project.settings = settings
        
        project.selected_triangles = data.get('selected_triangles', [])
        project.loads = data.get('loads', [])
        project.constraints = data.get('constraints', [])
        return project

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


class MaterialWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Material")
        self.parent = parent
        self.setFixedSize(300, 400)  # Fixed size window
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Material selection
        material_layout = QtWidgets.QHBoxLayout()
        material_label = QtWidgets.QLabel("Material")
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems([
            "AlloySteel",
            "Aluminum",
            "Titanium",
            "StainlessSteel",
            "Custom"
        ])
        material_layout.addWidget(material_label)
        material_layout.addWidget(self.material_combo)
        layout.addLayout(material_layout)
        
        # Properties form
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)
        
        # Create input fields with units
        self.young_input = QtWidgets.QLineEdit("2.1e+11")
        self.poisson_input = QtWidgets.QLineEdit("0.28")
        self.yield_input = QtWidgets.QLineEdit("5e+8")
        self.density_input = QtWidgets.QLineEdit("7700")
        self.thermal_cond_input = QtWidgets.QLineEdit("50")
        self.thermal_exp_input = QtWidgets.QLineEdit("1.3e-5")
        self.spec_heat_input = QtWidgets.QLineEdit("452")
        self.price_input = QtWidgets.QLineEdit("2")
        
        # Add rows to form with units
        form_layout.addRow("Young's Modulus (N/m^2):", self.young_input)
        form_layout.addRow("Poisson ratio ( - ):", self.poisson_input)
        form_layout.addRow("Yield Strength (N/m^2):", self.yield_input)
        form_layout.addRow("Density (kg/m^3):", self.density_input)
        form_layout.addRow("Thermal Conductivity (W/m-K):", self.thermal_cond_input)
        form_layout.addRow("Thermal Expansion (m/m-K):", self.thermal_exp_input)
        form_layout.addRow("Spec. Heat Capacity (J/kg-K):", self.spec_heat_input)
        form_layout.addRow("Price (US$/kg):", self.price_input)
        
        layout.addLayout(form_layout)
        
        # Optimization checkbox
        self.optimize_check = QtWidgets.QCheckBox("Do not optimize")
        layout.addWidget(self.optimize_check)
        
        # Apply button
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_material)
        layout.addWidget(self.apply_button)
        
        # Connect material change event
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        
        # Material data dictionary
        self.materials_data = {
            "AlloySteel": {
                "young": "2.1e+11",
                "poisson": "0.28",
                "yield": "5e+8",
                "density": "7700",
                "thermal_cond": "50",
                "thermal_exp": "1.3e-5",
                "spec_heat": "452",
                "price": "2"
            },
            "Aluminum": {
                "young": "7e+10",
                "poisson": "0.33",
                "yield": "2.8e+8",
                "density": "2700",
                "thermal_cond": "237",
                "thermal_exp": "2.3e-5",
                "spec_heat": "900",
                "price": "3"
            },
            "Titanium": {
                "young": "1.14e+11",
                "poisson": "0.34",
                "yield": "8.8e+8",
                "density": "4430",
                "thermal_cond": "21.9",
                "thermal_exp": "8.6e-6",
                "spec_heat": "522",
                "price": "20"
            },
            "StainlessSteel": {
                "young": "1.93e+11",
                "poisson": "0.31",
                "yield": "2.15e+8",
                "density": "8000",
                "thermal_cond": "16",
                "thermal_exp": "1.7e-5",
                "spec_heat": "500",
                "price": "4"
            }
        }

    def on_material_changed(self, material_name):
        if material_name != "Custom":
            material = self.materials_data[material_name]
            self.young_input.setText(material["young"])
            self.poisson_input.setText(material["poisson"])
            self.yield_input.setText(material["yield"])
            self.density_input.setText(material["density"])
            self.thermal_cond_input.setText(material["thermal_cond"])
            self.thermal_exp_input.setText(material["thermal_exp"])
            self.spec_heat_input.setText(material["spec_heat"])
            self.price_input.setText(material["price"])
        else:
            self.clear_inputs()

    def clear_inputs(self):
        self.young_input.clear()
        self.poisson_input.clear()
        self.yield_input.clear()
        self.density_input.clear()
        self.thermal_cond_input.clear()
        self.thermal_exp_input.clear()
        self.spec_heat_input.clear()
        self.price_input.clear()

    def apply_material(self):
        try:
            material_data = {
                "name": self.material_combo.currentText(),
                "young_modulus": float(self.young_input.text()),
                "poisson_ratio": float(self.poisson_input.text()),
                "yield_strength": float(self.yield_input.text()),
                "density": float(self.density_input.text()),
                "thermal_conductivity": float(self.thermal_cond_input.text()),
                "thermal_expansion": float(self.thermal_exp_input.text()),
                "specific_heat": float(self.spec_heat_input.text()),
                "price": float(self.price_input.text()),
                "do_not_optimize": self.optimize_check.isChecked()
            }
            
            # Validate inputs
            if not (0 < material_data["poisson_ratio"] < 0.5):
                raise ValueError("Poisson's ratio must be between 0 and 0.5")
            
            # Store material data in parent
            self.parent.material_data = material_data
            self.parent.message_text.append(f"\nMaterial applied: {material_data['name']}")
            
            # Change geometry color to indicate material application
            if hasattr(self.parent, 'stl_actor'):
                if material_data['name'] == "Custom":
                    self.parent.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                elif material_data['name'] == "AlloySteel":
                    self.parent.stl_actor.GetProperty().SetColor(0.7, 0.7, 0.8)
                elif material_data['name'] == "Aluminum":
                    self.parent.stl_actor.GetProperty().SetColor(0.9, 0.9, 0.9)
                elif material_data['name'] == "Titanium":
                    self.parent.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.7)
                elif material_data['name'] == "StainlessSteel":
                    self.parent.stl_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
                
                self.parent.vtkWidget.GetRenderWindow().Render()
            
            self.close()
            
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(e))

         


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
        self.load_type.addItems(["FixedXYZ", "Force"])
        self.load_type.currentTextChanged.connect(self.on_load_type_changed)
        load_layout.addWidget(self.load_type)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Force Options Group (initially hidden)
        self.force_options_group = QtWidgets.QGroupBox("Force Options")
        force_options_layout = QtWidgets.QVBoxLayout()
        
        # Magnitude input
        magnitude_layout = QtWidgets.QHBoxLayout()
        magnitude_layout.addWidget(QtWidgets.QLabel("Magnitude:"))
        self.magnitude_input = QtWidgets.QLineEdit()
        self.magnitude_input.setPlaceholderText("Enter magnitude")
        magnitude_layout.addWidget(self.magnitude_input)
        self.unit_label = QtWidgets.QLabel("N")
        magnitude_layout.addWidget(self.unit_label)
        force_options_layout.addLayout(magnitude_layout)
        
        # Direction selection
        self.direction_label = QtWidgets.QLabel("Base Direction:")
        force_options_layout.addWidget(self.direction_label)
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItems(["Global X", "Global Y", "Global Z"])
        force_options_layout.addWidget(self.direction_combo)
        
        # Angle inputs
        angles_group = QtWidgets.QGroupBox("Angle Adjustments")
        angles_layout = QtWidgets.QVBoxLayout()
        
        # XY plane rotation
        xy_layout = QtWidgets.QHBoxLayout()
        xy_layout.addWidget(QtWidgets.QLabel("XY Plane Angle:"))
        self.xy_angle_input = QtWidgets.QLineEdit()
        self.xy_angle_input.setPlaceholderText("Angle in XY plane (degrees)")
        xy_layout.addWidget(self.xy_angle_input)
        angles_layout.addLayout(xy_layout)
        
        # YZ plane rotation
        yz_layout = QtWidgets.QHBoxLayout()
        yz_layout.addWidget(QtWidgets.QLabel("YZ Plane Angle:"))
        self.yz_angle_input = QtWidgets.QLineEdit()
        self.yz_angle_input.setPlaceholderText("Angle in YZ plane (degrees)")
        yz_layout.addWidget(self.yz_angle_input)
        angles_layout.addLayout(yz_layout)
        
        # XZ plane rotation
        xz_layout = QtWidgets.QHBoxLayout()
        xz_layout.addWidget(QtWidgets.QLabel("XZ Plane Angle:"))
        self.xz_angle_input = QtWidgets.QLineEdit()
        self.xz_angle_input.setPlaceholderText("Angle in XZ plane (degrees)")
        xz_layout.addWidget(self.xz_angle_input)
        angles_layout.addLayout(xz_layout)
        
        angles_group.setLayout(angles_layout)
        force_options_layout.addWidget(angles_group)
        
        self.force_options_group.setLayout(force_options_layout)
        layout.addWidget(self.force_options_group)
        self.force_options_group.hide()  # Initially hidden
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_load)
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_angles)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(clear_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def clear_angles(self):
        self.xy_angle_input.clear()
        self.yz_angle_input.clear()
        self.xz_angle_input.clear()

    def on_load_type_changed(self, load_type):
        show_force_options = load_type == "Force"
        self.force_options_group.setVisible(show_force_options)
        self.adjustSize()

    def apply_load(self):
        load_type = self.load_type.currentText()
        
        if load_type == "FixedXYZ":
            self.apply_fixed_constraint()
        elif load_type == "Force":
            self.apply_force()
            
    def apply_fixed_constraint(self):
        if self.parent.stl_geom:
            # Clear existing constraint markers
            for actor in self.parent.constraint_actors:
                self.parent.renderer.RemoveActor(actor)
            self.parent.constraint_actors = []
            
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            
            MAX_MARKERS = 5
            THRESHOLD = 25
            
            if len(selected_faces) > THRESHOLD:
                step = len(selected_faces) // MAX_MARKERS
                display_indices = range(0, len(selected_faces), step)[:MAX_MARKERS]
                display_faces = [selected_faces[i] for i in display_indices]
                self.parent.message_text.append(f"\nShowing {len(display_faces)} markers for {len(selected_faces)} selected triangles")
            else:
                display_faces = selected_faces
                
            for triangle in display_faces:
                axes = vtk.vtkAxesActor()
                axes.SetTotalLength(0.05, 0.05, 0.05)
                transform = vtk.vtkTransform()
                transform.Translate(triangle['center'])
                transform.Scale(1, 1, 1)
                axes.SetUserTransform(transform)
                self.parent.renderer.AddActor(axes)
                self.parent.constraint_actors.append(axes)
            
            self.parent.vtkWidget.GetRenderWindow().Render()
            
    def apply_force(self):
        if not self.parent.stl_geom:
            return
            
        try:
            magnitude = float(self.magnitude_input.text())
            xy_angle = float(self.xy_angle_input.text()) if self.xy_angle_input.text() else 0
            yz_angle = float(self.yz_angle_input.text()) if self.yz_angle_input.text() else 0
            xz_angle = float(self.xz_angle_input.text()) if self.xz_angle_input.text() else 0
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter valid numeric values")
            return
            
        # Clear existing force markers
        for actor in self.parent.force_actors:
            self.parent.renderer.RemoveActor(actor)
        self.parent.force_actors = []
        
        selected_faces = self.parent.stl_geom.store_selected_triangles()
        
        MAX_MARKERS = 5
        THRESHOLD = 25
        
        if len(selected_faces) > THRESHOLD:
            step = len(selected_faces) // MAX_MARKERS
            display_indices = range(0, len(selected_faces), step)[:MAX_MARKERS]
            display_faces = [selected_faces[i] for i in display_indices]
        else:
            display_faces = selected_faces
            
        for triangle in display_faces:
            # Create arrow for visualization
            arrow = vtk.vtkArrowSource()
            arrow.SetTipLength(0.3)
            arrow.SetTipRadius(0.1)
            arrow.SetShaftRadius(0.03)
            
            # Set up transform
            transform = vtk.vtkTransform()
            transform.Translate(triangle['center'])
            
            # Apply base direction
            direction = self.direction_combo.currentText()
            if direction == "Global Y":
                transform.RotateZ(90)
            elif direction == "Global Z":
                transform.RotateY(-90)
            
            # Apply angle rotations
            transform.RotateZ(xy_angle)  # Rotation in XY plane
            transform.RotateY(xz_angle)  # Rotation in XZ plane
            transform.RotateX(yz_angle)  # Rotation in YZ plane
            
            # Scale arrow based on magnitude
            scale_factor = 0.1 * magnitude / 100
            transform.Scale(scale_factor, scale_factor, scale_factor)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(arrow.GetOutputPort())
            
            arrow_actor = vtk.vtkActor()
            arrow_actor.SetMapper(mapper)
            arrow_actor.SetUserTransform(transform)
            arrow_actor.GetProperty().SetColor(1, 0, 0)  # Red for force
            
            self.parent.renderer.AddActor(arrow_actor)
            self.parent.force_actors.append(arrow_actor)
            
        self.parent.vtkWidget.GetRenderWindow().Render()
        angle_msg = f" at angles (XY: {xy_angle}°, YZ: {yz_angle}°, XZ: {xz_angle}°)" if any([xy_angle, yz_angle, xz_angle]) else ""
        self.parent.message_text.append(f"\nApplied force to {len(selected_faces)} triangles{angle_msg}")
        self.close()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.settings = Settings()
        self.stl_geom = None
        self.constraint_actors = []
        self.force_actors = []
        self.material_data = None
        
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
            ("Save Project", "#009900", self.save_project),
            ("Load Project", "#009900", self.load_project),
            ("Units", "#0066CC", self.open_units_window),
            ("Geometry", "#0066CC", self.open_geometry_window),
            ("Material", "#0066CC", self.open_material_window),
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
    
    def open_material_window(self):
            dialog = MaterialWindow(self)
            dialog.exec_()
    
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
        
        # For global directions, use simpler transformation
        if normal in ([1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]):
            # Direct mapping for global directions
            matrix.Identity()
            if normal[0] != 0:  # X direction
                matrix.SetElement(0, 2, normal[0])
                matrix.SetElement(2, 0, -normal[0])
            elif normal[1] != 0:  # Y direction
                matrix.SetElement(1, 2, normal[1])
                matrix.SetElement(2, 1, -normal[1])
            elif normal[2] != 0:  # Z direction
                matrix.SetElement(2, 2, normal[2])
            return matrix
        
        # For arbitrary surface normal
        z = normal
        
        # Find perpendicular vector for x-axis
        if abs(z[2]) > 0.707:
            x = [-z[1], z[0], 0]
        else:
            x = [-z[2], 0, z[0]]
            
        # Normalize x
        mag = math.sqrt(sum(v*v for v in x))
        x = [v/mag for v in x]
        
        # Calculate y using cross product
        y = [
            z[1]*x[2] - z[2]*x[1],
            z[2]*x[0] - z[0]*x[2],
            z[0]*x[1] - z[1]*x[0]
        ]

        # Set matrix elements
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

        # Add clearing of force actors
        for actor in self.force_actors:
            self.renderer.RemoveActor(actor)
        self.force_actors = []
        
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
            self.update_highlights()  
            
            self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def on_right_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        cell_id = self.picker.GetCellId()
        if cell_id >= 0: 
            
            self.clear_selections()
            self.message_text.append("\nSelections cleared")
        
        
        self.interactor.GetInteractorStyle().OnRightButtonDown()

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self, self)
        dialog.exec_()

    def save_project(self):
        if not self.stl_geom:
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry loaded to save")
            return
            
        project = ProjectData()
        project.stl_file_path = self.stl_geom.file_path
        project.settings = self.settings
        
        # Save selected triangles
        if hasattr(self, 'stl_geom'):
            selected_triangles = self.stl_geom.store_selected_triangles()
            # Convert NumPy arrays and float32 to standard Python types
            for triangle in selected_triangles:
                if 'vertices' in triangle:
                    triangle['vertices'] = [list(map(float, vertex)) for vertex in triangle['vertices'].tolist()]
                if 'normal' in triangle:
                    triangle['normal'] = list(map(float, triangle['normal']))
                if 'center' in triangle:
                    triangle['center'] = list(map(float, triangle['center']))
                if 'area' in triangle:
                    triangle['area'] = float(triangle['area'])
            project.selected_triangles = selected_triangles
        
        # Save loads and constraints data
        for actor in self.force_actors:
            transform = actor.GetUserTransform()
            # Save individual rotation values instead of orientation
            rot_x, rot_y, rot_z = transform.GetOrientation()
            force_data = {
                'type': 'force',
                'position': list(map(float, transform.GetPosition())),
                'rotation_x': float(rot_x),
                'rotation_y': float(rot_y),
                'rotation_z': float(rot_z),
                'scale': list(map(float, transform.GetScale())),
                'color': list(map(float, actor.GetProperty().GetColor()))
            }
            project.loads.append(force_data)
        
        for actor in self.constraint_actors:
            transform = actor.GetUserTransform()
            constraint_data = {
                'type': 'fixed',
                'position': list(map(float, transform.GetPosition())),
                'scale': list(map(float, transform.GetScale()))
            }
            project.constraints.append(constraint_data)
        
        # Save to file
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "Project Files (*.prj);;All Files (*)",
            options=options
        )
        
        if file_path:
            if not file_path.endswith('.prj'):
                file_path += '.prj'
                
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(project.to_dict(), f, indent=4)
                self.message_text.append(f"\nProject saved successfully to {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project Files (*.prj);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                project = ProjectData.from_dict(data)
                
                # Clear current state
                self.clear_selections()
                
                # Load STL if needed
                if project.stl_file_path and (not self.stl_geom or self.stl_geom.file_path != project.stl_file_path):
                    self.load_stl_file(project.stl_file_path)
                
                # Load settings
                if project.settings:
                    self.settings = project.settings
                
                # Restore selections
                if self.stl_geom and project.selected_triangles:
                    for tri_data in project.selected_triangles:
                        self.stl_geom.tri_highlight[tri_data['index']] = True
                    self.update_highlights()
                
                # Restore loads and constraints
                for load_data in project.loads:
                    if load_data['type'] == 'force':
                        self.restore_force(load_data)
                
                for constraint_data in project.constraints:
                    if constraint_data['type'] == 'fixed':
                        self.restore_constraint(constraint_data)
                
                self.message_text.append(f"\nProject loaded successfully from {file_path}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def restore_force(self, load_data):
        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        arrow.SetShaftRadius(0.03)
        
        transform = vtk.vtkTransform()
        transform.Translate(*load_data['position'])
        
        # Apply rotations in order
        transform.RotateZ(load_data['rotation_z'])
        transform.RotateY(load_data['rotation_y'])
        transform.RotateX(load_data['rotation_x'])
        
        transform.Scale(*load_data['scale'])
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(arrow.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetUserTransform(transform)
        actor.GetProperty().SetColor(*load_data['color'])
        
        self.renderer.AddActor(actor)
        self.force_actors.append(actor)

    def restore_constraint(self, constraint_data):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.05, 0.05, 0.05)
        
        transform = vtk.vtkTransform()
        transform.Translate(constraint_data['position'])
        transform.Scale(constraint_data['scale'])
        
        axes.SetUserTransform(transform)
        self.renderer.AddActor(axes)
        self.constraint_actors.append(axes)

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