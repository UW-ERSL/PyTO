import sys
import os
import pyvista as pv
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from pyvistaqt import QtInteractor
from stl_reader import STLGeom
#---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("PyTO")
        self.resize(1280, 768)

        # Settings object for units
        self.settings = Settings()

        # Central widget and layout
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        # Horizontal layout for PyVista and sidebar
        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.setSpacing(10)
        self.h_layout.setContentsMargins(10, 10, 10, 10)

        # PyVista Frame
        self.pv_frame = QtWidgets.QFrame()
        self.pv_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.pv_layout = QtWidgets.QVBoxLayout(self.pv_frame)
        self.pv_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.pv_frame)
        self.pv_layout.addWidget(self.plotter.interactor)
        self.h_layout.addWidget(self.pv_frame, stretch=4)

        # Add small axes widget in the lower left and set parallel projection
        self.plotter.add_axes(interactive=False)
        self.plotter.set_background('white')
        self.plotter.enable_parallel_projection()

        # Sidebar
        self.sidebar = QtWidgets.QFrame()
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.sidebar.setFixedWidth(250)
        self.sidebar_buttons = {}
        buttons = [
            ("Units", "arrow"),
            ("Geometry", "arrow"),
            ("Material", "cross"),
            ("Structural Loads", "cross"),
            ("Thermal Loads", "cross"),
            ("Body force", "cross"),
            ("Display Options", "arrow"),
            ("Analysis", "cross"),
            ("TopOpt Constraints", "cross"),
            ("Structural TopOpt", "cross"),
            ("Thermal TopOpt", "cross"),
            ("TopOpt Results", "cross"),
            ("Projects", "arrow"),
            ("Help", "arrow")
        ]
        for text, initial_icon in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    color: black;
                    border: 1px solid #CCCCCC;
                    text-align: left;
                    padding: 4px;
                    font-family: 'Segoe UI';
                    font-size: 12pt;
                }
                QPushButton:hover {
                    background-color: #F0F0F0;
                }
            """)
            btn.setIcon(self.get_icon(initial_icon))
            btn.setIconSize(QSize(16, 16))
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons[text] = btn
            # Connect buttons to open dialogs
            if text == "Units":
                btn.clicked.connect(self.open_units_window)
            elif text == "Geometry":
                btn.clicked.connect(self.open_geometry_window)
            elif text == "Material":
                btn.clicked.connect(self.open_material_window)
        sidebar_layout.addStretch()
        self.h_layout.addWidget(self.sidebar, stretch=0)

        # Message Frame
        self.message_frame = QtWidgets.QFrame()
        self.message_frame.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        message_layout = QtWidgets.QVBoxLayout(self.message_frame)
        self.message_text = QtWidgets.QTextEdit()
        self.message_text.setFixedHeight(162)
        self.message_text.setStyleSheet("""
            QTextEdit {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
        """)
        self.message_text.setReadOnly(True)
        self.message_text.setText("Welcome to PyTO!")
        message_layout.addWidget(self.message_text)

        # Add layouts to main layout
        self.main_layout.addLayout(self.h_layout, stretch=4)
        self.main_layout.addWidget(self.message_frame, stretch=1)

        # Status Bar
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
        """)
        version_label = QtWidgets.QLabel("PyTO GUI Version 2025.01. ")
        build_label = QtWidgets.QLabel("GUI Build Date 6.26.2025. ")
        license_label = QtWidgets.QLabel(
            "This is an academic license, and should not be used for commercial purposes."
        )
        license_label.setStyleSheet("color: red;")
        status_bar.addWidget(version_label)
        status_bar.addWidget(build_label)
        status_bar.addWidget(license_label)

        # Highlight/selection system
        self.highlight_actor = None
        self.constraint_actors = []
        self.force_actors = []
        self.highlight_mode = 'coarse'
        self.stl_geom = None

    def get_icon(self, icon_type):
        base_path = os.path.dirname(__file__)
        icon_file = ""
        if icon_type == "arrow":
            icon_file = os.path.join(base_path, "arrow-right.png")
        elif icon_type == "arrow-blue":
            icon_file = os.path.join(base_path, "arrow-blue.png")
        elif icon_type == "cross":
            icon_file = os.path.join(base_path, "cross.png")
        elif icon_type == "check":
            icon_file = os.path.join(base_path, "check.png")
        if not os.path.exists(icon_file):
            return QIcon()
        return QIcon(icon_file)

    def set_sidebar_icon(self, button_text, icon_type):
        """Update the icon for a sidebar button."""
        btn = self.sidebar_buttons.get(button_text)
        if btn:
            btn.setIcon(self.get_icon(icon_type))

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self)
        dialog.exec_()

    def open_material_window(self):
        dialog = MaterialWindow(self)
        dialog.exec_()

    # def open_structural_loads_window(self):
    #     dialog = StructuralLoadsWindow(self)
    #     dialog.exec_()

    # ---------------- Highlight/Selection Functionality ----------------

    def update_highlights(self):
        """Update the highlighted triangles in the PyVista plotter."""
        if not self.stl_geom:
            return

        # Remove previous highlight mesh if present
        if self.highlight_actor:
            try:
                self.plotter.remove_actor(self.highlight_actor, reset_camera=False)
            except Exception:
                pass
            self.highlight_actor = None

        highlight_ids = [
            i for i, h in enumerate(self.stl_geom.tri_highlight)
            if h or (hasattr(self.stl_geom, "selected_triangles") and i in self.stl_geom.selected_triangles)
        ]
        if not highlight_ids:
            self.plotter.render()
            return

        mesh = self.stl_geom.mesh
        highlight_points = mesh.vectors[highlight_ids].reshape(-1, 3)
        highlight_faces_flat = np.arange(len(highlight_points)).reshape(-1, 3)
        highlight_faces_flat = np.column_stack((np.full(len(highlight_faces_flat), 3), highlight_faces_flat)).flatten()

        if len(highlight_points) == 0:
            self.plotter.render()
            return

        highlight_mesh = pv.PolyData(highlight_points, highlight_faces_flat)
        self.highlight_actor = self.plotter.add_mesh(
            highlight_mesh,
            color="yellow",
            opacity=0.5,
            show_edges=True,
            name="highlight",
            reset_camera=False
        )
        self.plotter.render()

    def clear_selections(self):
        """Clear all highlights and constraint/force markers."""
        if self.stl_geom:
            self.stl_geom.tri_highlight = [False] * getattr(self.stl_geom, "stl_n_triangles", 0)
            if hasattr(self.stl_geom, "selected_triangles"):
                self.stl_geom.selected_triangles.clear()
            self.update_highlights()

        for actor in getattr(self, "constraint_actors", []):
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self.constraint_actors = []

        for actor in getattr(self, "force_actors", []):
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self.force_actors = []

        self.plotter.render()
        if hasattr(self, "set_sidebar_icon"):
            self.set_sidebar_icon("Structural Loads", "cross")

    def on_left_button_press(self, picked_point):
        """Handle left click for selecting triangles."""
        if not self.stl_geom:
            return
        mesh = self.stl_geom.mesh
        tri_centers = mesh.vectors.mean(axis=1)
        dists = np.linalg.norm(tri_centers - picked_point, axis=1)
        cell_id = int(np.argmin(dists))
        selection_mode = getattr(self, 'highlight_mode', 'coarse')
        if selection_mode == 'triangle':
            highlighted_count, area = self.stl_geom.highlight_triangles_recursive(
                seed_triangle=cell_id, depth=0, cutoff_angle_degrees=0
            )
            self.update_highlights()
            self.message_text.append(f"Selected triangle {cell_id} with area {area:.6f} square units")
        else:
            highlighted_count, area = self.stl_geom.highlight_triangles_recursive(
                seed_triangle=cell_id, depth=500, cutoff_angle_degrees=15
            )
            self.update_highlights()
            self.message_text.append(f"Selected {highlighted_count} triangles with area {area:.6f} square units")

    def on_right_button_press(self, obj, event):
        """Handle right click for deselecting faces."""
        if not self.stl_geom:
            return
        click_pos = self.plotter.iren.GetEventPosition()
        picked_point = self.plotter.pick_mouse_position(click_pos, return_point=True)
        if picked_point is None:
            return
        mesh = self.stl_geom.mesh
        tri_centers = mesh.vectors.mean(axis=1)
        dists = np.linalg.norm(tri_centers - picked_point, axis=1)
        cell_id = int(np.argmin(dists))
        current_highlights = self.stl_geom.tri_highlight.copy()
        self.stl_geom.tri_highlight = [False] * len(self.stl_geom.tri_highlight)
        _, _ = self.stl_geom.highlight_triangles_recursive(cell_id, depth=500, cutoff_angle_degrees=30)
        faces_to_deselect = [i for i, h in enumerate(self.stl_geom.tri_highlight) if h]
        self.stl_geom.tri_highlight = current_highlights
        for face_id in faces_to_deselect:
            self.stl_geom.tri_highlight[face_id] = False
        self.update_highlights()  
#---------------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.unit_system = "MKS"
        self.temperature_unit = "Kelvin"
        self.angle_unit = "Degree"

        # Add conversion factors for different unit systems
        self.unit_conversions = {
            # Length conversions to/from meters
            "length": {
                "MKS": 1.0,       # meters (base)
                "mmKS": 1000.0,   # millimeters
                "IPS": 39.37      # inches
            },
            # Force conversions to/from Newtons
            "force": {
                "MKS": 1.0,      # Newtons (base)
                "mmKS": 1.0,     # Newtons
                "IPS": 0.2248    # pounds-force
            },
            # Stress conversions to/from Pascal
            "stress": {
                "MKS": 1.0,           # Pascal (base) 
                "mmKS": 1.0e-6,       # MPa
                "IPS": 1.45038e-4     # psi
            }
        }

    def update_settings(self, unit_system, temperature_unit, angle_unit):
        self.unit_system = unit_system
        self.temperature_unit = temperature_unit
        self.angle_unit = angle_unit

    def convert_length(self, value, from_system="MKS", to_system=None):
        """Convert a length value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        # Convert to meters first (base unit)
        value_in_meters = value / self.unit_conversions["length"][from_system]
        
        # Then convert to target unit
        return value_in_meters * self.unit_conversions["length"][to_system]
    
    def convert_force(self, value, from_system="MKS", to_system=None):
        """Convert a force value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        # Convert to newtons first (base unit)
        value_in_newtons = value / self.unit_conversions["force"][from_system]
        
        # Then convert to target unit
        return value_in_newtons * self.unit_conversions["force"][to_system]
    
    def convert_stress(self, value, from_system="MKS", to_system=None):
        """Convert a stress value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        # Convert to Pascal first (base unit)
        value_in_pascal = value / self.unit_conversions["stress"][from_system]
        
        # Then convert to target unit
        return value_in_pascal * self.unit_conversions["stress"][to_system]

    def get_length_unit_string(self):
        if self.unit_system == "MKS":
            return "m"
        elif self.unit_system == "mmKS":
            return "mm"
        elif self.unit_system == "IPS":
            return "in"
        return "m"
    
    def get_force_unit_string(self):
        """Return the appropriate force unit string"""
        if self.unit_system == "MKS" or self.unit_system == "mmKS":
            return "N"
        elif self.unit_system == "IPS":
            return "lbf"
        return "N"  # Default
    
    def get_stress_unit_string(self):
        """Return the appropriate stress unit string"""
        if self.unit_system == "MKS":
            return "Pa"
        elif self.unit_system == "mmKS":
            return "MPa"
        elif self.unit_system == "IPS":
            return "psi"
        return "Pa"  # Default

    def get_temperature_unit_symbol(self):
        if self.temperature_unit == "Celsius":
            return "°C"
        elif self.temperature_unit == "Fahrenheit":
            return "°F"
        else:
            return "K"
#---------------------------------------------------------------------------
class UnitsWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Units")
        self.resize(220, 180)
        self.settings = settings

        layout = QtWidgets.QVBoxLayout(self)

        # Unit system
        layout.addWidget(QtWidgets.QLabel("Unit System:"))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["MKS", "mmKS", "IPS"])
        self.unit_combo.setCurrentText(settings.unit_system)
        layout.addWidget(self.unit_combo)

        # Temperature unit
        layout.addWidget(QtWidgets.QLabel("Temperature Unit:"))
        self.temp_combo = QtWidgets.QComboBox()
        self.temp_combo.addItems(["Kelvin", "Celsius", "Fahrenheit"])
        self.temp_combo.setCurrentText(settings.temperature_unit)
        layout.addWidget(self.temp_combo)

        # Angle unit
        layout.addWidget(QtWidgets.QLabel("Angle Unit:"))
        self.angle_combo = QtWidgets.QComboBox()
        self.angle_combo.addItems(["Degree", "Radian"])
        self.angle_combo.setCurrentText(settings.angle_unit)
        layout.addWidget(self.angle_combo)

        # Apply button
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)

    def apply_settings(self):
        self.settings.update_settings(
            self.unit_combo.currentText(),
            self.temp_combo.currentText(),
            self.angle_combo.currentText()
        )
        if hasattr(self.parent(), "set_sidebar_icon"):
            self.parent().set_sidebar_icon("Units", "check")
        if hasattr(self.parent(), "set_sidebar_active"):
            self.parent().set_sidebar_active("Geometry")
        self.accept()
#---------------------------------------------------------------------------
class GeometryWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Geometry")
        self.resize(200, 100)
        self.parent = parent  # MainWindow instance

        layout = QtWidgets.QVBoxLayout(self)

        self.info_label = QtWidgets.QLabel("No geometry loaded.")
        layout.addWidget(self.info_label)

        load_btn = QtWidgets.QPushButton("Load STL Geometry")
        load_btn.clicked.connect(self.load_geometry)
        layout.addWidget(load_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

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
                self.stl_geom = STLGeom(file_path)
                self.stl_geom.plotGeometry(show_edges=False, show_axes=False, show_bounding_box=False, plotter=self.parent.plotter)
                area, volume, _, _ = self.stl_geom.compute_mass_properties()
                bounds = self.stl_geom.get_bounding_box()
                length_unit = self.parent.settings.get_length_unit_string() if hasattr(self.parent, "settings") else "m"

                info_lines = [
                    f"Model: {os.path.basename(file_path)}",
                    f"Volume: {volume:.2e} {length_unit}³",
                    f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                    f"Surface Area: {area:.2e} {length_unit}²"
                ]

                self.parent.plotter.add_text(
                    "\n".join(info_lines),
                    position="upper_left",
                    font_size=12,
                    color="black",
                    name="geometry_info",
                    font="arial",
                )
                if hasattr(self.parent, "set_sidebar_icon"):
                    self.parent.set_sidebar_icon("Geometry", "check")
                    self.parent.set_sidebar_icon("Material", "arrow")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load geometry: {str(e)}")
#---------------------------------------------------------------------------
class MaterialWindow(QtWidgets.QDialog):
    """Dialog for selecting and editing material properties."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Material")
        self.setFixedSize(300, 400)
        self.parent = parent 

        self.materials = {
            "Steel": {
                "Young's Modulus": 2.1e11,
                "Poisson's Ratio": 0.3,
                "Yield Strength": 2.5e8,
                "Density": 7850,
                "Thermal Conductivity": 50,
                "Thermal Expansion": 1.2e-5,
                "Specific Heat Capacity": 500,
                "Price": 0.8
            },
            "Aluminum": {
                "Young's Modulus": 7.0e10,
                "Poisson's Ratio": 0.33,
                "Yield Strength": 2.7e8,
                "Density": 2700,
                "Thermal Conductivity": 237,
                "Thermal Expansion": 2.3e-5,
                "Specific Heat Capacity": 900,
                "Price": 2.0
            },
            "Titanium": {
                "Young's Modulus": 1.16e11,
                "Poisson's Ratio": 0.34,
                "Yield Strength": 8.8e8,
                "Density": 4500,
                "Thermal Conductivity": 21.9,
                "Thermal Expansion": 8.6e-6,
                "Specific Heat Capacity": 520,
                "Price": 30.0
            },
            "Custom": {
                "Young's Modulus": 0.0,
                "Poisson's Ratio": 0.0,
                "Yield Strength": 0.0,
                "Density": 0.0,
                "Thermal Conductivity": 0.0,
                "Thermal Expansion": 0.0,
                "Specific Heat Capacity": 0.0,
                "Price": 0.0
            }
        }

        self.material_applied = False

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        form_layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # Material name dropdown
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(self.materials.keys())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        form_layout.addRow("Material Name:", self.material_combo)

        # Create fields
        self.fields = {}

        self.fields["Young's Modulus"] = QtWidgets.QLineEdit()
        form_layout.addRow(self._label_with_unit("Young's Modulus"), self.fields["Young's Modulus"])

        self.fields["Poisson's Ratio"] = QtWidgets.QLineEdit()
        form_layout.addRow("Poisson's Ratio (-):", self.fields["Poisson's Ratio"])

        self.fields["Yield Strength"] = QtWidgets.QLineEdit()
        form_layout.addRow(self._label_with_unit("Yield Strength"), self.fields["Yield Strength"])

        self.fields["Density"] = QtWidgets.QLineEdit()
        form_layout.addRow(self._label_with_unit("Density"), self.fields["Density"])

        self.fields["Thermal Conductivity"] = QtWidgets.QLineEdit()
        form_layout.addRow("Thermal Conductivity (W/m-K):", self.fields["Thermal Conductivity"])

        self.fields["Thermal Expansion"] = QtWidgets.QLineEdit()
        form_layout.addRow("Thermal Expansion (m/m-K):", self.fields["Thermal Expansion"])

        self.fields["Specific Heat Capacity"] = QtWidgets.QLineEdit()
        form_layout.addRow("Spec. Heat Capacity (J/kg-K):", self.fields["Specific Heat Capacity"])

        self.fields["Price"] = QtWidgets.QLineEdit()
        form_layout.addRow("Price (US$/kg):", self.fields["Price"])

        for field in self.fields.values():
            field.setMaximumWidth(120)

        layout.addLayout(form_layout)

        # Checkbox and buttons
        self.optimize_check = QtWidgets.QCheckBox("Do not optimize")
        layout.addWidget(self.optimize_check)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_material)
        layout.addWidget(self.apply_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.on_material_changed(self.material_combo.currentText())

    def _label_with_unit(self, key):
        """Return label with dynamic unit for key."""
        s = self.parent.settings
        if key == "Young's Modulus" or key == "Yield Strength":
            return f"{key} ({s.get_stress_unit_string()}):"
        elif key == "Density":
            return f"{key} (kg/{s.get_length_unit_string()}³):"
        return key

    def on_material_changed(self, name):
        mat = self.materials[name]
        for key, field in self.fields.items():
            field.setText(str(mat[key]))
            field.setReadOnly(name != "Custom")
        if self.material_applied:
            self.material_combo.setEnabled(False)
            for field in self.fields.values():
                field.setReadOnly(True)
            self.apply_btn.setEnabled(False)

    def apply_material(self):
        name = self.material_combo.currentText()
        if name == "Custom":
            # Save custom values
            for key in self.fields:
                try:
                    val = float(self.fields[key].text())
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid value for {key}")
                    return
                self.materials["Custom"][key] = val
        self.material_applied = True
        self.on_material_changed(name)
        if hasattr(self.parent, "set_sidebar_icon"):
            self.parent.set_sidebar_icon("Material", "check")
        

    def update_units(self):
        # Update only the labels that are dynamic
        form_layout = self.layout().itemAt(0).layout()
        form_layout.labelForField(self.fields["Young's Modulus"]).setText(self._label_with_unit("Young's Modulus"))
        form_layout.labelForField(self.fields["Yield Strength"]).setText(self._label_with_unit("Yield Strength"))
        form_layout.labelForField(self.fields["Density"]).setText(self._label_with_unit("Density"))

    def showEvent(self, event):
        super().showEvent(event)
        self.update_units()

#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())