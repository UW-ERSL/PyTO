import sys
import os
import pyvista as pv
import numpy as np
import math
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from pyvistaqt import QtInteractor
from stl_reader import STLGeom
import bound_cond
import mat_lib
import linear_solvers
from hex_mesher import HexMesher
import hex_structural_fea
"""
1) Need to Implement Analysis Windows and downstream
"""
#---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("PyTO")
        self.resize(1280, 768)

        #Settings object for units
        self.settings = Settings()

        #Live Variable to track UI state
        self.LivVar = {
            "geometry_loaded": False,
            "material_defined": False,
            "structural_loads": {
                "applied": False,
                "fixed_constraints": False,
                "forces_applied": False
            },
            "thermal_loads": {
                "applied": False,
                "heat_sources": False,
                "convection_applied": False
            },
            "mesh_generated": False,
            "analysis": {
                "performed": False,
                "structural": False,
                "thermal": False
            },
            "topopt": {
                "constraints_defined": False,
                "structural_performed": False,
                "thermal_performed": False
            },
            "current_step": "init"  # Tracks the current workflow step
        }

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
        self.picker = pv._vtk.vtkCellPicker()

        #XYZ axes and parallel projection
        self.plotter.add_axes(interactive=False)
        self.plotter.set_background('white')
        self.plotter.enable_parallel_projection()

        # init 
        self.constrained_triangles = set()  # Track which triangles have constraints
        self.constraint_actors = []  # Store constraint visualization actors
        self.constraint_data = []  # Initialize constraint data list
        self.force_data = []       # Initialize force data list
        self.material_data = None

        # Sidebar
        self.sidebar = QtWidgets.QFrame()
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.sidebar.setFixedWidth(250)
        self.sidebar_buttons = {}
        
        # Define buttons with initial icons based on workflow state
        buttons = [
            ("Units", "arrow"),  # Always available
            ("Geometry", "arrow"),  # Always available to start
            ("Material", "cross"),  # Not available until geometry is loaded
            ("Structural Loads", "cross"),  # Not available until material is defined
            ("Thermal Loads", "cross"),  # Not available until material is defined
            ("Body force", "cross"),  # Not available initially
            ("Display Options", "arrow"),  # Always available
            ("Analysis", "cross"),  # Not available until loads are applied
            ("TopOpt Constraints", "cross"),  # Not available until loads are applied
            ("Structural TopOpt", "cross"),  # Not available until constraints defined
            ("Thermal TopOpt", "cross"),  # Not available until constraints defined
            ("TopOpt Results", "cross"),  # Not available until optimization is done
            ("Projects", "arrow"),  # Always available
            ("Help", "arrow")  # Always available
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

             # Connect buttons to click handler
            btn.clicked.connect(lambda checked, name=text: self.sidebar_button_clicked(name))
            
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

    def update_LivVar(self, key, value=True):
        """
        Update the LivVar state tracker and reflect changes in UI
        
        Parameters:
        -----------
        key : str
            Path to the key in LivVar, using dot notation (e.g., 'structural_loads.applied')
        value : any
            Value to set (default: True)
        """
        # Parse the key path
        keys = key.split('.')
        target = self.LivVar
        
        # Navigate to nested dict location
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
        
        #Update current workflow step based on the change
        if key == 'geometry_loaded' and value:
            self.LivVar['current_step'] = 'geometry_loaded'
            #Update availability of next steps
            self.set_sidebar_icon("Material", "arrow")
            self.set_sidebar_icon("Structural Loads", "arrow")
            self.set_sidebar_icon("Thermal Loads", "arrow")
            self.message_text.append("Geometry loaded. You can now define material properties.")
        elif key == 'material_defined' and value:
            self.LivVar['current_step'] = 'material_defined'
            #Material is complete, loads can be defined
            self.set_sidebar_icon("Material", "check")
            self.message_text.append("Material defined. You can now apply structural or thermal loads.")
        elif key in ['structural_loads.forces_applied', 'structural_loads.fixed_constraints'] and value:
            #Check if BOTH forces and constraints are applied
            if self.LivVar['structural_loads']['forces_applied'] and self.LivVar['structural_loads']['fixed_constraints']:
                self.LivVar['structural_loads']['applied'] = True
                self.LivVar['current_step'] = 'loads_applied'
                self.set_sidebar_icon("Structural Loads", "check")
                self.set_sidebar_icon("Analysis", "arrow")
                self.set_sidebar_icon("TopOpt Constraints", "arrow")
                self.message_text.append("Structural loads complete (forces and constraints applied). You can now run analysis or define TopOpt constraints.")
            else:
                # Only one component applied so far
                missing = []
                if not self.LivVar['structural_loads']['forces_applied']:
                    missing.append("forces")
                if not self.LivVar['structural_loads']['fixed_constraints']:
                    missing.append("constraints")
                self.message_text.append(f"Structural loads partially applied. Still need: {', '.join(missing)}")
        elif key == 'thermal_loads.applied' and value:
            self.LivVar['current_step'] = 'loads_applied'
            self.set_sidebar_icon("Thermal Loads", "check")
            self.set_sidebar_icon("Analysis", "arrow")
            # Enable TopOpt Constraints immediately after thermal loads are applied
            self.set_sidebar_icon("TopOpt Constraints", "arrow")
            self.message_text.append("Thermal loads applied. You can now run analysis or define TopOpt constraints.")
        elif key == 'mesh_generated' and value:
            self.LivVar['current_step'] = 'mesh_generated'
        elif key == 'analysis.performed' and value:
            self.LivVar['current_step'] = 'analysis_performed'
            self.set_sidebar_icon("Analysis", "check")
        elif key == 'topopt.constraints_defined' and value:
            self.LivVar['current_step'] = 'topopt_ready'
            self.set_sidebar_icon("TopOpt Constraints", "check")
            self.set_sidebar_icon("Structural TopOpt", "arrow")
            self.set_sidebar_icon("Thermal TopOpt", "arrow")
        elif key == 'topopt.structural_performed' and value:
            self.set_sidebar_icon("Structural TopOpt", "check")
        elif key == 'topopt.thermal_performed' and value:
            self.set_sidebar_icon("Thermal TopOpt", "check")

    def check_workflow_readiness(self, target_step):
        """
        Check if the workflow is ready for a specific step
        
        Parameters:
        -----------
        target_step : str
            The step to check readiness for
            
        Returns:
        --------
        bool, str
            Whether the workflow is ready, and a message explaining why if not
        """
        if target_step == 'material' and not self.LivVar['geometry_loaded']:
            return False, "Please load geometry first before defining material properties."
            
        elif target_step == 'loads' and not self.LivVar['material_defined']:
            return False, "Please define material properties first before applying loads."
            
        elif target_step == 'analysis':
            # Check if either structural OR thermal loads have been applied
            loads_applied = (self.LivVar['structural_loads']['applied'] or 
                            self.LivVar['thermal_loads']['applied'])
            if not loads_applied:
                return False, "Please apply structural or thermal loads first before running analysis."
        
        # Change this to allow TopOpt Constraints after loads are applied
        elif target_step == 'topopt_constraints':
            # Check if either structural OR thermal loads have been applied
            loads_applied = (self.LivVar['structural_loads']['applied'] or 
                            self.LivVar['thermal_loads']['applied'])
            if not loads_applied:
                return False, "Please apply structural or thermal loads first before defining TopOpt constraints."
            
        elif target_step == 'topopt' and not self.LivVar['topopt']['constraints_defined']:
            return False, "Please define TopOpt constraints first before running topology optimization."
            
        return True, "Ready"

    def sidebar_button_clicked(self, name):
        """Handle sidebar button clicks with workflow validation"""
        # Check if this button should be available based on workflow state
        ready, message = self.check_workflow_readiness_for_button(name)
        
        if not ready:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)
            return
            
        # Route the button action based on its name
        if name == "Units":
            self.open_units_window()
        elif name == "Geometry":
            self.open_geometry_window()
        elif name == "Material":
            self.open_material_window()
        elif name == "Structural Loads":
            self.open_structural_loads_window()
        elif name == "Thermal Loads":
            # Add thermal loads window when implemented
            QtWidgets.QMessageBox.information(self, "Coming Soon", "Thermal Loads window will be implemented soon.")
        elif name == "Display Options":
            QtWidgets.QMessageBox.information(self, "Coming Soon", "Display Options window will be implemented soon.")
        elif name == "Analysis":
           self.open_analysis_window()
        elif name == "TopOpt Constraints":
            QtWidgets.QMessageBox.information(self, "Coming Soon", "TopOpt Constraints window will be implemented soon.")
        elif name == "Structural TopOpt":
            QtWidgets.QMessageBox.information(self, "Coming Soon", "Structural TopOpt window will be implemented soon.")
        elif name == "Thermal TopOpt":
            QtWidgets.QMessageBox.information(self, "Coming Soon", "Thermal TopOpt window will be implemented soon.")
        elif name == "Projects":
            QtWidgets.QMessageBox.information(self, "Coming Soon", "Projects window will be implemented soon.")
        elif name == "Help":
            QtWidgets.QMessageBox.information(self, "Help", "PyTO Help documentation will be available soon.")

    def check_workflow_readiness_for_button(self, button_name):
        """
        Check if the workflow is ready for a specific button to be used
        
        Parameters:
        -----------
        button_name : str
            The button name to check readiness for
                
        Returns:
        --------
        bool, str
            Whether the button can be used, and a message explaining why if not
        """
        # Map buttons to their workflow step requirements
        step_map = {
            "Material": "material",
            "Structural Loads": "loads",
            "Thermal Loads": "loads", 
            "Analysis": "analysis",
            "TopOpt Constraints": "topopt_constraints",
            "Structural TopOpt": "topopt",
            "Thermal TopOpt": "topopt"
        }
        
        # Always available buttons
        if button_name not in step_map:
            return True, "Ready"
            
        # Check readiness for the mapped step
        return self.check_workflow_readiness(step_map[button_name])

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

    def update_highlights(self):
        if not self.stl_geom:
            return

        # Store current camera position before any operations
        camera_position = self.plotter.camera_position

        # Remove old highlight
        if self.highlight_actor:
            try:
                self.plotter.remove_actor(self.highlight_actor, reset_camera=False)
            except:
                pass
            self.highlight_actor = None

        # Get currently highlighted triangles
        highlight_ids = []
        for i, h in enumerate(self.stl_geom.tri_highlight):
            if h and i not in self.constrained_triangles:  # Don't highlight constrained triangles
                highlight_ids.append(i)
        
        if not highlight_ids:
            # Restore camera position and render
            self.plotter.camera_position = camera_position
            self.plotter.render()
            return

        # Collect vertices and faces from highlighted triangles
        vertices = []
        faces = []
        vertex_count = 0
        
        for tri_id in highlight_ids:
            # Get triangle vertices from mesh.vectors
            triangle_vertices = self.stl_geom.mesh.vectors[tri_id]
            
            # Add vertices to list
            for vertex in triangle_vertices:
                vertices.append(vertex)
            
            # Create face (triangle with 3 vertices)
            face = [3, vertex_count, vertex_count + 1, vertex_count + 2]
            faces.extend(face)
            vertex_count += 3
        
        # Convert to numpy arrays
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Create PyVista mesh
        highlight_mesh = pv.PolyData(vertices, faces).compute_normals(cell_normals=True, point_normals=True)
        
        # Calculate offset
        bounds = self.plotter.bounds
        offset = 0.001 * ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5 if bounds else 0.001
        
        # Create double-sided mesh
        front_mesh = highlight_mesh.copy()
        front_mesh.points = front_mesh.points + front_mesh.point_data['Normals'] * offset
        back_mesh = highlight_mesh.copy()
        back_mesh.points = back_mesh.points - back_mesh.point_data['Normals'] * offset
        
        # Add highlight with reset_camera=False
        self.highlight_actor = self.plotter.add_mesh(
            front_mesh + back_mesh,
            color="red",
            opacity=0.6,
            culling=False,
            lighting=False,
            reset_camera=False
        )
        
        # Restore camera position and render
        self.plotter.camera_position = camera_position
        self.plotter.render()

    def on_left_button_press(self, picked_point, picker=None):
        """Handle left click for selecting triangles."""
        if not picker or not self.stl_geom:
            return
        
        cell_id = picker.GetCellId()
        if cell_id < 0:
            return
        
        # Check if clicked triangle is constrained or has force applied
        if cell_id in self.constrained_triangles:
            self.message_text.append("Cannot select constrained triangle.")
            return
        
        mode = getattr(self, 'highlight_mode', 'coarse')
        depth, angle = (0, 0) if mode == 'triangle' else (500, 15)
        
        # Temporarily store current highlights to check for constrained/force-applied triangles
        old_highlights = self.stl_geom.tri_highlight.copy()
        count, area = self.stl_geom.highlight_triangles_recursive(cell_id, depth, angle)
        
        # Check if any of the newly selected triangles are constrained or have forces applied
        newly_selected = []
        for i, (old, new) in enumerate(zip(old_highlights, self.stl_geom.tri_highlight)):
            if not old and new:  # Newly selected triangle
                if i in self.constrained_triangles:
                    self.stl_geom.tri_highlight[i] = False  # Deselect constrained triangle
                    count -= 1
                else:
                    newly_selected.append(i)
        
        self.update_highlights()
        
        if newly_selected:
            self.message_text.append(f"Selected {count} triangle{'s' if count > 1 else ''} with area {area:.6f} square units")
        else:
            self.message_text.append("Cannot select constrained or force-applied triangles.")

    def on_right_button_press(self, obj, event):
        """Handle right click for deselecting triangles - deselect all selected triangles."""
        try:
            # Count currently selected triangles
            selected_count = sum(1 for h in self.stl_geom.tri_highlight if h)
            
            if selected_count == 0:
                self.message_text.append("No triangles selected to deselect.")
                return
            
            # Deselect ALL highlighted triangles
            for i in range(len(self.stl_geom.tri_highlight)):
                self.stl_geom.tri_highlight[i] = False
            
            # Update visualization
            self.update_highlights()
            
            # Show message
            self.message_text.append(f"Deselected all {selected_count} selected triangles.")
            
        except Exception as e:
            print(f"Error in deselection: {str(e)}")

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self)
        dialog.exec_()

    def open_material_window(self):
        dialog = MaterialWindow(self)
        dialog.exec_()

    def open_structural_loads_window(self):
        self.structural_loads_window = StructuralLoadsWindow(self)
        self.structural_loads_window.show()

    # def open_analysis_window(self):
    #     """Open the analysis window"""
    #     dialog = AnalysisWindow(self)
    #     dialog.exec_()
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
        
        # Add message about units being applied
        if hasattr(self.parent(), "message_text"):
            self.parent().message_text.append(f"Units defined: {self.unit_combo.currentText()}, " + 
                                      f"{self.temp_combo.currentText()}, {self.angle_combo.currentText()}")
        
        self.accept()
#---------------------------------------------------------------------------
class GeometryWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Geometry")
        self.resize(200, 120)  
        self.parent = parent 

        layout = QtWidgets.QVBoxLayout(self)

        self.info_label = QtWidgets.QLabel("No geometry loaded.")
        layout.addWidget(self.info_label)

        load_btn = QtWidgets.QPushButton("Load STL Geometry")
        load_btn.clicked.connect(self.load_geometry)
        layout.addWidget(load_btn)

        # Add Update Geometry button
        self.update_btn = QtWidgets.QPushButton("Update Geometry")
        self.update_btn.clicked.connect(self.update_geometry)
        self.update_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.update_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Update button state based on current geometry
        self.update_button_state()

    def update_button_state(self):
        """Enable/disable Update Geometry button based on whether geometry is loaded"""
        has_geometry = self.parent.stl_geom is not None
        self.update_btn.setEnabled(has_geometry)
        
        if has_geometry:
            self.info_label.setText("Geometry loaded successfully.")
        else:
            self.info_label.setText("No geometry loaded.")

    def load_geometry(self):
        """Load STL geometry"""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select STL File",
            "",
            "STL Files (*.stl *.STL);;All Files (*)",
            options=options
        )
        if file_path:
            # Store camera position
            camera_pos = self.parent.plotter.camera_position
            
            
            #self.parent.plotter.clear_actors()
    
            # Create new STL geometry
            self.stl_geom = STLGeom(file_path)
            
            # Plot geometry
            self.stl_geom.plotGeometry(
                show_edges=False, 
                show_axes=False, 
                show_bounding_box=False, 
                plotter=self.parent.plotter
            )
            
            # Calculate properties
            area, volume, _, _ = self.stl_geom.compute_mass_properties()
            bounds = self.stl_geom.get_bounding_box()
            length_unit = self.parent.settings.get_length_unit_string()

            # Add geometry info text
            info_lines = [
                f"Model: {os.path.basename(file_path)}",
                f"Volume: {volume:.2e} {length_unit}³",
                f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                f"Surface Area: {area:.2e} {length_unit}²"
            ]

            # Remove old geometry info text
            self.parent.plotter.remove_actor("geometry_info")

            self.parent.plotter.add_text(
                "\n".join(info_lines),
                position="upper_left",
                font_size=12,
                color="black",
                name="geometry_info",
                font="arial",
            )

            # Set geometry reference
            self.parent.stl_geom = self.stl_geom

            # Enable picking
            self.parent.plotter.disable_picking()
            
            self.parent.plotter.enable_point_picking(
                callback=self.parent.on_left_button_press,
                use_picker=True,
                picker='cell',
                show_message=False,
                left_clicking=True,
                show_point=False
            )
            
            # Add right-click observer
            self.parent.plotter.iren.add_observer("RightButtonPressEvent", self.parent.on_right_button_press)
            
            # Update LivVar
            self.parent.update_LivVar('geometry_loaded', True)
            
            # Update sidebar icons
            self.parent.set_sidebar_icon("Geometry", "check")
            self.parent.set_sidebar_icon("Material", "arrow")

            # Update button state
            self.update_button_state()
            
            # Success message
            self.parent.message_text.append(f"Geometry loaded: {os.path.basename(file_path)}")

    def update_geometry(self):
        """Update geometry
        1) to update the geometry need to clear the existing geometry
        2) then load the new geometry"""

        # Clear the plotter completely
        self.parent.plotter.clear_actors()

        self.parent.plotter.reset_camera()

        # Clear existing geometry reference
        #self.parent.stl_geom = None

        # Disable picking
        self.parent.plotter.disable_picking()

        self.update_button_state()  # Disable update button

        self.load_geometry()
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

        # Update LivVar to indicate material is defined
        self.parent.update_LivVar('material_defined', True)
        
        self.parent.message_text.append(f"Material '{name}' applied successfully.")

        self.parent.set_sidebar_icon("Material", "check")
        self.parent.set_sidebar_icon("Structural Loads", "arrow")

        self.close()
        

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
class StructuralLoadsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowModality(QtCore.Qt.NonModal) 
        self.setWindowTitle("Structural Loads")
        self.parent = parent

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Load Set spinbox
        load_set_layout = QtWidgets.QHBoxLayout()
        load_set_label = QtWidgets.QLabel("Load Set")
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        load_set_layout.addWidget(load_set_label)
        load_set_layout.addWidget(self.load_set_spin)
        layout.addLayout(load_set_layout)

        # Selection dropdown
        selection_layout = QtWidgets.QHBoxLayout()
        selection_label = QtWidgets.QLabel("Selection")
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(["Coarse Cylinder", "Triangle"])
        self.selection_combo.currentIndexChanged.connect(self.selection_mode_changed)
        selection_layout.addWidget(selection_label)
        selection_layout.addWidget(self.selection_combo)
        layout.addLayout(selection_layout)

        # Load Type dropdown"
        load_type_layout = QtWidgets.QHBoxLayout()
        load_type_label = QtWidgets.QLabel("Load Type")
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems(["Force", "Torque", "Fixed XYZ", "Fixed X", "Fixed Y", "Fixed Z"])
        load_type_layout.addWidget(load_type_label)
        load_type_layout.addWidget(self.load_type)
        layout.addLayout(load_type_layout)

        # Force components
        self.force_group = QtWidgets.QGroupBox("Force Components")
        force_layout = QtWidgets.QVBoxLayout(self.force_group)
        self.x_force_spin = QtWidgets.QDoubleSpinBox()
        self.x_force_spin.setRange(-1e6, 1e6)
        self.x_force_spin.setDecimals(1)
        self.x_force_spin.setPrefix("X: ")
        self.y_force_spin = QtWidgets.QDoubleSpinBox()
        self.y_force_spin.setRange(-1e6, 1e6)
        self.y_force_spin.setDecimals(1)
        self.y_force_spin.setPrefix("Y: ")
        self.z_force_spin = QtWidgets.QDoubleSpinBox()
        self.z_force_spin.setRange(-1e6, 1e6)
        self.z_force_spin.setDecimals(1)
        self.z_force_spin.setPrefix("Z: ")
        force_layout.addWidget(self.x_force_spin)
        force_layout.addWidget(self.y_force_spin)
        force_layout.addWidget(self.z_force_spin)
        layout.addWidget(self.force_group)

        # Torque components
        self.torque_group = QtWidgets.QGroupBox("Torque")
        torque_layout = QtWidgets.QVBoxLayout(self.torque_group)
        self.torque_magnitude_spin = QtWidgets.QDoubleSpinBox()
        self.torque_magnitude_spin.setRange(-1e6, 1e6)
        self.torque_magnitude_spin.setDecimals(1)
        self.torque_magnitude_spin.setPrefix("Magnitude: ")
        torque_layout.addWidget(self.torque_magnitude_spin)
        self.torque_group.hide()
        layout.addWidget(self.torque_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply")
        # Connect button dynamically based on load type
        self.connect_apply_button()
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        # Connect load type change to update button connection
        self.load_type.currentTextChanged.connect(self.on_load_type_changed)
        self.selection_mode_changed(0)

    def connect_apply_button(self):
        """Connect the apply button to the appropriate method based on current load type"""
        # Disconnect any existing connections
        try:
            self.apply_button.clicked.disconnect()
        except:
            pass
        
        # Connect to appropriate method based on current load type
        load_type = self.load_type.currentText()
        if load_type == "Force":
            self.apply_button.clicked.connect(self.apply_force)
        elif load_type == "Torque":
            self.apply_button.clicked.connect(self.apply_torque)
        elif load_type == "Fixed XYZ":
            self.apply_button.clicked.connect(self.apply_fixed_constraint)
        elif load_type == "Fixed X":
            self.apply_button.clicked.connect(self.apply_fixed_constraint_x)
        elif load_type == "Fixed Y":
            self.apply_button.clicked.connect(self.apply_fixed_constraint_y)
        elif load_type == "Fixed Z":
            self.apply_button.clicked.connect(self.apply_fixed_constraint_z)

    def selection_mode_changed(self, index):
        """Handle changes in the selection mode."""
        selection_mode = self.selection_combo.currentText()
        if selection_mode == "Triangle":
            self.parent.highlight_mode = "triangle"
        else:
            self.parent.highlight_mode = "coarse"

    def on_load_type_changed(self, load_type):
        """Show/hide force or torque input fields and reconnect apply button."""
        show_force = load_type == "Force"
        show_torque = load_type == "Torque"
        self.force_group.setVisible(show_force)
        self.torque_group.setVisible(show_torque)
        
        # Reconnect the apply button to the appropriate method
        self.connect_apply_button()
        
        self.adjustSize()

    def apply_force(self):
        """Apply forces in X, Y, and Z directions"""
            
        # Get selected triangles
        selected_triangles_data = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles_data:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for force application.")
            return
        
        try:
            # Get force values from spinboxes
            fx = self.x_force_spin.value()
            fy = self.y_force_spin.value()
            fz = self.z_force_spin.value()
            
            # Check if at least one force component is non-zero
            if fx == 0 and fy == 0 and fz == 0:
                QtWidgets.QMessageBox.warning(self, "Zero Force", "Please enter at least one non-zero force component.")
                return
            
            # Convert forces to base units
            current_unit_system = self.parent.settings.unit_system
            if current_unit_system != "MKS":
                fx = self.parent.settings.convert_force(fx, current_unit_system, "MKS")
                fy = self.parent.settings.convert_force(fy, current_unit_system, "MKS")
                fz = self.parent.settings.convert_force(fz, current_unit_system, "MKS")
            
            # Apply forces for each non-zero component
            applied_forces = []
            if fx != 0:
                self.visualize_x_force_arrows(selected_triangles_data, fx)
                applied_forces.append(f"X: {fx:+.1f}")
            
            if fy != 0:
                self.visualize_y_force_arrows(selected_triangles_data, fy)
                applied_forces.append(f"Y: {fy:+.1f}")
            
            if fz != 0:
                self.visualize_z_force_arrows(selected_triangles_data, fz)
                applied_forces.append(f"Z: {fz:+.1f}")
            
            # Store force data
            force_info = {
                'triangles': [tri_data['index'] for tri_data in selected_triangles_data],
                'triangle_data': selected_triangles_data,
                'force_x': fx,
                'force_y': fy,
                'force_z': fz,
                'load_set': self.load_set_spin.value(),
                'type': 'force_xyz'
            }
            
            self.parent.force_data.append(force_info)
            
            # Clear highlights
            for tri_data in selected_triangles_data:
                self.parent.stl_geom.tri_highlight[tri_data['index']] = False
            
            self.parent.update_highlights()
            
            # Get force unit
            force_unit = self.parent.settings.get_force_unit_string()
            
            # Create message showing which forces were applied
            force_components = ", ".join(applied_forces)
            self.parent.message_text.append(
                f"Applied forces ({force_components}) {force_unit} to {len(selected_triangles_data)} triangles."
            )

            # Update LivVar to indicate forces are applied
            self.parent.update_LivVar('structural_loads.forces_applied', True)
            
            # AUTO-RESET: Clear all spinboxes after successful application
            self.x_force_spin.setValue(0)
            self.y_force_spin.setValue(0)
            self.z_force_spin.setValue(0)
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to apply force:\n{str(e)}")

    def visualize_arrows(self, selected_triangles_data, force_value, direction_vector, color, axis_name):
        """create directional arrow visualization for positive/negative forces"""
        
        # Threshold system
        MAX_MARKERS = 5
        THRESHOLD = 25
        
        if len(selected_triangles_data) > THRESHOLD:
            step = len(selected_triangles_data) // MAX_MARKERS
            display_indices = range(0, len(selected_triangles_data), step)[:MAX_MARKERS]
            display_triangles = [selected_triangles_data[i] for i in display_indices]
        else:
            display_triangles = selected_triangles_data
        
        # Calculate arrow size based on model
        bounds = self.parent.plotter.bounds
        if bounds:
            model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            arrow_scale = model_size * 0.1
            geom_size = model_size  # For text positioning
        else:
            arrow_scale = 1.0
            geom_size = 1.0
        
        # Handle positive/negative force direction
        actual_direction = direction_vector * (1.0 if force_value >= 0 else -1.0)
        
        # Get triangle centers
        centers = []
        for tri_data in display_triangles:
            centers.append(tri_data['center'])
        
        if centers:
            centers = np.array(centers)
            
            # Create arrows in the actual direction (positive or negative)
            directions = np.tile(actual_direction, (len(centers), 1))
            scales = np.full(len(centers), arrow_scale)
            
            # Create arrows using PyVista
            arrows = pv.PolyData(centers)
            arrows['vectors'] = directions * scales.reshape(-1, 1)
            
            # Create arrow glyphs
            arrow_glyph = arrows.glyph(orient='vectors', scale='vectors', factor=1.0)
            
            # Add to plotter
            force_actor = self.parent.plotter.add_mesh(
                arrow_glyph,
                color=color,
                show_edges=False,
                name=f'{axis_name.lower()}_force_arrows_{len(self.parent.force_actors)}'
            )
            
            # Store actor (force_actors is initialized in MainWindow.__init__)
            self.parent.force_actors.append(force_actor)
            
            # Add text label
            if display_triangles:
                first_triangle = display_triangles[0]
                
                # Get force unit
                force_unit = self.parent.settings.get_force_unit_string()
                
                # Calculate text position 
                text_offset = 0.12 * geom_size 
                
                # Normalize direction for text placement
                dx, dy, dz = actual_direction
                magnitude = np.linalg.norm(actual_direction)
                if magnitude > 0:
                    dx, dy, dz = dx/magnitude, dy/magnitude, dz/magnitude
                
                # Calculate text position perpendicular to force direction 
                text_pos = [
                    first_triangle['center'][0] + text_offset * (-dy),  # Perpendicular to direction
                    first_triangle['center'][1] + text_offset * (dx),   # Perpendicular to direction
                    first_triangle['center'][2] + text_offset * 0.2    # Small Z offset
                ]
                
                # Create force text showing the actual force value with sign
                force_text = f"{axis_name}: {force_value:+.1f} {force_unit}"
                
                # Add text
                text_actor = self.parent.plotter.add_point_labels(
                    [text_pos], 
                    [force_text],
                    point_color=color,
                    font_size=12,
                    text_color=color,
                    name=f'{axis_name.lower()}_force_text_{len(self.parent.force_actors)}'
                )
                
                self.parent.force_actors.append(text_actor)
            
            # Render
            self.parent.plotter.render()

    def visualize_x_force_arrows(self, selected_triangles_data, fx):
        """Create X-direction arrow visualization (positive or negative)"""
        x_direction = np.array([1.0, 0.0, 0.0])
        self.visualize_arrows(selected_triangles_data, fx, x_direction, 'red', 'X')

    def visualize_y_force_arrows(self, selected_triangles_data, fy):
        """Create Y-direction arrow visualization (positive or negative)"""
        y_direction = np.array([0.0, 1.0, 0.0])
        self.visualize_arrows(selected_triangles_data, fy, y_direction, 'red', 'Y')

    def visualize_z_force_arrows(self, selected_triangles_data, fz):
        """Create Z-direction arrow visualization (positive or negative)"""
        z_direction = np.array([0.0, 0.0, 1.0])
        self.visualize_arrows(selected_triangles_data, fz, z_direction, 'red', 'Z')

    def apply_torque(self):
        selected_faces = self.parent.stl_geom.store_selected_triangles()
        if not selected_faces:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for torque application.")
            return
        self.parent.message_text.append(f"Applied torque to {len(selected_faces)} triangles.")

    def visualize_constraints(self):
        """Create black visualization for constrained triangles"""
        if not self.parent.stl_geom or not self.parent.constrained_triangles:
            return

        # Remove old constraint actors
        for actor in self.parent.constraint_actors:
            try:
                self.parent.plotter.remove_actor(actor, reset_camera=False)
            except:
                pass
        self.parent.constraint_actors.clear()

        # Create mesh for constrained triangles using existing STL data
        vertices = []
        faces = []
        vertex_count = 0
        
        for tri_id in self.parent.constrained_triangles:
            # Get triangle vertices from mesh.vectors
            triangle_vertices = self.parent.stl_geom.mesh.vectors[tri_id]
            
            # Add vertices to list
            for vertex in triangle_vertices:
                vertices.append(vertex)
            
            # Create face (triangle with 3 vertices)
            face = [3, vertex_count, vertex_count + 1, vertex_count + 2]
            faces.extend(face)
            vertex_count += 3
        
        if vertices:
            # Convert to numpy arrays
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Create PyVista mesh
            constraint_mesh = pv.PolyData(vertices, faces).compute_normals(cell_normals=True, point_normals=True)
            
            # Calculate offset (slightly larger than highlight offset)
            bounds = self.parent.plotter.bounds
            offset = 0.002 * ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5 if bounds else 0.002
            
            # Create offset mesh
            offset_mesh = constraint_mesh.copy()
            offset_mesh.points = offset_mesh.points + offset_mesh.point_data['Normals'] * offset
            
            # Add black constraint visualization
            constraint_actor = self.parent.plotter.add_mesh(
                offset_mesh,
                color="black",
                opacity=0.8,
                culling=False,
                lighting=True
            )
            self.parent.constraint_actors.append(constraint_actor)
        
        self.parent.plotter.render()

    def apply_fixed_constraint(self):
        """Apply fixed XYZ constraint"""
            
        # Get triangle data
        selected_triangles_data = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles_data:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for constraint.")
            return
        
        # Extract triangle indices from the data
        selected_faces = [tri_data['index'] for tri_data in selected_triangles_data]
        
        # Add selected triangles to constrained set
        self.parent.constrained_triangles.update(selected_faces)
        
        # Clear highlights for the constrained triangles
        for face_id in selected_faces:
            self.parent.stl_geom.tri_highlight[face_id] = False
        
        # Update visualizations
        self.parent.update_highlights()  # Remove red highlights
        self.visualize_constraints()  # Add black constraint visualization
        
        # Store constraint data for later use in analysis
        self.parent.constraint_data.append({
            'type': 'Fixed XYZ',
        })

        # Update LivVar to indicate constraints are applied
        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
    
    def apply_fixed_constraint_x(self):
        """Apply fixed X constraint"""
        selected_triangles_data = self.parent.stl_geom.store_selected_triangles()
        if not selected_triangles_data:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for constraint.")
            return
        
        # Extract triangle indices from the data
        selected_faces = [tri_data['index'] for tri_data in selected_triangles_data]
        
        # Add selected triangles to constrained set
        self.parent.constrained_triangles.update(selected_faces)
        
        # Clear highlights for the constrained triangles
        for face_id in selected_faces:
            self.parent.stl_geom.tri_highlight[face_id] = False
        
        # Update visualizations
        self.parent.update_highlights()  # Remove red highlights
        self.visualize_constraints()  # Add black constraint visualization

        # Store constraint data for later use in analysis
        self.parent.constraint_data.append({
            'type': 'Fixed X',
        })

        self.parent.update_LivVar('structural_loads.fixed_constraints', True)

    def apply_fixed_constraint_y(self):
        """Apply fixed Y constraint"""
        selected_triangles_data = self.parent.stl_geom.store_selected_triangles()
        if not selected_triangles_data:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for constraint.")
            return
        
        # Extract triangle indices from the detailed data
        selected_faces = [tri_data['index'] for tri_data in selected_triangles_data]
        
        # Add selected triangles to constrained set
        self.parent.constrained_triangles.update(selected_faces)
        
        # Clear highlights for the constrained triangles
        for face_id in selected_faces:
            self.parent.stl_geom.tri_highlight[face_id] = False
        
        # Update visualizations
        self.parent.update_highlights()  # Remove red highlights
        self.visualize_constraints()  # Add black constraint visualization

        # Store constraint data for later use in analysis
        self.parent.constraint_data.append({
            'type': 'Fixed Y',
        })

        self.parent.update_LivVar('structural_loads.fixed_constraints', True)

    def apply_fixed_constraint_z(self):
        """Apply fixed Z constraint"""
        selected_triangles_data = self.parent.stl_geom.store_selected_triangles()
        if not selected_triangles_data:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No triangles selected for constraint.")
            return
        
        # Extract triangle indices from the detailed data
        selected_faces = [tri_data['index'] for tri_data in selected_triangles_data]
        
        # Add selected triangles to constrained set
        self.parent.constrained_triangles.update(selected_faces)
        
        # Clear highlights for the constrained triangles
        for face_id in selected_faces:
            self.parent.stl_geom.tri_highlight[face_id] = False
        
        # Update visualizations
        self.parent.update_highlights()  # Remove red highlights
        self.visualize_constraints()  # Add black constraint visualization

        # Store constraint data for later use in analysis
        self.parent.constraint_data.append({
            'type': 'Fixed Z',
        })

        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
#---------------------------------------------------------------------------

#----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())