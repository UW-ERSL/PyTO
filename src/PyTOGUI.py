import sys
import os
import vtk
import pyvista as pv
import numpy as np
import math
import time
from scipy.sparse import coo_matrix
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from pyvistaqt import QtInteractor
from stl_reader import STLGeom
import bound_cond
import mat_lib
import linear_solvers
from hex_mesher import HexMesher
from hex_thermal_fea import HexThermalFEA
import hex_structural_fea
from topopt_mma import topopt_mma
from topopt_common import TOParams, TO_QOI
from topopt_ocm import topopt_optimality_criteria
from topopt_pareto import topopt_pareto
from topopt_levelset import topopt_levelset 
"""
1) writing a new functions of plotting (making changes in the exising functions of plotting)
2) Adaptive sizing of Arrows for topopt constraints
3) Need to Implement Help window
4) Cyclic (Z) constraints are wrong, need to fix the logic for cyclic symmetry
5) To introduce Torque loads, need to implement a new load type
6) Need to implement Plane method for selecting triangles
7) 
8) 
9)  
10)  
11)
"""
DEFAULT_FONT_SIZE = 32
#---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    WINDOW_SIZE = (1280, 768)
    SIDEBAR_WIDTH = 250
    MESSAGE_HEIGHT = 162
    
    #Workflow configuration
    WORKFLOW_STEPS = {
        'geometry_loaded': {
            'enables': ['Material', 'Structural Loads', 'Thermal Loads'],
            'message': "Geometry loaded. You can now define material properties."
        },
        'material_defined': {
            'enables': [],
            'message': "Material defined. You can now apply structural or thermal loads.",
            'icon': 'check'
        },
        'structural_loads_complete': {
            'enables': ['Analysis', 'TopOpt Constraints'],
            'message': "Structural loads complete (forces and constraints applied). You can now run analysis or define TopOpt constraints.",
            'icon': 'check'
        },
        'thermal_loads_complete': {
            'enables': ['Analysis', 'TopOpt Constraints'],
            'message': "Thermal loads applied. You can now run analysis or define TopOpt constraints.",
            'icon': 'check'
        },
        'analysis_performed': {
            'message': "Analysis completed.",
            'icon': 'check'
        },
        'topopt_constraints_defined': {
            'enables': ['Structural TopOpt', 'Thermal TopOpt'],
            'message': "TopOpt constraints defined. You can now run topology optimization.",
            'icon': 'check'
        }
    }
    
    #Button configuration with workflow requirements
    BUTTON_CONFIG = [
        {"name": "Units", "icon": "arrow", "always_enabled": True, "handler": "open_units_window"},
        {"name": "Geometry", "icon": "arrow", "always_enabled": True, "handler": "open_geometry_window"},
        {"name": "Material", "icon": "cross", "requires": "geometry_loaded", "handler": "open_material_window"},
        {"name": "Structural Loads", "icon": "cross", "requires": "material_defined", "handler": "open_structural_loads_window"},
        {"name": "Thermal Loads", "icon": "cross", "requires": "material_defined", "handler": "open_thermal_loads_window"},
        {"name": "Body force", "icon": "cross", "requires": "material_defined", "handler": "show_coming_soon"},
        {"name": "Display Options", "icon": "arrow", "always_enabled": True, "handler": "open_display_options_window"},
        {"name": "Analysis", "icon": "cross", "requires": "loads_applied", "handler": "open_analysis_window"},
        {"name": "TopOpt Constraints", "icon": "cross", "requires": "loads_applied", "handler": "open_topopt_constraints_window"},
        {"name": "Structural TopOpt", "icon": "cross", "requires": "topopt_constraints_defined", "handler": "open_structural_topopt_window"},
        {"name": "Thermal TopOpt", "icon": "cross", "requires": "topopt_constraints_defined", "handler": "show_coming_soon"},
        {"name": "TopOpt Results", "icon": "cross", "requires": "topopt_performed", "handler": "show_coming_soon"},
        {"name": "Projects", "icon": "arrow", "always_enabled": True, "handler": "open_projects_window"},
        {"name": "Help", "icon": "arrow", "always_enabled": True, "handler": "show_help"}
    ]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.init_window()
        self.init_state()
        self.init_ui()
        self.setup_visualization()
        self.connect_events()

    def init_window(self):
        """Initialize basic window properties"""
        self.setWindowTitle("PyTO")
        self.resize(*self.WINDOW_SIZE)

    def init_state(self):
        """Initialize application state and settings"""
        self.settings = Settings()
        self.LivVar = self.create_initial_state()
        
        # Initialize data containers
        self.constrained_triangles = set()
        self.constraint_actors = []
        self.constraint_data = []
        self.force_data = []
        self.force_actors = []
        self.material_data = None
        self.applied_material = None
        self.highlight_actor = None
        self.highlight_mode = 'coarse'
        self.stl_geom = None
        self.hex_mesh = None

        self.topopt_constraints = None

    def create_initial_state(self):
        """Create the initial LivVar state dictionary"""
        return {
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
            "current_step": "init"
        }

    def init_ui(self):
        """Initialize the ui"""
        # Main layout setup
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        #Create main horizontal layout
        self.create_main_layout()
        self.create_sidebar()
        self.create_message_frame()
        self.create_status_bar()

    def create_main_layout(self):
        """Create the main horizontal layout with PyVista frame"""
        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.setSpacing(10)
        self.h_layout.setContentsMargins(10, 10, 10, 10)
        
        #PyVista Frame
        self.pv_frame = QtWidgets.QFrame()
        self.pv_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.pv_layout = QtWidgets.QVBoxLayout(self.pv_frame)
        self.pv_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.pv_frame)
        self.pv_layout.addWidget(self.plotter.interactor)
        self.h_layout.addWidget(self.pv_frame, stretch=4)

    def create_sidebar(self):
        """Create sidebar with buttons based on configuration"""
        self.sidebar = QtWidgets.QFrame()
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.sidebar.setFixedWidth(self.SIDEBAR_WIDTH)
        self.sidebar_buttons = {}
        
        button_style = """
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
        """
        
        for config in self.BUTTON_CONFIG:
            btn = self.create_button(config, button_style)
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons[config["name"]] = btn
            
        sidebar_layout.addStretch()
        self.h_layout.addWidget(self.sidebar, stretch=0)

    def create_button(self, config, style):
        """Create a single button based on configuration"""
        btn = QtWidgets.QPushButton(config["name"])
        btn.setStyleSheet(style)
        btn.setIcon(self.get_icon(config["icon"]))
        btn.setIconSize(QSize(16, 16))
        btn.clicked.connect(lambda: self.handle_button_click(config))
        return btn

    def create_message_frame(self):
        """Create message display frame"""
        self.message_frame = QtWidgets.QFrame()
        self.message_frame.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        message_layout = QtWidgets.QVBoxLayout(self.message_frame)
        
        self.message_text = QtWidgets.QTextEdit()
        self.message_text.setFixedHeight(self.MESSAGE_HEIGHT)
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
        
        # Add to main layout
        self.main_layout.addLayout(self.h_layout, stretch=4)
        self.main_layout.addWidget(self.message_frame, stretch=1)

    def create_status_bar(self):
        """Create status bar with version information"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
        """)
        
        labels_config = [
            ("PyTO GUI Version 2025.01. ", ""),
            ("GUI Build Date 6.26.2025. ", ""),
            ("This is an academic license, and should not be used for commercial purposes.", "color: red;")
        ]
        
        for text, style in labels_config:
            label = QtWidgets.QLabel(text)
            if style:
                label.setStyleSheet(style)
            status_bar.addWidget(label)

    def setup_visualization(self):
        """Setup PyVista visualization environment"""
        self.picker = pv._vtk.vtkCellPicker()
        self.plotter.add_axes(interactive=False)
        self.plotter.set_background('white')
        self.plotter.enable_parallel_projection()

#################################################################
    def connect_events(self):
        """Connect event handlers"""
        # Event connections
        pass
#################################################################

    def handle_button_click(self, config):
        """button click handler"""
        if not self.check_button_availability(config):
            return
            
        handler_method = getattr(self, config["handler"], None)
        if handler_method and callable(handler_method):
            handler_method()
        else:
            self.show_coming_soon()

    def check_button_availability(self, config):
        """Check if button is available based on workflow state"""
        if config.get("always_enabled", False):
            return True
            
        requirement = config.get("requires")
        if not requirement:
            return True
            
        if requirement == "loads_applied":
            loads_ready = (self.LivVar['structural_loads']['applied'] or 
                        self.LivVar['thermal_loads']['applied'])
            if not loads_ready:
                self.show_workflow_warning("Please apply structural or thermal loads first.")
                return False
        elif requirement == "topopt_constraints_defined":
            if not self.LivVar.get('topopt', {}).get('constraints_defined', False):
                self.show_workflow_warning("Please define topology optimization constraints first.")
                return False
        elif requirement in ["material_defined", "geometry_loaded"]:
            if not self.LivVar.get(requirement, False):
                self.show_workflow_warning(f"Please complete {requirement.replace('_', ' ')} first.")
                return False
                
        return True

    def show_workflow_warning(self, message):
        """Show workflow warning message"""
        QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def update_LivVar(self, key, value=True):
        """Update state and trigger UI updates"""
        self.set_nested_value(self.LivVar, key.split('.'), value)
        self.handle_state_change(key, value)

    def set_nested_value(self, dictionary, keys, value):
        """Set value in nested dictionary using key path"""
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    def handle_state_change(self, key, value):
        """Handle state changes and update UI accordingly"""
        state_handlers = {
            'geometry_loaded': self.handle_geometry_loaded,
            'material_defined': self.handle_material_defined,
            'structural_loads.forces_applied': self.handle_structural_loads_change,
            'structural_loads.fixed_constraints': self.handle_structural_loads_change,
            'thermal_loads.applied': self.handle_thermal_loads_applied,
            'topopt.constraints_defined': self.handle_topopt_constraints_defined,
        }
        
        handler = state_handlers.get(key)
        if handler:
            handler(value)

    def handle_geometry_loaded(self, value):
        if value:
            self.update_workflow_step('geometry_loaded')

    def handle_material_defined(self, value):
        if value:
            self.update_workflow_step('material_defined')

    def handle_structural_loads_change(self, value):
        if (self.LivVar['structural_loads']['forces_applied'] and 
            self.LivVar['structural_loads']['fixed_constraints']):
            self.LivVar['structural_loads']['applied'] = True
            self.update_workflow_step('structural_loads_complete')

    def handle_thermal_loads_applied(self, value):
        if value:
            self.update_workflow_step('thermal_loads_complete')

    def handle_topopt_constraints_defined(self, value):
        if value:
            self.update_workflow_step('topopt_constraints_defined')

    def update_workflow_step(self, step_key):
        """Update workflow step and associated UI elements"""
        step_config = self.WORKFLOW_STEPS.get(step_key, {})
        
        #Update enabled buttons
        for button_name in step_config.get('enables', []):
            self.set_sidebar_icon(button_name, "arrow")
            
        #Update current step icon
        icon = step_config.get('icon', 'check')
        button_name = self.get_button_name_for_step(step_key)
        if button_name:
            self.set_sidebar_icon(button_name, icon)
            
        #Show message
        message = step_config.get('message', '')
        if message:
            self.message_text.append(message)

    def get_button_name_for_step(self, step_key):
        """Map workflow step to button name"""
        step_to_button = {
            'material_defined': 'Material',
            'structural_loads_complete': 'Structural Loads',
            'thermal_loads_complete': 'Thermal Loads',
            'analysis_performed': 'Analysis',
            'topopt_constraints_defined': 'TopOpt Constraints'
        }
        return step_to_button.get(step_key)

    def get_icon(self, icon_type):
        """Get icon based on type"""
        base_path = os.path.dirname(__file__)
        icon_files = {
            "arrow": "arrow-right.png",
            "arrow-blue": "arrow-blue.png", 
            "cross": "cross.png",
            "check": "check.png"
        }
        
        icon_file = os.path.join(base_path, icon_files.get(icon_type, ""))
        return QIcon(icon_file) if os.path.exists(icon_file) else QIcon()

    def set_sidebar_icon(self, button_text, icon_type):
        """Update sidebar button icon"""
        btn = self.sidebar_buttons.get(button_text)
        if btn:
            btn.setIcon(self.get_icon(icon_type))

    def update_highlights(self):
        """Update triangle highlights"""
        if not self.stl_geom:
            return

        camera_position = self.plotter.camera_position
        self.remove_highlight_actor()
        
        highlight_ids = self.get_highlight_ids()
        if not highlight_ids:
            self.restore_camera_and_render(camera_position)
            return

        highlight_mesh = self.create_highlight_mesh(highlight_ids)
        self.add_highlight_mesh(highlight_mesh)
        self.restore_camera_and_render(camera_position)

    def remove_highlight_actor(self):
        """Remove existing highlight actor"""
        if self.highlight_actor:
            self.plotter.remove_actor(self.highlight_actor, reset_camera=False)
            self.highlight_actor = None

    def get_highlight_ids(self):
        """Get IDs of triangles to highlight"""
        return [i for i, h in enumerate(self.stl_geom.tri_highlight) 
                if h and i not in self.constrained_triangles]

    def create_highlight_mesh(self, highlight_ids):
        """Create mesh for highlighted triangles"""
        vertices, faces = [], []
        vertex_count = 0
        
        for tri_id in highlight_ids:
            triangle_vertices = self.stl_geom.mesh.vectors[tri_id]
            vertices.extend(triangle_vertices)
            faces.extend([3, vertex_count, vertex_count + 1, vertex_count + 2])
            vertex_count += 3
        
        highlight_mesh = pv.PolyData(np.array(vertices), np.array(faces))
        return highlight_mesh.compute_normals(cell_normals=True, point_normals=True)

    def add_highlight_mesh(self, highlight_mesh):
        """Add highlight mesh to plotter"""
        bounds = self.plotter.bounds
        offset = (0.001 * ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + 
                          (bounds[5]-bounds[4])**2)**0.5 if bounds else 0.001)
        
        # Create double-sided mesh
        front_mesh = highlight_mesh.copy()
        front_mesh.points = front_mesh.points + front_mesh.point_data['Normals'] * offset
        back_mesh = highlight_mesh.copy() 
        back_mesh.points = back_mesh.points - back_mesh.point_data['Normals'] * offset
        
        self.highlight_actor = self.plotter.add_mesh(
            front_mesh + back_mesh,
            color="red",
            opacity=0.6,
            culling=False,
            lighting=False,
            reset_camera=False
        )

    def restore_camera_and_render(self, camera_position):
        """Restore camera position and render"""
        self.plotter.camera_position = camera_position
        self.plotter.render()

    def on_left_button_press(self, picked_point, picker=None):
        """Handle left click for triangle selection"""
        if not picker or not self.stl_geom:
            return
        
        cell_id = picker.GetCellId()
        if cell_id < 0 or cell_id in self.constrained_triangles:
            self.message_text.append("Cannot select constrained triangle." if cell_id >= 0 
                                   else "No triangle selected.")
            return
        
        count, area = self.select_triangles(cell_id)
        if count > 0:
            self.update_highlights()
            self.message_text.append(f"Selected {count} triangle{'s' if count > 1 else ''} "
                                   f"with area {area:.6f} square units")

    def select_triangles(self, cell_id):
        """Select triangles based on current mode"""
        mode = getattr(self, 'highlight_mode', 'coarse')
        depth, angle = (0, 0) if mode == 'triangle' else (500, 15)
        return self.stl_geom.highlight_triangles_recursive(cell_id, depth, angle)

    def on_right_button_press(self, obj, event):
        """Handle right click for deselecting all triangles"""
        if not hasattr(self, 'stl_geom') or not self.stl_geom:
            return
            
        selected_count = sum(1 for h in self.stl_geom.tri_highlight if h)
        if selected_count == 0:
            self.message_text.append("No triangles selected to deselect.")
            return
        
        # Deselect all triangles
        for i in range(len(self.stl_geom.tri_highlight)):
            self.stl_geom.tri_highlight[i] = False
        
        self.update_highlights()
        self.message_text.append(f"Deselected all {selected_count} selected triangles.")

    def notify_display_options_update(self):
        """Notify display options window to update checkbox states"""
        # Check if display options window exists and is visible
        for child in self.findChildren(QtWidgets.QDialog):
            if isinstance(child, DisplayOptionsWindow) and child.isVisible():
                child.update_checkbox_states()

    #Window opening methods
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

    def open_thermal_loads_window(self):
        self.thermal_loads_window = ThermalLoadsWindow(self)
        self.thermal_loads_window.show()

    def open_analysis_window(self):
        dialog = AnalysisWindow(self)
        dialog.show()

    def open_topopt_constraints_window(self):
        # Only remove mesh/analysis/result actors, keep geometry, loads, constraints, and info
        keep_names = {
            "geometry_info",
            "stl_geometry",
        }
        # Add all structural/thermal load actors and constraint actors to keep_names
        if hasattr(self, "force_actors"):
            for actor in self.force_actors:
                if hasattr(actor, "GetName"):
                    keep_names.add(actor.GetName())
        if hasattr(self, "constraint_actors"):
            for actor in self.constraint_actors:
                if hasattr(actor, "GetName"):
                    keep_names.add(actor.GetName())
        if hasattr(self, "thermal_loads_window"):
            for attr in ["fixed_temp_actors", "heat_source_actors", "total_heat_actors"]:
                if hasattr(self.thermal_loads_window, attr):
                    for actor in getattr(self.thermal_loads_window, attr):
                        if hasattr(actor, "GetName"):
                            keep_names.add(actor.GetName())
        if hasattr(self, "topopt_constraint_actors"):
            for actor in self.topopt_constraint_actors.values():
                if isinstance(actor, list):
                    for a in actor:
                        if hasattr(a, "GetName"):
                            keep_names.add(a.GetName())
                else:
                    if hasattr(actor, "GetName"):
                        keep_names.add(actor.GetName())

        # Remove all actors not in keep_names
        for name in list(self.plotter.actors.keys()):
            if name not in keep_names:
                self.plotter.remove_actor(name, reset_camera=False)

        # Re-plot the STL geometry if not present
        if self.stl_geom and "stl_geometry" not in self.plotter.actors:
            self.stl_geom.plotGeometry(
                show_edges=False,
                show_axes=False,
                show_bounding_box=False,
                plotter=self.plotter
            )
        self.update_highlights()
        self.topopt_constraints_window = TopOptConstraintsWindow(self)
        self.topopt_constraints_window.show()

    def open_structural_topopt_window(self):
        self.structural_topopt_window = StructuralTopOptWindow(self)
        self.structural_topopt_window.show()

    def open_projects_window(self):
        dialog = ProjectsWindow(self)
        dialog.exec_()

    def open_display_options_window(self):
        dialog = DisplayOptionsWindow(self)
        dialog.show()

    def show_coming_soon(self):
        QtWidgets.QMessageBox.information(self, "Coming Soon", 
                                         "This feature will be implemented soon.")

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "Help", 
                                         "PyTO Help documentation will be available soon.")
#---------------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.unit_system = "MKS"
        self.temperature_unit = "Kelvin"
        self.angle_unit = "Degree"

        #Add conversion factors for different unit systems
        self.unit_conversions = {
            #Length conversions to/from meters
            "length": {
                "MKS": 1.0,       # meters (base)
                "mmKS": 1000.0,   # millimeters
                "IPS": 39.37      # inches
            },
            #Force conversions to/from Newtons
            "force": {
                "MKS": 1.0,      # Newtons (base)
                "mmKS": 1.0,     # Newtons
                "IPS": 0.2248    # pounds-force
            },
            #Stress conversions to/from Pascal
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
            
        #Convert to meters first (base unit)
        value_in_meters = value / self.unit_conversions["length"][from_system]
        
        #Then convert to target unit
        return value_in_meters * self.unit_conversions["length"][to_system]
    
    def convert_force(self, value, from_system="MKS", to_system=None):
        """Convert a force value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        #Convert to newtons first (base unit)
        value_in_newtons = value / self.unit_conversions["force"][from_system]
        
        #Then convert to target unit
        return value_in_newtons * self.unit_conversions["force"][to_system]
    
    def convert_stress(self, value, from_system="MKS", to_system=None):
        """Convert a stress value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        #Convert to Pascal first (base unit)
        value_in_pascal = value / self.unit_conversions["stress"][from_system]
        
        #Then convert to target unit
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
        """Return the force unit string"""
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

        #Unit system
        layout.addWidget(QtWidgets.QLabel("Unit System:"))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["MKS", "mmKS", "IPS"])
        self.unit_combo.setCurrentText(settings.unit_system)
        layout.addWidget(self.unit_combo)

        #Temperature unit
        layout.addWidget(QtWidgets.QLabel("Temperature Unit:"))
        self.temp_combo = QtWidgets.QComboBox()
        self.temp_combo.addItems(["Kelvin", "Celsius", "Fahrenheit"])
        self.temp_combo.setCurrentText(settings.temperature_unit)
        layout.addWidget(self.temp_combo)

        #Angle unit
        layout.addWidget(QtWidgets.QLabel("Angle Unit:"))
        self.angle_combo = QtWidgets.QComboBox()
        self.angle_combo.addItems(["Degree", "Radian"])
        self.angle_combo.setCurrentText(settings.angle_unit)
        layout.addWidget(self.angle_combo)

        #Apply button
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
        
        #just a message about units being applied
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

            # Get material name or 'None'
            material_name = "None"
            if getattr(self.parent, "applied_material", None):
                material_name = self.parent.applied_material.get("name", "None")

            # Add geometry info text
            info_lines = [
                f"Model: {os.path.basename(file_path)}",
                f"Volume: {volume:.2e} {length_unit}³",
                f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                f"Material: {material_name}"
            ]

            # Remove old geometry info text
            self.parent.plotter.remove_actor("geometry_info")

            self.parent.plotter.add_text(
                "\n".join(info_lines),
                position="upper_left",
                font_size=10,
                color="black",
                name="geometry_info",
            )

            # Set geometry reference
            self.parent.stl_geom = self.stl_geom

            # Enable picking
            self.parent.plotter.disable_picking()
            
            picker = pv._vtk.vtkCellPicker()
            picker.SetTolerance(1e-8)
            self.parent.plotter.enable_point_picking(
                callback=self.parent.on_left_button_press,
                use_picker=True,
                picker=picker,  
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

        #Material name dropdown
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(self.materials.keys())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        form_layout.addRow("Material Name:", self.material_combo)

        #Create fields
        self.fields = {}

        self.fields["Young's Modulus"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Young's Modulus"), self.fields["Young's Modulus"])

        self.fields["Poisson's Ratio"] = QtWidgets.QLineEdit()
        form_layout.addRow("Poisson's Ratio (-):", self.fields["Poisson's Ratio"])

        self.fields["Yield Strength"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Yield Strength"), self.fields["Yield Strength"])

        self.fields["Density"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Density"), self.fields["Density"])

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

        #Checkbox and buttons
        self.optimize_check = QtWidgets.QCheckBox("Do not optimize")
        layout.addWidget(self.optimize_check)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_material)
        layout.addWidget(self.apply_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.on_material_changed(self.material_combo.currentText())

    def label_with_unit(self, key):
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
            value = mat[key]
            # Show large/small numbers in scientific notation, others as normal
            if isinstance(value, float) and (abs(value) >= 1e5 or (abs(value) < 1e-2 and value != 0)):
                field.setText(f"{value:.2e}")
            else:
                field.setText(str(value))
            field.setReadOnly(name != "Custom")
        if self.material_applied:
            self.material_combo.setEnabled(False)
            for field in self.fields.values():
                field.setReadOnly(True)
            self.apply_btn.setEnabled(False)

    def apply_material(self):
        name = self.material_combo.currentText()
        if name == "Custom":
            #Save custom values
            for key in self.fields:
                try:
                    val = float(self.fields[key].text())
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid value for {key}")
                    return
                self.materials["Custom"][key] = val
        self.material_applied = True
        self.on_material_changed(name)

        #Store the applied material in parent for later use
        self.parent.applied_material = {
            'name': name,
            'properties': self.materials[name].copy()
        }

        #Update geometry info text if geometry is loaded
        if getattr(self.parent, "stl_geom", None):
            self.parent.plotter.remove_actor("geometry_info")
            # Reuse the info text logic from GeometryWindow
            area, volume, _, _ = self.parent.stl_geom.compute_mass_properties()
            bounds = self.parent.stl_geom.get_bounding_box()
            length_unit = self.parent.settings.get_length_unit_string()
            material_name = self.parent.applied_material.get("name", "None")
            info_lines = [
                f"Model: {os.path.basename(self.parent.stl_geom.file_path)}",
                f"Volume: {volume:.2e} {length_unit}³",
                f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                f"Material: {material_name}"
            ]
            self.parent.plotter.add_text(
                "\n".join(info_lines),
                position="upper_left",
                font_size=12,
                color="black",
                name="geometry_info",
                font="arial",
            )

        #Update LivVar to indicate material is defined
        self.parent.update_LivVar('material_defined', True)
        
        self.parent.message_text.append(f"Material '{name}' applied successfully.")

        self.parent.set_sidebar_icon("Material", "check")
        self.parent.set_sidebar_icon("Structural Loads", "arrow")

        self.close()
        
    def update_units(self):
        #Dyanmic Label Update
        form_layout = self.layout().itemAt(0).layout()
        form_layout.labelForField(self.fields["Young's Modulus"]).setText(self.label_with_unit("Young's Modulus"))
        form_layout.labelForField(self.fields["Yield Strength"]).setText(self.label_with_unit("Yield Strength"))
        form_layout.labelForField(self.fields["Density"]).setText(self.label_with_unit("Density"))

    def showEvent(self, event):
        super().showEvent(event)
        self.update_units()
#---------------------------------------------------------------------------
class StructuralLoadsWindow(QtWidgets.QDialog):
    # Class-level configuration
    FORCE_RANGE = (-1e6, 1e6)
    FORCE_DECIMALS = 1
    ARROW_THRESHOLD = 25
    MAX_MARKERS = 5
    ARROW_SCALE_FACTOR = 0.1
    CONSTRAINT_OFFSET_FACTOR = 0.002
    HIGHLIGHT_OFFSET_FACTOR = 0.001
    
    #Load type configuration
    LOAD_TYPES = {
        "Force": {
            "handler": "apply_force",
            "show_force": True,
            "show_torque": False,
            "directions": {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]},
            "color": "red"
        },
        "Torque": {
            "handler": "apply_torque", 
            "show_force": False,
            "show_torque": True,
            "color": "blue"
        },
        "Fixed XYZ": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed XYZ",
            "axes": ["X", "Y", "Z"],
            "show_force": False,
            "show_torque": False
        },
        "Fixed X": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed X", 
            "axes": ["X"],
            "show_force": False,
            "show_torque": False
        },
        "Fixed Y": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed Y",
            "axes": ["Y"], 
            "show_force": False,
            "show_torque": False
        },
        "Fixed Z": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed Z",
            "axes": ["Z"],
            "show_force": False,
            "show_torque": False
        }
    }
    
    SELECTION_MODES = {
        "Coarse Cylinder": "coarse",
        "Triangle": "triangle"
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_window()
        self.init_ui()
        self.connect_events()

    def init_window(self):
        """Initialize window properties"""
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Structural Loads")

    def init_ui(self):
        """Initialize user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.create_load_set_control(layout)
        self.create_selection_control(layout) 
        self.create_load_type_control(layout)
        self.create_force_controls(layout)
        self.create_torque_controls(layout)
        self.create_buttons(layout)

    def create_load_set_control(self, parent_layout):
        """Create load set spinbox control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Load Set"))
        
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        layout.addWidget(self.load_set_spin)
        
        parent_layout.addLayout(layout)

    def create_selection_control(self, parent_layout):
        """Create selection mode control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Selection"))
        
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(list(self.SELECTION_MODES.keys()))
        layout.addWidget(self.selection_combo)
        
        parent_layout.addLayout(layout)

    def create_load_type_control(self, parent_layout):
        """Create load type control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Load Type"))
        
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems(list(self.LOAD_TYPES.keys()))
        layout.addWidget(self.load_type)
        
        parent_layout.addLayout(layout)

    def create_force_controls(self, parent_layout):
        """Create force component controls"""
        self.force_group = QtWidgets.QGroupBox("Force Components")
        layout = QtWidgets.QVBoxLayout(self.force_group)
        
        # Create force spinboxes dynamically
        self.force_spinboxes = {}
        for axis in ["X", "Y", "Z"]:
            spinbox = self.create_force_spinbox(f"{axis}: ")
            self.force_spinboxes[axis] = spinbox
            layout.addWidget(spinbox)
        
        parent_layout.addWidget(self.force_group)

    def create_force_spinbox(self, prefix):
        """Create a single force spinbox with standard settings"""
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(*self.FORCE_RANGE)
        spinbox.setDecimals(self.FORCE_DECIMALS)
        spinbox.setPrefix(prefix)
        return spinbox

    def create_torque_controls(self, parent_layout):
        """Create torque controls"""
        self.torque_group = QtWidgets.QGroupBox("Torque")
        layout = QtWidgets.QVBoxLayout(self.torque_group)
        
        self.torque_magnitude_spin = self.create_force_spinbox("Magnitude: ")
        layout.addWidget(self.torque_magnitude_spin)
        
        self.torque_group.hide()
        parent_layout.addWidget(self.torque_group)

    def create_buttons(self, parent_layout):
        """Create control buttons"""
        layout = QtWidgets.QHBoxLayout()
        
        self.apply_button = QtWidgets.QPushButton("Apply")
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        layout.addWidget(self.apply_button)
        layout.addWidget(close_button)
        parent_layout.addLayout(layout)

    def connect_events(self):
        """Connect all event handlers"""
        self.selection_combo.currentIndexChanged.connect(self.on_selection_mode_changed)
        self.load_type.currentTextChanged.connect(self._on_load_type_changed)
        self.apply_button.clicked.connect(self.on_apply_clicked)
        
        #Set initial state
        self.on_selection_mode_changed(0)
        self._on_load_type_changed(self.load_type.currentText())

    def on_selection_mode_changed(self, index):
        """Handle selection mode changes"""
        mode_text = self.selection_combo.currentText()
        self.parent.highlight_mode = self.SELECTION_MODES[mode_text]
        if self.parent.stl_geom:
            # Turn ON triangle edges in "Triangle" mode, OFF otherwise
            for name, actor in self.parent.plotter.actors.items():
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if self.parent.highlight_mode == "triangle":
                        if hasattr(prop, 'EdgeVisibilityOn'):
                            prop.EdgeVisibilityOn()
                            prop.SetEdgeColor(0, 0, 0)
                            prop.SetLineWidth(1)
                    else:
                        if hasattr(prop, 'EdgeVisibilityOff'):
                            prop.EdgeVisibilityOff()
            if self.parent.highlight_mode == "triangle":
                self.parent.update_highlights()
            else:
                if getattr(self.parent, "highlight_actor", None):
                    self.parent.plotter.remove_actor(self.parent.highlight_actor, reset_camera=False)
                    self.parent.highlight_actor = None
                self.parent.plotter.render()

    def _on_load_type_changed(self, load_type):
        """Handle load type changes and update UI visibility"""
        config = self.LOAD_TYPES[load_type]
        
        self.force_group.setVisible(config.get("show_force", False))
        self.torque_group.setVisible(config.get("show_torque", False))
        
        self.adjustSize()

    def on_apply_clicked(self):
        """apply button"""
        load_type = self.load_type.currentText()
        config = self.LOAD_TYPES[load_type]
        handler_name = config["handler"]
        
        handler_method = getattr(self, handler_name, None)
        if handler_method and callable(handler_method):
            if handler_name == "apply_constraint":
                handler_method(config)
            else:
                handler_method()

    def get_selected_triangles(self):
        """Get selected triangles with validation"""
        selected_triangles = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "No Selection", 
                                         "No triangles selected for load application.")
            return None
        
        return selected_triangles

    def convert_forces_to_base_units(self, forces):
        """Convert forces from current units to base units"""
        current_unit_system = self.parent.settings.unit_system
        if current_unit_system == "MKS":
            return forces
        
        converted = {}
        for axis, force_value in forces.items():
            converted[axis] = self.parent.settings.convert_force(
                force_value, current_unit_system, "MKS"
            )
        return converted

    def get_force_values(self):
        """Get force values from spinboxes"""
        forces = {}
        for axis, spinbox in self.force_spinboxes.items():
            forces[axis] = spinbox.value()
        return forces

    def validate_forces(self, forces):
        """Validate that at least one force component is non-zero"""
        if all(force == 0 for force in forces.values()):
            QtWidgets.QMessageBox.warning(self, "Zero Force", 
                                         "Please enter at least one non-zero force component.")
            return False
        return True

    def clear_force_inputs(self):
        """Clear all force input spinboxes"""
        for spinbox in self.force_spinboxes.values():
            spinbox.setValue(0)

    def apply_force(self):
        """Apply resultant force to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return

        forces = self.get_force_values()
        if not self.validate_forces(forces):
            return

        # Convert to base units
        converted_forces = self.convert_forces_to_base_units(forces)
        force_vec = np.array([converted_forces['X'], converted_forces['Y'], converted_forces['Z']])
        nonzero = [k for k, v in converted_forces.items() if abs(v) > 1e-12]
        norm = np.linalg.norm(force_vec)
        if norm == 0:
            return

        unit_direction = force_vec / norm

        # Decide label value: show signed value if only one axis, else resultant (always positive)
        if len(nonzero) == 1:
            label_value = converted_forces[nonzero[0]]
        else:
            label_value = norm

        self.visualize_force_arrows(selected_triangles, norm, unit_direction, "red", "Resultant", label_value)

        self.store_force_data(selected_triangles, converted_forces)
        self.update_force(selected_triangles, [f"Resultant: {label_value:+.1f}"])

    def store_force_data(self, selected_triangles, forces):
        """Store force data for analysis"""
        force_info = {
            'triangles': [tri_data['index'] for tri_data in selected_triangles],
            'triangle_data': selected_triangles,
            'force_x': forces['X'],
            'force_y': forces['Y'], 
            'force_z': forces['Z'],
            'load_set': self.load_set_spin.value(),
            'type': 'force_xyz'
        }
        self.parent.force_data.append(force_info)

    def update_force(self, selected_triangles, applied_forces):
        """update force application - clear highlights, update state, show message"""
        #Clear highlights
        for tri_data in selected_triangles:
            self.parent.stl_geom.tri_highlight[tri_data['index']] = False
        
        self.parent.update_highlights()
        
        #Show success message
        force_unit = self.parent.settings.get_force_unit_string()
        force_components = ", ".join(applied_forces)
        self.parent.message_text.append(
            f"Applied forces ({force_components}) {force_unit} to {len(selected_triangles)} triangles."
        )
        
        #Update state and clear inputs
        self.parent.update_LivVar('structural_loads.forces_applied', True)
        self.clear_force_inputs()

        #Notify display options window
        self.parent.notify_display_options_update()

    def visualize_force_arrows(self, selected_triangles, force_value, direction_vector, color, axis_name, label_value=None):
        """Create directional arrow visualization for forces"""
        display_triangles = self.get_display_triangles(selected_triangles)
        if not display_triangles:
            return

        arrow_scale = self.calculate_arrow_scale()
        actual_direction = np.array(direction_vector) * (1.0 if force_value >= 0 else -1.0)
        centers = np.array([tri_data['center'] for tri_data in display_triangles])
        arrows = self.create_arrow_mesh(centers, actual_direction, arrow_scale)
        force_actor = self.parent.plotter.add_mesh(
            arrows,
            color=color,
            show_edges=False,
            name=f'{axis_name.lower()}_force_arrows_{len(self.parent.force_actors)}'
        )
        self.parent.force_actors.append(force_actor)

        # Pass label_value to the label function
        self.add_force_text_label(display_triangles[0], label_value if label_value is not None else force_value, actual_direction, color, axis_name)
        self.parent.plotter.render()

    def get_display_triangles(self, selected_triangles):
        """Get triangles to display based on threshold system"""
        if len(selected_triangles) > self.ARROW_THRESHOLD:
            step = len(selected_triangles) // self.MAX_MARKERS
            indices = range(0, len(selected_triangles), step)[:self.MAX_MARKERS]
            return [selected_triangles[i] for i in indices]
        return selected_triangles

    def calculate_arrow_scale(self):
        """Calculate appropriate arrow scale based on model size"""
        bounds = self.parent.plotter.bounds
        if bounds:
            model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            return model_size * self.ARROW_SCALE_FACTOR
        return 1.0

    def create_arrow_mesh(self, centers, direction, scale):
        """Create arrow mesh for visualization"""
        directions = np.tile(direction, (len(centers), 1))
        scales = np.full(len(centers), scale)
        
        arrows = pv.PolyData(centers)
        arrows['vectors'] = directions * scales.reshape(-1, 1)
        
        return arrows.glyph(orient='vectors', scale='vectors', factor=1.0)

    def add_force_text_label(self, triangle, force_value, direction, color, axis_name):
        """Add text label for force visualization at the tip of the first arrow"""
        # Calculate arrow scale and offset
        geom_size = self.calculate_arrow_scale() / self.ARROW_SCALE_FACTOR
        text_offset = 0.12 * geom_size

        # Normalize direction for text placement
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            unit_dir = direction / magnitude
            # Arrow tip position: center + direction * arrow_scale
            arrow_tip = np.array(triangle['center']) + unit_dir * self.calculate_arrow_scale()
            text_pos = arrow_tip.tolist()

            # Create and add text
            force_unit = self.parent.settings.get_force_unit_string()
            force_text = f"{force_value:+.1f} {force_unit}"

            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [force_text],
                font_size=DEFAULT_FONT_SIZE, text_color=color,
                shape_opacity=0,
                name=f'{axis_name.lower()}_force_text_{len(self.parent.force_actors)}'
            )
            self.parent.force_actors.append(text_actor)

    def apply_torque(self):
        """Apply torque to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        self.parent.message_text.append(f"Applied torque to {len(selected_triangles)} triangles.")

    def apply_constraint(self, config):
        """constraint application method"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        # Extract triangle indices
        selected_faces = [tri_data['index'] for tri_data in selected_triangles]
        
        # Apply constraint
        self.apply_constraint_to_triangles(selected_faces, config["constraint_type"])
        
        # Update visualizations and state
        self.update_constraint_application(selected_faces, config["constraint_type"])

    def apply_constraint_to_triangles(self, triangle_ids, constraint_type):
        """Apply constraint to specific triangles"""
        # Add to constrained set
        self.parent.constrained_triangles.update(triangle_ids)
        
        # Clear highlights
        for face_id in triangle_ids:
            self.parent.stl_geom.tri_highlight[face_id] = False
        
        # Store constraint data
        self.parent.constraint_data.append({'type': constraint_type})

    def update_constraint_application(self, triangle_ids, constraint_type):
        """update constraint application"""
        #Update visualizations
        self.parent.update_highlights()
        self.visualize_constraints()
        
        #Update state
        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        
        #Show message
        self.parent.message_text.append(
            f"Applied {constraint_type} constraint to {len(triangle_ids)} triangles."
        )

        #Notify display options window
        self.parent.notify_display_options_update()

    def visualize_constraints(self):
        """Create black visualization for constrained triangles"""
        if not self.parent.stl_geom or not self.parent.constrained_triangles:
            return

        #Remove old constraint actors
        self.remove_constraint_actors()
        
        #Create constraint mesh
        constraint_mesh = self.create_constraint_mesh()
        if constraint_mesh:
            self.add_constraint_visualization(constraint_mesh)
        
        self.parent.plotter.render()

    def remove_constraint_actors(self):
        """Remove existing constraint actors"""
        for actor in self.parent.constraint_actors:
            self.parent.plotter.remove_actor(actor, reset_camera=False)
        self.parent.constraint_actors.clear()

    def create_constraint_mesh(self):
        """Create mesh for constrained triangles"""
        vertices, faces = [], []
        vertex_count = 0
        
        for tri_id in self.parent.constrained_triangles:
            triangle_vertices = self.parent.stl_geom.mesh.vectors[tri_id]
            
            vertices.extend(triangle_vertices)
            faces.extend([3, vertex_count, vertex_count + 1, vertex_count + 2])
            vertex_count += 3
        
        if not vertices:
            return None
        
        #Create PyVista mesh
        vertices = np.array(vertices)
        faces = np.array(faces)
        return pv.PolyData(vertices, faces).compute_normals(cell_normals=True, point_normals=True)

    def add_constraint_visualization(self, constraint_mesh):
        """Add constraint visualization to plotter"""
        #Calculate offset
        bounds = self.parent.plotter.bounds
        offset = (self.CONSTRAINT_OFFSET_FACTOR * 
                 ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5 
                 if bounds else self.CONSTRAINT_OFFSET_FACTOR)
        
        #Create offset mesh
        offset_mesh = constraint_mesh.copy()
        offset_mesh.points = offset_mesh.points + offset_mesh.point_data['Normals'] * offset
        
        #Add visualization
        constraint_actor = self.parent.plotter.add_mesh(
            offset_mesh,
            color="black",
            opacity=0.8,
            culling=False,
            lighting=True
        )
        self.parent.constraint_actors.append(constraint_actor)
#---------------------------------------------------------------------------
class ThermalLoadsWindow(QtWidgets.QDialog):
    ARROW_THRESHOLD = 25
    MAX_MARKERS = 5
    ARROW_SCALE_FACTOR = 0.1
    TEXT_OFFSET_FACTOR = 0.12
    
    # Selection mode configuration
    SELECTION_MODES = {
        "Coarse Cylinder": "coarse",
        "Triangle": "triangle"
    }
    
    # Thermal load type configuration
    THERMAL_TYPES = {
        "Temperature": {
            "handler": "apply_temperature",
            "group": "temp_group",
            "value_field": "temp_value_spin",
            "data_key": "temperature",
            "storage_list": "fixed_temps",
            "actor_list": "fixed_temp_actors",
            "color": "blue",
            "label": "Temp",
            "type_name": "fixed_temperature",
            "default_value": 300,
            "range": (0, 2000),
            "decimals": 1,
            "arrow_direction": "inward"  # Special handling for temperature
        },
        "Heat Flux": {
            "handler": "apply_heat_flux",
            "group": "heat_flux_group", 
            "value_field": "heat_flux_value_spin",
            "data_key": "heat_flux",
            "storage_list": "heat_sources",
            "actor_list": "heat_source_actors",
            "color": "orange",
            "label": "Heat Flux",
            "type_name": "heat_flux",
            "default_value": 1000,
            "range": (-1e6, 1e6),
            "decimals": 1,
            "arrow_direction": "outward"
        },
        "Total Heat": {
            "handler": "apply_total_heat",
            "group": "total_heat_group",
            "value_field": "total_heat_value_spin", 
            "data_key": "total_heat",
            "storage_list": "total_heat_sources",
            "actor_list": "total_heat_actors",
            "color": "red",
            "label": "Total Heat",
            "type_name": "total_heat",
            "default_value": 100,
            "range": (-1e6, 1e6),
            "decimals": 1,
            "arrow_direction": "outward"
        }
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_window()
        self.init_data_structures()
        self.init_ui()
        self.connect_events()

    def init_window(self):
        """Initialize window properties"""
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Thermal Loads")

    def init_data_structures(self):
        """Initialize thermal loads data structures and actor lists"""
        self.thermal_loads = {
            "fixed_temps": [],
            "heat_sources": [], 
            "total_heat_sources": []
        }
        
        # Initialize actor lists dynamically from configuration
        for config in self.THERMAL_TYPES.values():
            setattr(self, config["actor_list"], [])

    def init_ui(self):
        """Initialize user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.create_selection_control(layout)
        self.create_thermal_type_control(layout)
        self.create_thermal_groups(layout)
        self.create_buttons(layout)

    def create_selection_control(self, parent_layout):
        """Create selection mode control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Selection"))
        
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(list(self.SELECTION_MODES.keys()))
        layout.addWidget(self.selection_combo)
        
        parent_layout.addLayout(layout)

    def create_thermal_type_control(self, parent_layout):
        """Create thermal type control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Thermal Type"))
        
        self.thermal_type = QtWidgets.QComboBox()
        self.thermal_type.addItems(list(self.THERMAL_TYPES.keys()))
        layout.addWidget(self.thermal_type)
        
        parent_layout.addLayout(layout)

    def create_thermal_groups(self, parent_layout):
        """Create all thermal input groups"""
        # Temperature group
        self.temp_group = self.create_temperature_group()
        parent_layout.addWidget(self.temp_group)
        
        # Heat Flux group  
        self.heat_flux_group = self.create_heat_flux_group()
        parent_layout.addWidget(self.heat_flux_group)
        
        # Total Heat group
        self.total_heat_group = self.create_total_heat_group()
        parent_layout.addWidget(self.total_heat_group)
        
        # Set initial visibility
        self.update_group_visibility("Temperature")

    def create_temperature_group(self):
        """Create temperature input group"""
        temp_unit = self.parent.settings.get_temperature_unit_symbol()
        config = self.THERMAL_TYPES["Temperature"]
        
        group = QtWidgets.QGroupBox("Temperature")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel(f"Temperature ({temp_unit})"))
        
        self.temp_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.temp_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_heat_flux_group(self):
        """Create heat flux input group"""
        length_unit = self.parent.settings.get_length_unit_string()
        config = self.THERMAL_TYPES["Heat Flux"]
        
        group = QtWidgets.QGroupBox("Heat Flux")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel(f"Heat Flux (W/{length_unit}²)"))
        
        self.heat_flux_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.heat_flux_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_total_heat_group(self):
        """Create total heat input group"""
        config = self.THERMAL_TYPES["Total Heat"]
        
        group = QtWidgets.QGroupBox("Total Heat")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel("Total Heat (W)"))
        
        self.total_heat_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.total_heat_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_spinbox(self, config):
        """Create a spinbox with configuration"""
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(*config["range"])
        spinbox.setValue(config["default_value"])
        spinbox.setDecimals(config["decimals"])
        return spinbox

    def create_buttons(self, parent_layout):
        """Create control buttons"""
        layout = QtWidgets.QHBoxLayout()
        
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.on_apply_clicked)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        layout.addWidget(apply_button)
        layout.addWidget(close_button)
        parent_layout.addLayout(layout)

    def connect_events(self):
        """Connect event handlers"""
        self.selection_combo.currentIndexChanged.connect(self.on_selection_mode_changed)
        self.thermal_type.currentTextChanged.connect(self.on_thermal_type_changed)
        
        # Set initial state
        self.on_selection_mode_changed(0)

    def on_selection_mode_changed(self, index):
        """Handle selection mode changes"""
        mode_text = self.selection_combo.currentText()
        self.parent.highlight_mode = self.SELECTION_MODES[mode_text]

    def on_thermal_type_changed(self, thermal_type):
        """Handle thermal type changes and update UI visibility"""
        self.update_group_visibility(thermal_type)
        self.adjustSize()

    def update_group_visibility(self, thermal_type):
        """Update visibility of thermal input groups"""
        # Hide all groups first
        self.temp_group.setVisible(False)
        self.heat_flux_group.setVisible(False)
        self.total_heat_group.setVisible(False)
        
        # Show the relevant group
        if thermal_type == "Temperature":
            self.temp_group.setVisible(True)
        elif thermal_type == "Heat Flux":
            self.heat_flux_group.setVisible(True)
        elif thermal_type == "Total Heat":
            self.total_heat_group.setVisible(True)

    def on_apply_clicked(self):
        """apply button handler"""
        thermal_type = self.thermal_type.currentText()
        config = self.THERMAL_TYPES[thermal_type]
        handler_name = config["handler"]
        
        handler_method = getattr(self, handler_name, None)
        if handler_method and callable(handler_method):
            handler_method(config)

    def get_selected_triangles(self):
        """Get selected triangles"""
        selected_triangles = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "No Selection", 
                                         "No triangles selected for thermal load application.")
            return None
        
        return selected_triangles

    def get_spinbox_value(self, config):
        """Get value from the appropriate spinbox"""
        field_name = config["value_field"]
        spinbox = getattr(self, field_name)
        return spinbox.value()

    def store_thermal_data(self, selected_triangles, value, config):
        """Store thermal load data"""
        thermal_info = {
            'triangles': [tri_data['index'] for tri_data in selected_triangles],
            'triangle_data': selected_triangles,
            config["data_key"]: value,
            'type': config["type_name"]
        }
        
        storage_list = self.thermal_loads[config["storage_list"]]
        storage_list.append(thermal_info)

    def create_thermal_visualization(self, selected_triangles, value, config):
        """Create thermal arrows and text visualization"""
        display_triangles = self.get_display_triangles(selected_triangles)
        
        if not display_triangles:
            return
        
        arrow_scale = self.calculate_arrow_scale()
        centers, normals = self.extract_triangle_data(display_triangles)
        
        # Create arrows with direction based on config
        arrow_centers, arrow_directions = self.calculate_arrow_vectors(
            centers, normals, arrow_scale, config["arrow_direction"]
        )
        
        # Create and add arrow visualization
        thermal_actor = self.create_arrow_mesh(arrow_centers, arrow_directions, config)
        self.store_actor(thermal_actor, config)
        
        # Add text label
        self.add_thermal_text_label(display_triangles[0], value, config)
        
        self.parent.plotter.render()

    def get_display_triangles(self, selected_triangles):
        """Get triangles to display based on threshold system"""
        if len(selected_triangles) > self.ARROW_THRESHOLD:
            step = len(selected_triangles) // self.MAX_MARKERS
            indices = range(0, len(selected_triangles), step)[:self.MAX_MARKERS]
            return [selected_triangles[i] for i in indices]
        return selected_triangles

    def calculate_arrow_scale(self):
        """Calculate appropriate arrow scale based on model size"""
        bounds = self.parent.plotter.bounds
        if bounds:
            model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            return model_size * self.ARROW_SCALE_FACTOR
        return 1.0

    def extract_triangle_data(self, display_triangles):
        """Extract centers and normals from triangle data"""
        centers = np.array([tri_data['center'] for tri_data in display_triangles])
        normals = np.array([tri_data['normal'] for tri_data in display_triangles])
        return centers, normals

    def calculate_arrow_vectors(self, centers, normals, arrow_scale, direction_type):
        """Calculate arrow positions and directions based on type"""
        if direction_type == "inward":
            # Temperature arrows point inward (reversed) and are offset
            arrow_directions = -normals * arrow_scale
            arrow_centers = centers + normals * arrow_scale
        else:
            # Heat flux and total heat arrows point outward
            arrow_directions = normals * arrow_scale
            arrow_centers = centers
        
        return arrow_centers, arrow_directions

    def create_arrow_mesh(self, arrow_centers, arrow_directions, config):
        """Create arrow mesh for visualization"""
        arrows = pv.PolyData(arrow_centers)
        arrows['vectors'] = arrow_directions
        arrow_glyph = arrows.glyph(orient='vectors', scale='vectors', factor=1.0)
        
        actor_count = len(getattr(self, config["actor_list"]))
        name = f'{config["label"].lower().replace(" ", "_")}_arrows_{actor_count}'
        
        return self.parent.plotter.add_mesh(
            arrow_glyph,
            color=config["color"],
            show_edges=False,
            name=name
        )

    def store_actor(self, actor, config):
        """Store actor in appropriate list"""
        actor_list = getattr(self, config["actor_list"])
        actor_list.append(actor)

    def add_thermal_text_label(self, triangle, value, config):
        """Add text label for thermal visualization"""
        arrow_scale = self.calculate_arrow_scale()
        
        # Get triangle normal for positioning
        normal = np.array(triangle['normal'])
        center = np.array(triangle['center'])
        
        # Calculate text position with more prominent offset
        if config["arrow_direction"] == "inward":
            # Temperature: place text clearly visible above the surface
            text_pos = center + normal * (arrow_scale * 1.5)
        else:
            # Heat flux and total heat: place text at the arrow tip
            text_pos = center + normal * (arrow_scale * 1.5)
        
        # Ensure text position is a list for PyVista
        text_pos = text_pos.tolist()
        
        # Create text with appropriate units
        text = self.format_thermal_text(value, config)
        
        # Add text label with enhanced visibility
        actor_count = len(getattr(self, config["actor_list"]))
        name = f'{config["label"].lower().replace(" ", "_")}_text_{actor_count}'
        
        try:
            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [text],
                font_size=16,  # Even larger font
                text_color=config["color"],
                name=name,
                always_visible=True,
                shape_opacity=0,
                pickable=False  # Prevent interference with triangle picking
            )
            
            # Store text actor
            self.store_actor(text_actor, config)
            print(f"Successfully created text actor for {config['label']}")
            
        except Exception as e:
            print(f"Error creating text label for {config['label']}: {e}")
            # Fallback: try simpler text creation
            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [text],
                font_size=16,
                text_color=config["color"]
            )
            self.store_actor(text_actor, config)

    def format_thermal_text(self, value, config):
        """Format text display based on thermal type"""
        label = config["label"]
        
        if label == "Temp":
            temp_unit = self.parent.settings.get_temperature_unit_symbol()
            return f"{value:.1f} {temp_unit}"
        elif label == "Heat Flux":
            length_unit = self.parent.settings.get_length_unit_string()
            return f"{value:.1f} W/{length_unit}²"
        else:  # Total Heat
            return f"{value:.1f} W"

    def clear_selection(self, selected_triangles):
        """Clear selection after applying thermal load"""
        for tri_data in selected_triangles:
            self.parent.stl_geom.tri_highlight[tri_data['index']] = False
        self.parent.update_highlights()

    def show_success_message(self, thermal_type, value, triangle_count):
        """Show success message for applied thermal load"""
        config = self.THERMAL_TYPES[thermal_type]
        text = self.format_thermal_text(value, config)
        
        if thermal_type == "Temperature":
            message = f"Applied temperature {text} to {triangle_count} triangles."
        elif thermal_type == "Heat Flux":
            message = f"Applied heat flux {text} to {triangle_count} triangles."
        else:  # Total Heat
            message = f"Applied total heat {text} to {triangle_count} triangles."
        
        self.parent.message_text.append(message)

    # thermal load application methods
    def apply_temperature(self, config):
        """Apply fixed temperature to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Temperature", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()

    def apply_heat_flux(self, config):
        """Apply heat flux to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Heat Flux", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()

    def apply_total_heat(self, config):
        """Apply total heat to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Total Heat", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()
#---------------------------------------------------------------------------
class AnalysisWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Analysis")
        self.setFixedSize(280, 400)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Load Set
        load_set_layout = QtWidgets.QHBoxLayout()
        load_set_layout.addWidget(QtWidgets.QLabel("Load Set"))
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        load_set_layout.addWidget(self.load_set_spin)
        layout.addLayout(load_set_layout)
        
        # Mesh Quality
        mesh_layout = QtWidgets.QHBoxLayout()
        mesh_layout.addWidget(QtWidgets.QLabel("Mesh Quality"))
        self.mesh_combo = QtWidgets.QComboBox()
        self.mesh_combo.addItems(["Very Coarse", "Coarse", "Normal", "Fine", "Very Fine"])
        self.mesh_combo.currentTextChanged.connect(self.on_mesh_quality_changed)
        mesh_layout.addWidget(self.mesh_combo)
        layout.addLayout(mesh_layout)
        
        # Number of Elements
        elements_layout = QtWidgets.QHBoxLayout()
        elements_layout.addWidget(QtWidgets.QLabel("#Elements"))
        self.elements_spin = QtWidgets.QSpinBox()
        self.elements_spin.setRange(1000, 1000000)
        self.elements_spin.setValue(10000)
        elements_layout.addWidget(self.elements_spin)
        layout.addLayout(elements_layout)
        
        # Solver Type
        solver_layout = QtWidgets.QHBoxLayout()
        solver_layout.addWidget(QtWidgets.QLabel("Solver Type"))
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems(["PARDISO", "DPCG", "PCG", "PYAMG", "SPSOLVE"])
        solver_layout.addWidget(self.solver_combo)
        layout.addLayout(solver_layout)
        
        # Thermal Effect
        self.thermal_check = QtWidgets.QCheckBox("Include Thermal Effect")
        layout.addWidget(self.thermal_check)
        
        # Zero-strain Temperature
        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.addWidget(QtWidgets.QLabel("Zero-strain T(K):"))
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0, 1000)
        self.temp_spin.setValue(300)
        temp_layout.addWidget(self.temp_spin)
        layout.addLayout(temp_layout)
        
        # Buttons
        self.mesh_button = QtWidgets.QPushButton("Generate Mesh")
        self.mesh_button.clicked.connect(self.generate_mesh)
        layout.addWidget(self.mesh_button)
        
        self.thermal_button = QtWidgets.QPushButton("Thermal Analysis")
        self.thermal_button.clicked.connect(self.run_thermal_analysis)
        layout.addWidget(self.thermal_button)
        
        self.structural_button = QtWidgets.QPushButton("Structural Analysis")
        self.structural_button.clicked.connect(self.run_structural_analysis)
        layout.addWidget(self.structural_button)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
    
    def on_mesh_quality_changed(self, quality):
        """Update element count based on mesh quality"""
        element_counts = {"Very Coarse": 10000, "Coarse": 25000, "Normal": 50000, "Fine": 75000, "Very Fine": 100000}
        self.elements_spin.setValue(element_counts.get(quality, 10000))

    def generate_mesh(self):
        """Generate mesh and visualize with element colors"""
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "No Geometry", "Please load geometry first.")
            return
        
        # Create mesh
        t0 = time.time()
        mesher = HexMesher()
        mesher.createMeshFromSTLFile(self.parent.stl_geom.file_path, self.elements_spin.value())
        t1 = time.time()
        print(f"Mesh generation: {t1-t0:.2f} s")
        self.parent.hex_mesh = mesher

        t2 = time.time()
        

        # Prepare mesh for FEA
        self.prepare_mesh_for_analysis(mesher, "structural")

        t3 = time.time()
        print(f"Mesh preparation: {t3-t2:.2f} s")
        
        # Check if thermal loads exist
        if (hasattr(self.parent, 'thermal_loads_window') and 
            self.parent.thermal_loads_window and
            self.parent.thermal_loads_window.thermal_loads and
            any(self.parent.thermal_loads_window.thermal_loads.get(key, []) for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])):
            # Show thermal colored mesh
            self.visualize_colored_mesh("thermal")
        else:
            # Show structural colored mesh
            self.visualize_colored_mesh("structural")
            t4 = time.time()
            print(f"Mesh plotting: {t4-t3:.2f} s")
        
        # Update state
        self.parent.update_LivVar('mesh_generated', True)
        self.parent.message_text.append(f"Mesh generated: {mesher.num_elems} elements, {mesher.num_nodes} nodes")

    def get_boundary_mapping_data(self):
        """Get common boundary mapping data used by multiple methods"""
        mesh = self.parent.hex_mesh
        boundary_nodes = mesh.get_boundary_nodes()
        boundary_points = mesh.node_xyz[boundary_nodes]
        tolerance = min(mesh.elem_size) * 0.9
        return boundary_nodes, boundary_points, tolerance

    def map_triangles_to_surface_nodes(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None):
        """Map triangles to surface nodes - base method for both element and thermal node mapping"""
        if not triangle_indices:
            return set()
        
        # Use cached boundary data if not provided
        if boundary_nodes is None:
            boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        surface_nodes = set()
        for tri_idx in triangle_indices:
            distances = self.parent.stl_geom.find_points_triangle_distances_vectorized(boundary_points, tri_idx)
            close_mask = distances < tolerance
            surface_nodes.update(boundary_nodes[close_mask])
        
        return surface_nodes
    
    #introduced new method to build node to element map
    def build_node_to_elem_map(self, mesh):
        """Build a mapping from node index to set of element indices."""
        from collections import defaultdict
        node_to_elem = defaultdict(set)
        elemArray = np.asarray(mesh.elemArray)
        for elem_id, nodes in enumerate(elemArray):
            for node in nodes:
                node_to_elem[node].add(elem_id)
        return node_to_elem

    def map_triangles_to_elements(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None, node_to_elem=None):
        surface_nodes = self.map_triangles_to_surface_nodes(triangle_indices, boundary_nodes, boundary_points, tolerance)
        if node_to_elem is None:
            node_to_elem = self.build_node_to_elem_map(self.parent.hex_mesh)
        # Union all elements containing any of the surface nodes
        surface_elements = set()
        for node_id in surface_nodes:
            surface_elements.update(node_to_elem[node_id])
        return surface_elements

    def map_triangles_to_thermal_nodes(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None):
        """Map triangles to thermal nodes"""
        return list(self.map_triangles_to_surface_nodes(triangle_indices, boundary_nodes, boundary_points, tolerance))

    def get_element_colors(self, color_type="structural"):
        mesh = self.parent.hex_mesh
        element_colors = np.full(mesh.num_elems, np.nan)  # Default gray

        # Build node-to-element map once
        node_to_elem = self.build_node_to_elem_map(mesh)
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()

        if color_type == "structural":
            if self.parent.constrained_triangles:
                constrained_elements = self.map_triangles_to_elements(
                    list(self.parent.constrained_triangles), boundary_nodes, boundary_points, tolerance, node_to_elem
                )
                element_colors[list(constrained_elements)] = 0.0  # Black

            for force_info in self.parent.force_data:
                force_elements = self.map_triangles_to_elements(
                    force_info['triangles'], boundary_nodes, boundary_points, tolerance, node_to_elem
                )
                element_colors[list(force_elements)] = 1.0  # Red

        elif color_type == "thermal":
            if not (self.parent.thermal_loads_window and self.parent.thermal_loads_window.thermal_loads):
                return element_colors

            thermal_loads = self.parent.thermal_loads_window.thermal_loads
            thermal_color_map = {
                'fixed_temps': 0.0,      # Blue
                'heat_sources': 0.5,     # Orange  
                'total_heat_sources': 1.0 # Red
            }
            for load_type, color_value in thermal_color_map.items():
                for load_data in thermal_loads.get(load_type, []):
                    affected_elements = self.map_triangles_to_elements(
                        load_data['triangles'], boundary_nodes, boundary_points, tolerance, node_to_elem
                    )
                    element_colors[list(affected_elements)] = color_value

        return element_colors

    def visualize_colored_mesh(self, visualization_type="structural"):
        """Mesh visualization for both structural and thermal analysis"""
        
        # Clear actors except geometry info
        for name in list(self.parent.plotter.actors.keys()):
            if name != 'geometry_info':
                self.parent.plotter.remove_actor(name, reset_camera=False)
        
        # Get appropriate element colors
        element_colors = self.get_element_colors(visualization_type)
        
        # Create colored mesh polydata
        mesh_polydata = self.create_mesh_polydata(element_colors)
        
        # Define visualization parameters based on type
        viz_params = {
            "structural": {
                "cmap": "viridis",
                "name": "colored_mesh"
            },
            "thermal": {
                "cmap": "coolwarm", 
                "name": "thermal_mesh"
            }
        }
        
        params = viz_params.get(visualization_type, viz_params["structural"])
        
        # Add to plotter
        self.parent.plotter.add_mesh(
            mesh_polydata,
            scalars='element_colors',
            cmap=params["cmap"],
            clim=[0, 1],
            show_edges=True,
            edge_color='black',
            line_width=1,
            show_scalar_bar=False,
            name=params["name"]
        )
        
        self.parent.plotter.reset_camera()

    # def create_mesh_polydata(self, element_colors):
    #     mesh = self.parent.hex_mesh

    #     t0 = time.time()
    #     # Vectorized cell array: shape (num_elems, 9), first column is 8, rest are node indices
    #     num_elems = mesh.num_elems
    #     elemArray = np.asarray(mesh.elemArray, dtype=np.int64)
    #     cells = np.hstack([np.full((num_elems, 1), 8, dtype=np.int64), elemArray])
    #     cells = cells.flatten()
    #     t1 = time.time()
    #     print(f"[PROFILE] Vectorized cell array: {t1-t0:.4f} s")

    #     grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON] * num_elems, mesh.node_xyz)
    #     grid.cell_data['element_colors'] = element_colors

    #     t2 = time.time()
    #     print(f"[PROFILE] UnstructuredGrid creation: {t2-t1:.4f} s")

    #     surf = grid.extract_surface()
    #     t3 = time.time()
    #     print(f"[PROFILE] extract_surface: {t3-t2:.4f} s")

    #     return surf

    def create_mesh_polydata(self, element_colors):
        mesh = self.parent.hex_mesh
        num_elems = mesh.num_elems
        elemArray = np.asarray(mesh.elemArray, dtype=np.int64)
        cells = np.hstack([np.full((num_elems, 1), 8, dtype=np.int64), elemArray])
        cells = cells.flatten()
        grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON] * num_elems, mesh.node_xyz)
        grid.cell_data['element_colors'] = element_colors
        return grid  # Do not call extract_surface()

    def prepare_mesh_for_analysis(self, mesh, analysis_type="structural"):
        """Mesh preparation for both structural and thermal FEA"""

        t_start = time.time()

        # Common attributes for both analysis types
        mesh.elemPseudoDensity = np.ones(mesh.num_elems)
        mesh.elemNeighborsArray = np.zeros((mesh.num_elems, 6), dtype=int)
        mesh.node_indices = np.zeros((mesh.num_nodes, 4), dtype=int)
        mesh.elemComponentId = np.zeros(mesh.num_elems, dtype=int)
        t_common = time.time()
        print(f"[PROFILE] Common attributes: {t_common - t_start:.4f} s")

        if analysis_type == "thermal":
            # Thermal: 1 DOF per node
            mesh.edofMat = np.zeros((mesh.num_elems, 8), dtype=int)
            for elem_id in range(mesh.num_elems):
                mesh.edofMat[elem_id] = mesh.elemArray[elem_id]
            t_edof = time.time()
            print(f"[PROFILE] Thermal edofMat: {t_edof - t_common:.4f} s")

            # Create node_idx for thermal FEA
            row_indices, col_indices = [], []
            for elem_id in range(mesh.num_elems):
                element_nodes = mesh.elemArray[elem_id]
                for node_i in element_nodes:
                    for node_j in element_nodes:
                        row_indices.append(node_i)
                        col_indices.append(node_j)
            mesh.node_idx = np.column_stack((row_indices, col_indices))
            t_nodeidx = time.time()
            print(f"[PROFILE] Thermal node_idx: {t_nodeidx - t_edof:.4f} s")

        else:  # structural
            # Structural: 3 DOFs per node
            mesh.edofMat = np.zeros((mesh.num_elems, 24), dtype=int)
            for elem_id in range(mesh.num_elems):
                element_nodes = mesh.elemArray[elem_id]
                dofs = []
                for node in element_nodes:
                    dofs.extend([3*node, 3*node+1, 3*node+2])
                mesh.edofMat[elem_id] = dofs
            t_edof = time.time()
            print(f"[PROFILE] Structural edofMat: {t_edof - t_common:.4f} s")

        # Common element-to-node mapping
        self.create_element_to_node_mapping(mesh)
        t_mapping = time.time()
        print(f"[PROFILE] Element-to-node mapping: {t_mapping - (t_nodeidx if analysis_type=='thermal' else t_edof):.4f} s")
        print(f"[PROFILE] Total prepare_mesh_for_analysis: {t_mapping - t_start:.4f} s")

    def create_element_to_node_mapping(self, mesh):
        """Create element to node field mapping matrix (common for both analysis types)"""
        import time
        t0 = time.time()

        row_indices, col_indices, data = [], [], []
        weight = 1.0 / 8.0  # Equal weight for 8 nodes per hex element

        t1 = time.time()
        for elem_id in range(mesh.num_elems):
            element_nodes = mesh.elemArray[elem_id]
            for node in element_nodes:
                row_indices.append(node)
                col_indices.append(elem_id)
                data.append(weight)
        t2 = time.time()
        print(f"[PROFILE] Loop over elements: {t2 - t1:.4f} s")

        mesh.elem_to_node_field_mapping = coo_matrix(
            (data, (row_indices, col_indices)), 
            shape=(mesh.num_nodes, mesh.num_elems)
        ).tocsr()
        t3 = time.time()
        print(f"[PROFILE] Sparse matrix construction: {t3 - t2:.4f} s")
        print(f"[PROFILE] Total create_element_to_node_mapping: {t3 - t0:.4f} s")

    def create_material_properties(self):
        """Create material properties object for both analysis types"""
        material_props = self.parent.applied_material['properties']
        return mat_lib.Material(
            name="Applied_Material",
            youngs_modulus=material_props['Young\'s Modulus'],
            poissons_ratio=material_props['Poisson\'s Ratio'],
            mass_density=material_props['Density'],
            thermal_conductivity=material_props['Thermal Conductivity'],
            specific_heat=material_props['Specific Heat Capacity'],
            thermal_expansion=material_props['Thermal Expansion'],
            cost=material_props['Price'],
            yield_strength=material_props['Yield Strength']
        )

    def get_solver(self):
        """Get solver based on current selection"""
        solver_map = {
            "PARDISO": linear_solvers.Solvers.PARDISO,
            "DPCG": linear_solvers.Solvers.DPCG,
            "PCG": linear_solvers.Solvers.PCG,
            "PYAMG": linear_solvers.Solvers.PYAMG,
            "SPSOLVE": linear_solvers.Solvers.SPSOLVE
        }
        return solver_map[self.solver_combo.currentText()]

    def run_structural_analysis(self):
        """Run structural analysis using existing mesh and boundary conditions"""
        if not self.parent.hex_mesh:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return
        
        if not self.parent.applied_material:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return
        
        # Prepare mesh for structural analysis
        self.prepare_mesh_for_analysis(self.parent.hex_mesh, "structural")
        
        # Get boundary mapping data
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        # Map constrained triangles to fixed nodes
        fixed_nodes = {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
        
        if self.parent.constrained_triangles and self.parent.constraint_data:
            constrained_elements = self.map_triangles_to_elements(
                list(self.parent.constrained_triangles), boundary_nodes, boundary_points, tolerance
            )
            
            # Convert elements to nodes
            constrained_nodes = set()
            for elem_id in constrained_elements:
                constrained_nodes.update(self.parent.hex_mesh.elemArray[elem_id])
            
            # Apply constraints based on constraint_data
            for constraint in self.parent.constraint_data:
                constraint_type = constraint['type']
                if constraint_type == 'Fixed XYZ':
                    fixed_nodes['xyz'].update(constrained_nodes)
                elif constraint_type == 'Fixed X':
                    fixed_nodes['x'].update(constrained_nodes)
                elif constraint_type == 'Fixed Y':
                    fixed_nodes['y'].update(constrained_nodes)
                elif constraint_type == 'Fixed Z':
                    fixed_nodes['z'].update(constrained_nodes)
        
        # Map force triangles to load nodes
        load_nodes_groups = []
        load_forces = []
        
        for force_info in self.parent.force_data:
            force_elements = self.map_triangles_to_elements(
                force_info['triangles'], boundary_nodes, boundary_points, tolerance
            )
            
            # Convert elements to nodes
            force_nodes = set()
            for elem_id in force_elements:
                force_nodes.update(self.parent.hex_mesh.elemArray[elem_id])
            
            if force_nodes:
                load_nodes_groups.append(list(force_nodes))
                load_forces.append([force_info['force_x'], force_info['force_y'], force_info['force_z']])
        
        # Prepare load data for solver
        load_data = {
            'load_nodes_groups': load_nodes_groups,
            'load_forces': load_forces
        }
        
        # Process data for solver
        mesh, mat_prop, bc = self.process_data_for_solver(
            self.parent.hex_mesh,
            fixed_nodes,
            load_data,
            self.parent.applied_material['properties']
        )
        
        # Create FEA solver
        fe_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=self.get_solver(),
            rtol=1e-8
        )
        
        # Solve
        self.parent.message_text.append("Starting structural analysis...")
        solution = fe_solver.solve()
        fe_solver.postprocess()

        # Store solver for visualization
        self.parent.fe_solver = fe_solver
        
        if 'colored_mesh' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('colored_mesh', reset_camera=False)

        scale_info = fe_solver.plot_deformation(plotter=self.parent.plotter)
        if scale_info:
            self.parent.message_text.append(scale_info)
        
        # Store results
        self.parent.analysis_results = {
            'displacement': solution,
            'stress': fe_solver.vonMisesStress,
            'max_deformation': fe_solver.max_deformation,
            'max_stress': np.max(fe_solver.vonMisesStress)
        }
        
        # Update UI
        self.parent.update_LivVar('analysis.structural', True)
        self.parent.update_LivVar('analysis.performed', True)
        
        self.parent.message_text.append(f"Analysis complete. Max deformation: {fe_solver.max_deformation:.4e}")
        self.parent.message_text.append(f"Max von Mises stress: {np.max(fe_solver.vonMisesStress):.4e}")

    def run_thermal_analysis(self):
        """Run thermal analysis using existing mesh and thermal boundary conditions"""
        # Validate prerequisites
        if not self.parent.hex_mesh:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return
        
        if not self.parent.applied_material:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return
        
        # Check thermal loads
        if not self.parent.thermal_loads_window or not self.parent.thermal_loads_window.thermal_loads:
            QtWidgets.QMessageBox.warning(self, "No Thermal Loads", "Please apply thermal loads first.")
            return
        
        thermal_loads = self.parent.thermal_loads_window.thermal_loads
        has_loads = any(thermal_loads.get(key, []) for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])
        if not has_loads:
            QtWidgets.QMessageBox.warning(self, "No Thermal Loads", "Please apply thermal loads first.")
            return
        
        # Prepare mesh for thermal FEA
        self.prepare_mesh_for_analysis(self.parent.hex_mesh, "thermal")
        
        # Get boundary mapping data
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        # Process thermal boundary conditions
        thermal_bc = self.process_thermal_boundary_conditions(
            self.parent.hex_mesh, thermal_loads, boundary_nodes, boundary_points, tolerance
        )
        
        # Create material properties and solver
        mat_prop = self.create_material_properties()
        solver = self.get_solver()
        
        # Create and solve thermal FEA
        fe_solver = HexThermalFEA(mesh=self.parent.hex_mesh, mat_prop=mat_prop, bc=thermal_bc, solver=solver)
        
        self.parent.message_text.append("Starting thermal analysis...")
        temperature_solution = fe_solver.solve()
        fe_solver.postprocess()
        
        # Store solver and results
        self.parent.thermal_fe_solver = fe_solver
        self.parent.thermal_results = {
            'temperature': temperature_solution,
            'heat_flux': fe_solver.strain if fe_solver.strain is not None else None,
            'max_temperature': np.max(temperature_solution),
            'min_temperature': np.min(temperature_solution)
        }
        
        # Clear colored mesh and use thermal FEA's built-in visualization
        if 'colored_mesh' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('colored_mesh', reset_camera=False)

        if 'stl_geometry' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('stl_geometry', reset_camera=False)
        
        # Use thermal FEA's built-in plot_temperature method
        temp_info = fe_solver.plot_temperature(self.parent.plotter)
        if temp_info:
            self.parent.message_text.append(temp_info)
        
        # Update UI state
        self.parent.update_LivVar('analysis.thermal', True)
        self.parent.update_LivVar('analysis.performed', True)
        
        # Success messages
        self.parent.message_text.append("Thermal analysis complete.")
        self.parent.message_text.append(f"Temperature range: {np.min(temperature_solution):.2f} - {np.max(temperature_solution):.2f} K")

    def process_thermal_boundary_conditions(self, mesh, thermal_loads, boundary_nodes, boundary_points, tolerance):
        """Process thermal loads and create boundary conditions"""
        thermal_force = np.zeros(mesh.num_nodes)
        fixed_temp_nodes = []
        fixed_temp_values = []
        
        # Process fixed temperature BCs
        for temp_load in thermal_loads.get('fixed_temps', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                temp_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            fixed_temp_nodes.extend(affected_nodes)
            fixed_temp_values.extend([temp_load['temperature']] * len(affected_nodes))
        
        # Process heat flux BCs
        for heat_load in thermal_loads.get('heat_sources', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                heat_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            if affected_nodes:
                heat_per_node = heat_load['heat_flux'] / len(affected_nodes)
                for node_id in affected_nodes:
                    thermal_force[node_id] += heat_per_node
        
        # Process total heat BCs
        for total_heat_load in thermal_loads.get('total_heat_sources', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                total_heat_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            if affected_nodes:
                heat_per_node = total_heat_load['total_heat'] / len(affected_nodes)
                for node_id in affected_nodes:
                    thermal_force[node_id] += heat_per_node
        
        # Create boundary conditions
        fixed_temp_nodes = np.array(fixed_temp_nodes, dtype=int) if fixed_temp_nodes else np.array([], dtype=int)
        fixed_temp_values = np.array(fixed_temp_values, dtype=float) if fixed_temp_values else np.array([], dtype=float)
        
        return bound_cond.BC(
            force=thermal_force,
            fixed_dofs=fixed_temp_nodes,
            dirichlet_values=fixed_temp_values
        )

    @staticmethod
    def process_data_for_solver(existing_mesh, fixed_nodes, load_data, material_props):
        """Process mesh and boundary condition data for structural solver"""
        mesh = existing_mesh
        
        # Process fixed nodes
        fixed_dofs = []
        for node in fixed_nodes['xyz']:
            fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
            
        for node in fixed_nodes['x']:
            fixed_dofs.append(3*node)
            
        for node in fixed_nodes['y']:
            fixed_dofs.append(3*node + 1)
            
        for node in fixed_nodes['z']:
            fixed_dofs.append(3*node + 2)
            
        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
        
        # Process loads
        force = np.zeros(3*mesh.num_nodes)
        for nodes, force_vector in zip(load_data['load_nodes_groups'], load_data['load_forces']):
            if nodes:
                force_per_node = np.array(force_vector) / len(nodes)
                for node in nodes:
                    force[3*node:3*node + 3] += force_per_node

        # Create boundary conditions and material properties
        bc = bound_cond.BC(force=force,
                        fixed_dofs=fixed_dofs,
                        dirichlet_values=dirichlet_values)
        
        mat_prop = mat_lib.Material(
            name="Applied_Material",
            youngs_modulus=material_props['Young\'s Modulus'],
            poissons_ratio=material_props['Poisson\'s Ratio'],
            mass_density=material_props['Density'],
            thermal_conductivity=material_props['Thermal Conductivity'],
            specific_heat=material_props['Specific Heat Capacity'],
            thermal_expansion=material_props['Thermal Expansion'],
            cost=material_props['Price'],
            yield_strength=material_props['Yield Strength']
        )
        
        return mesh, mat_prop, bc
#----------------------------------------------------------------------------
class TopOptConstraintsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("TopOpt Constraints")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(300, 600)
        self.parent = parent
        self.applied = False
        
        # Initialize constraint actors
        if not hasattr(self.parent, 'topopt_constraint_actors'):
            self.parent.topopt_constraint_actors = {}
        
        # Define constraint configuration
        self.constraint_config = {
            'manufacturing': [
                ('extrude', 'Extrude', 'combo', ["XDir", "YDir", "ZDir"]),
                ('am_build', 'AM Build', 'combo', ["+XDir", "+YDir", "+ZDir", "-XDir", "-YDir", "-ZDir"]),
                ('draw_direction', 'Draw Direction', 'combo', ["XDir", "YDir", "ZDir"]),
                ('cyclic_symmetry', 'CyclicSym(Z)', 'combo', ["(2) 180 deg", "(3) 120 deg", "(4) 90 deg", "(5) 72 deg", "(6) 60 deg", "(7) 51 deg","(8) 45 deg"])
            ],
            'patterns': [
                ('x_grid', 'XGridPattern', 'spin', (1, 10, 2)),
                ('y_grid', 'YGridPattern', 'spin', (1, 10, 2)),
                ('z_grid', 'ZGridPattern', 'spin', (1, 10, 2))
            ],
            'performance': [
                ('stress_safety', 'StressSafety', 'double_spin', (0.1, 10.0, 1.0)),
                ('max_displacement', 'MaxDisp(m)', 'double_spin', (0, 1000, 160.0, 6)),
                ('min_frequency', 'MinFreq(Hz)', 'double_spin', (0, 10000, 1000.0)),
                ('max_temperature', 'MaxTemp(K)', 'double_spin', (0, 5000, 2000.0))
            ],
            'symmetry': [
                ('x_symmetry', 'X-Symmetry', 'check_only'),
                ('y_symmetry', 'Y-Symmetry', 'check_only'),
                ('z_symmetry', 'Z-Symmetry', 'check_only')
            ],
            'other': [
                ('connected_topology', 'Connected Topology', 'check_only', True),
                ('keep_fixed_faces', 'Keep Fixed Faces', 'check_only')
            ]
        }
        
        self.widgets = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        container = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(container)
        
        # Create widgets from configuration
        for category, constraints in self.constraint_config.items():
            for constraint in constraints:
                self.create_constraint_widget(constraint, form_layout)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Apply button
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_constraints)
        layout.addWidget(apply_button)
        
        self.connect_signals()

        self.connect_signals()

        # Restore checkbox states from parent.topopt_constraints if available
        if hasattr(self.parent, 'topopt_constraints') and self.parent.topopt_constraints:
            for category, constraint_data in self.parent.topopt_constraints.items():
                for constraint_key, constraint_value in constraint_data.items():
                    if isinstance(constraint_value, bool):
                        widget_name = f'{constraint_key}_check'
                        if widget_name in self.widgets:
                            self.widgets[widget_name].setChecked(constraint_value)
                    elif isinstance(constraint_value, dict):
                        check_widget = f'{constraint_key}_check'
                        if check_widget in self.widgets:
                            self.widgets[check_widget].setChecked(constraint_value.get('enabled', False))
                        if 'value' in constraint_value:
                            for widget_type in ['combo', 'spin']:
                                value_widget = f'{constraint_key}_{widget_type}'
                                if value_widget in self.widgets:
                                    widget = self.widgets[value_widget]
                                    if widget_type == 'combo':
                                        widget.setCurrentText(str(constraint_value['value']))
                                    else:
                                        widget.setValue(constraint_value['value'])
                                    break
    
    def create_constraint_widget(self, constraint_def, layout):
        """Create widget based on constraint definition"""
        key = constraint_def[0]
        label = constraint_def[1]
        widget_type = constraint_def[2]
        
        # Create checkbox
        checkbox = QtWidgets.QCheckBox(label)
        self.widgets[f'{key}_check'] = checkbox
        
        if widget_type == 'combo':
            # Combo box widget
            combo = QtWidgets.QComboBox()
            combo.addItems(constraint_def[3])
            self.widgets[f'{key}_combo'] = combo
            layout.addRow(checkbox, combo)
            
        elif widget_type == 'spin':
            # Spin box widget
            spin = QtWidgets.QSpinBox()
            min_val, max_val, default = constraint_def[3]
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            self.widgets[f'{key}_spin'] = spin
            layout.addRow(checkbox, spin)
            
        elif widget_type == 'double_spin':
            # Double spin box widget
            spin = QtWidgets.QDoubleSpinBox()
            params = constraint_def[3]
            spin.setRange(params[0], params[1])
            spin.setValue(params[2])
            if len(params) > 3:  # decimals specified
                spin.setDecimals(params[3])
            self.widgets[f'{key}_spin'] = spin
            layout.addRow(checkbox, spin)
            
        elif widget_type == 'check_only':
            # Checkbox only
            if len(constraint_def) > 3 and constraint_def[3]:  # default checked
                checkbox.setChecked(True)
            layout.addRow(checkbox)
    
    def connect_signals(self):
        """Connect all widget signals to update methods"""
        for widget_name, widget in self.widgets.items():
            if 'check' in widget_name:
                widget.stateChanged.connect(self.update_visualizations)
            elif 'combo' in widget_name or 'spin' in widget_name:
                if hasattr(widget, 'currentTextChanged'):
                    widget.currentTextChanged.connect(self.update_visualizations)
                elif hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self.update_visualizations)
    
    def update_visualizations(self):
        """Update all constraint visualizations based on current widget states"""
        # Clear all existing visualizations
        self.clear_all_visualizations()
        
        if not self.parent.stl_geom:
            return
            
        bbox = self.parent.stl_geom.get_bounding_box()
        center = [(bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, (bbox[4] + bbox[5])/2]
        
        # Update each constraint type
        self.update_arrow_constraints(bbox, center)
        self.update_grid_patterns(bbox)
        self.update_symmetry_planes(bbox, center)
        
        self.parent.plotter.render()
    
    def update_arrow_constraints(self, bbox, center):
        """Update arrow-based constraints (extrude, draw_direction)"""
        arrow_configs = {
            'extrude': {'colors': {'XDir': 'red', 'YDir': 'green', 'ZDir': 'blue'}, 'bidirectional': False},
            'draw_direction': {'colors': {'XDir': 'green', 'YDir': 'green', 'ZDir': 'green'}, 'bidirectional': True}
        }
        
        for constraint_key, config in arrow_configs.items():
            if self.get_widget_value(f'{constraint_key}_check'):
                direction_key = self.get_widget_value(f'{constraint_key}_combo')
                self.create_directional_arrows(constraint_key, direction_key, bbox, center, config)
    
    def create_directional_arrows(self, constraint_key, direction_key, bbox, center, config):
        """Create arrows for directional constraints"""
        directions = {"XDir": [1, 0, 0], "YDir": [0, 1, 0], "ZDir": [0, 0, 1]}
        sizes = {"XDir": bbox[1] - bbox[0], "YDir": bbox[3] - bbox[2], "ZDir": bbox[5] - bbox[4]}
        
        direction = directions[direction_key]
        color = config['colors'][direction_key]
        size = sizes[direction_key]
        
        actors = []
        
        if config['bidirectional']:
            # Create two arrows from center
            half_size = size / 2
            actors.append(self.create_arrow_actor(center, direction, color, half_size))
            actors.append(self.create_arrow_actor(center, [-d for d in direction], color, half_size))
        else:
            # Create single arrow from boundary
            start_pos = self.get_boundary_start_position(direction_key, bbox, center)
            actors.append(self.create_arrow_actor(start_pos, direction, color, size))
        
        self.parent.topopt_constraint_actors[constraint_key] = actors if len(actors) > 1 else actors[0]
    
    def get_boundary_start_position(self, direction_key, bbox, center):
        """Get starting position at model boundary for arrows"""
        positions = {
            "XDir": [bbox[0], center[1], center[2]],
            "YDir": [center[0], bbox[2], center[2]],
            "ZDir": [center[0], center[1], bbox[4]]
        }
        return positions[direction_key]
    
    def update_grid_patterns(self, bbox):
        """Update grid pattern visualizations"""
        any_grid_enabled = any(self.get_widget_value(f'{axis}_grid_check') for axis in ['x', 'y', 'z'])
        
        if any_grid_enabled:
            # Create bounding box
            box = pv.Box(bounds=bbox)
            actor = self.parent.plotter.add_mesh(box, style='wireframe', color='gray', line_width=2)
            self.parent.topopt_constraint_actors['bounding_box'] = actor
            
            # Create grid planes for each enabled axis
            grid_config = {
                'x': {'normal': [1, 0, 0], 'color': 'red', 'bbox_indices': (0, 1)},
                'y': {'normal': [0, 1, 0], 'color': 'green', 'bbox_indices': (2, 3)},
                'z': {'normal': [0, 0, 1], 'color': 'blue', 'bbox_indices': (4, 5)}
            }
            
            for axis, config in grid_config.items():
                if self.get_widget_value(f'{axis}_grid_check'):
                    divisions = self.get_widget_value(f'{axis}_grid_spin')
                    actors = self.create_grid_planes_for_axis(axis, divisions, bbox, config)
                    self.parent.topopt_constraint_actors[f'{axis}_grid'] = actors
    
    def create_grid_planes_for_axis(self, axis, divisions, bbox, config):
        """Create grid planes for a specific axis"""
        actors = []
        min_pos, max_pos = config['bbox_indices']
        
        for i in range(divisions):
            if divisions == 1:
                pos = (bbox[min_pos] + bbox[max_pos]) / 2
            else:
                pos = bbox[min_pos] + (bbox[max_pos] - bbox[min_pos]) * i / (divisions - 1)
            
            center = self.get_plane_center(axis, pos, bbox)
            size = self.get_plane_size(axis, bbox)
            actor = self.create_plane_actor(center, config['normal'], size, config['color'])
            actors.append(actor)
        
        return actors
    
    def get_plane_center(self, axis, position, bbox):
        """Get center position for plane based on axis"""
        centers = {
            'x': [position, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2],
            'y': [(bbox[0]+bbox[1])/2, position, (bbox[4]+bbox[5])/2],
            'z': [(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, position]
        }
        return centers[axis]
    
    def get_plane_size(self, axis, bbox):
        """Get appropriate size for plane based on axis"""
        sizes = {
            'x': max(bbox[3] - bbox[2], bbox[5] - bbox[4]) * 1.1,
            'y': max(bbox[1] - bbox[0], bbox[5] - bbox[4]) * 1.1,
            'z': max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 1.1
        }
        return sizes[axis]
    
    def update_symmetry_planes(self, bbox, center):
        """Update symmetry plane visualizations"""
        symmetry_config = {
            'x_symmetry': {'normal': [1, 0, 0], 'color': 'red'},
            'y_symmetry': {'normal': [0, 1, 0], 'color': 'green'},
            'z_symmetry': {'normal': [0, 0, 1], 'color': 'blue'}
        }
        
        size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]) * 1.2
        
        for symmetry_key, config in symmetry_config.items():
            if self.get_widget_value(f'{symmetry_key}_check'):
                actor = self.create_plane_actor(center, config['normal'], size, config['color'])
                self.parent.topopt_constraint_actors[symmetry_key] = actor
        
        # Handle cyclic symmetry - UPDATED VERSION
        if self.get_widget_value('cyclic_symmetry_check'):
            angle_text = self.get_widget_value('cyclic_symmetry_combo')
            
            # Parse the number of planes from the text
            # Format: "(2) 180 deg", "(3) 120 deg", etc.
            num_planes = int(angle_text.split('(')[1].split(')')[0])
            
            actors = []
            
            # Calculate adaptive dimensions
            x_size = bbox[1] - bbox[0]
            y_size = bbox[3] - bbox[2] 
            z_size = bbox[5] - bbox[4]
            max_radius = max(x_size, y_size) / 2
            
            # Create the specified number of planes starting from center
            for i in range(num_planes):
                # Calculate angle for each plane (evenly distributed around Z-axis)
                angle = (2 * np.pi * i) / num_planes
                
                # Create normal vector in XY plane (rotating around Z-axis)
                normal = [np.cos(angle), np.sin(angle), 0]
                
                # Create adaptive plane that extends from center to boundary
                actor = self.create_cyclic_plane(center, normal, bbox, 'cyan')
                actors.append(actor)
            
            self.parent.topopt_constraint_actors['cyclic_symmetry'] = actors

    def create_cyclic_plane(self, center, normal, bbox, color):
        """Create plane for cyclic symmetry"""
        # Calculate bounding box dimensions
        x_size = bbox[1] - bbox[0]
        y_size = bbox[3] - bbox[2]
        z_size = bbox[5] - bbox[4]
        
        max_radius = max(x_size, y_size) / 2 * 1.5 
        
        # Create a vertical plane that extends radially from center
        plane_points = vtk.vtkPoints()
        
        # Rotate the normal 180 degrees (opposite direction)
        rotated_normal = [-normal[0], -normal[1], 0]
        
        # Calculate plane height (vertical extent in Z direction)
        plane_height = z_size * 1.2  # Extend beyond model bounds
        
        # Define the four corners of the vertical radial plane
        # Bottom center point
        plane_points.InsertNextPoint(center[0], center[1], center[2] - plane_height/2)
        
        # Top center point  
        plane_points.InsertNextPoint(center[0], center[1], center[2] + plane_height/2)
        
        # Bottom outer point (along rotated normal direction)
        outer_x = center[0] + max_radius * rotated_normal[0]
        outer_y = center[1] + max_radius * rotated_normal[1]
        plane_points.InsertNextPoint(outer_x, outer_y, center[2] - plane_height/2)
        
        # Top outer point
        plane_points.InsertNextPoint(outer_x, outer_y, center[2] + plane_height/2)
        
        # Create the quad (rectangle) - vertical plane from center outward
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, 0)  # Bottom center
        quad.GetPointIds().SetId(1, 2)  # Bottom outer
        quad.GetPointIds().SetId(2, 3)  # Top outer  
        quad.GetPointIds().SetId(3, 1)  # Top center
        
        # Create cell array and add the quad
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(plane_points)
        polydata.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set appearance - make more visible
        color_rgb = self._get_color_rgb(color)
        actor.GetProperty().SetColor(*color_rgb)
        actor.GetProperty().SetOpacity(0.6)  # Increased opacity for better visibility
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(3)  # Thicker edge lines
        
        # Enable two-sided rendering so plane is visible from both sides
        actor.GetProperty().SetBackfaceCulling(False)
        actor.GetProperty().SetFrontfaceCulling(False)
        
        # Add lighting for better visibility
        actor.GetProperty().SetInterpolationToGouraud()
        
        self.parent.plotter.add_actor(actor)
        return actor
    
    def get_widget_value(self, widget_name):
        """Get value from any widget type"""
        widget = self.widgets.get(widget_name)
        if not widget:
            return None
            
        if hasattr(widget, 'isChecked'):
            return widget.isChecked()
        elif hasattr(widget, 'currentText'):
            return widget.currentText()
        elif hasattr(widget, 'value'):
            return widget.value()
        return None
    
    def clear_all_visualizations(self):
        """Remove all constraint visualizations"""
        for actor_key, actor in list(self.parent.topopt_constraint_actors.items()):
            if actor:
                if isinstance(actor, list):
                    for a in actor:
                        self.parent.plotter.remove_actor(a, reset_camera=False)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
        
        self.parent.topopt_constraint_actors.clear()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if not self.applied:
            self.clear_all_visualizations()
            self.parent.plotter.render()
        event.accept()
    
    def create_arrow_actor(self, position, direction, color, scale):
        """Create an arrow actor"""
        if isinstance(scale, (list, tuple, np.ndarray)):
            scale = float(scale[0]) if len(scale) > 0 else 1.0
        else:
            scale = float(scale)

        bbox = self.parent.stl_geom.get_bounding_box()
        # model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        
        # shaft_radius = model_size * 0.002 
        # tip_radius = model_size * 0.004     
        # tip_length = model_size * 0.02

        #bbox = self.parent.stl_geom.get_bounding_box()
        # Calculate the diagonal length of the bounding box
        dx = bbox[1] - bbox[0]
        dy = bbox[3] - bbox[2]
        dz = bbox[5] - bbox[4]
        diag_length = (dx**2 + dy**2 + dz**2) ** 0.1

        # Use the diagonal length for scaling
        shaft_radius = diag_length * 0.01   # Increase as needed
        tip_radius = diag_length * 0.02     # Increase as needed
        tip_length = diag_length * 0.08     # Increase as needed

        arrow = pv.Arrow(
            start=position, 
            direction=direction, 
            scale=scale * 1.1,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            tip_length=tip_length
        )
        
        return self.parent.plotter.add_mesh(arrow, color=color)
        
    def create_plane_actor(self, center, normal, size, color):
        """Create a plane actor"""
        bbox = self.parent.stl_geom.get_bounding_box()
        x_size = (bbox[1] - bbox[0]) * 1.08
        y_size = (bbox[3] - bbox[2]) * 1.08
        z_size = (bbox[5] - bbox[4]) * 1.08
        
        if abs(normal[0]) > 0.5:
            i_size, j_size = y_size, z_size
        elif abs(normal[1]) > 0.5:
            i_size, j_size = x_size, z_size
        else:
            i_size, j_size = x_size, y_size
        
        plane = vtk.vtkPlaneSource()
        plane.SetCenter(center)
        plane.SetNormal(normal)
        
        # Calculate perpendicular vectors
        v1 = [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0]
        t1 = [normal[1] * v1[2] - normal[2] * v1[1],
              normal[2] * v1[0] - normal[0] * v1[2],
              normal[0] * v1[1] - normal[1] * v1[0]]
        
        mag = (t1[0]**2 + t1[1]**2 + t1[2]**2)**0.5
        t1 = [t1[0]/mag, t1[1]/mag, t1[2]/mag]
        
        t2 = [normal[1] * t1[2] - normal[2] * t1[1],
              normal[2] * t1[0] - normal[0] * t1[2],
              normal[0] * t1[1] - normal[1] * t1[0]]
        
        t1, t2 = t2, [-t1[0], -t1[1], -t1[2]]
        
        half_i, half_j = i_size / 2, j_size / 2
        
        origin = [center[0] - half_i * t1[0] - half_j * t2[0],
                  center[1] - half_i * t1[1] - half_j * t2[1],
                  center[2] - half_i * t1[2] - half_j * t2[2]]
        
        point1 = [center[0] + half_i * t1[0] - half_j * t2[0],
                  center[1] + half_i * t1[1] - half_j * t2[1],
                  center[2] + half_i * t1[2] - half_j * t2[2]]
        
        point2 = [center[0] - half_i * t1[0] + half_j * t2[0],
                  center[1] - half_i * t1[1] + half_j * t2[1],
                  center[2] - half_i * t1[2] + half_j * t2[2]]
        
        plane.SetOrigin(origin)
        plane.SetPoint1(point1)
        plane.SetPoint2(point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        color_rgb = self._get_color_rgb(color)
        actor.GetProperty().SetColor(*color_rgb)
        actor.GetProperty().SetOpacity(0.3)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(2)
        
        self.parent.plotter.add_actor(actor)
        return actor

    def _get_color_rgb(self, color_name):
        """Convert color name to RGB tuple"""
        color_map = {
            'red': (1.0, 0.0, 0.0), 'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
            'cyan': (0.0, 1.0, 1.0), 'magenta': (1.0, 0.0, 1.0), 'yellow': (1.0, 1.0, 0.0),
            'white': (1.0, 1.0, 1.0), 'black': (0.0, 0.0, 0.0)
        }
        return color_map.get(color_name, (0.5, 0.5, 0.5))
        
    def apply_constraints(self):
        """Apply all constraint settings"""
        constraints = {}
        
        # Build constraints dictionary from configuration
        for category, constraint_list in self.constraint_config.items():
            constraints[category] = {}
            for constraint_def in constraint_list:
                key = constraint_def[0]
                widget_type = constraint_def[2]
                
                if widget_type == 'check_only':
                    constraints[category][key] = self.get_widget_value(f'{key}_check')
                else:
                    constraints[category][key] = {
                        'enabled': self.get_widget_value(f'{key}_check'),
                        'value': self.get_widget_value(f'{key}_combo') or self.get_widget_value(f'{key}_spin')
                    }
        
        # Store constraints in parent
        self.parent.topopt_constraints = constraints
        self.applied = True
        
        # FIX: Update UI state 
        self.parent.update_LivVar('topopt.constraints_defined', True)
        self.parent.set_sidebar_icon("TopOpt Constraints", "check")
        self.parent.set_sidebar_icon("Structural TopOpt", "arrow")
        self.parent.set_sidebar_icon("Thermal TopOpt", "arrow")
        self.parent.message_text.append(f"TopOpt constraints applied successfully. Constraints stored: {len(constraints)} categories")
        
        # ADD DEBUG: Print the LivVar state to verify
        print(f"DEBUG: LivVar topopt state: {self.parent.LivVar.get('topopt', {})}")

        # Notify display options window
        self.parent.notify_display_options_update()
#----------------------------------------------------------------------------
class StructuralTopOptWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Structural TopOpt")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(300, 400)
        self.parent = parent
        
        self.optimization_running = False
        self.optimization_thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Method selection
        method_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Method")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["DENSITY-MMA", "DENSITY-OC", "PARETO", "LEVELSET"])
        self.method_combo.setCurrentText("DENSITY-MMA")
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)
        
        # Volume fraction
        vol_layout = QtWidgets.QHBoxLayout()
        vol_label = QtWidgets.QLabel("Volume Fraction")
        self.vol_spinbox = QtWidgets.QDoubleSpinBox()
        self.vol_spinbox.setRange(0.1, 0.9)
        self.vol_spinbox.setSingleStep(0.05)
        self.vol_spinbox.setValue(0.5)
        self.vol_spinbox.setDecimals(2)
        vol_layout.addWidget(vol_label)
        vol_layout.addWidget(self.vol_spinbox)
        layout.addLayout(vol_layout)
        
        # Use all loads checkbox
        self.use_all_loads = QtWidgets.QCheckBox("Use all Loads?")
        self.use_all_loads.setChecked(True)
        layout.addWidget(self.use_all_loads)
        
        # Save Intermediate checkbox
        self.save_intermediate = QtWidgets.QCheckBox("Save Intermediate?")
        layout.addWidget(self.save_intermediate)
        
        # Optimize button
        self.optimize_button = QtWidgets.QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.optimize_button)
        
        # Stop button
        self.stop_button = QtWidgets.QPushButton("STOP OPTIMIZATION!")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #ff6b6b; }")
        layout.addWidget(self.stop_button)
        
        # Results display area
        self.results_label = QtWidgets.QLabel("Results will appear here")
        self.results_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        self.results_label.setWordWrap(True)
        self.results_label.setMinimumHeight(100)
        layout.addWidget(self.results_label)
        
        # Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def start_optimization(self):
        """Start the structural topology optimization process"""
        if not self.check_prerequisites():
            return
        
        method = self.method_combo.currentText()
        volume_fraction = self.vol_spinbox.value()
        use_all_loads = self.use_all_loads.isChecked()
        save_intermediate = self.save_intermediate.isChecked()
        
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.results_label.setText("Starting optimization...")
        
        self.parent.message_text.append(f"Starting structural topology optimization")
        self.parent.message_text.append(f"Method: {method}, Volume Fraction: {volume_fraction}")
        
        import threading
        self.optimization_thread = threading.Thread(
            target=self.run_optimization,
            args=(method, volume_fraction, use_all_loads, save_intermediate)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        return True
        
    def run_optimization(self, method, volume_fraction, use_all_loads, save_intermediate):
        """Run the topology optimization using the selected method"""
        
        # Create TO parameters
        to_params = TOParams()
        to_params.Objective = (TO_QOI.COMPLIANCE, "minimize", 1.0)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, "<=", volume_fraction)]
        to_params.APPLY_FILTER_TO_DENSITY = True
        to_params.APPLY_FILTER_TO_SENSITIVITY = True
        to_params.FILTER_RADIUS = 1.5
        
        # Apply topopt constraints
        self.apply_topopt_constraints_to_params(to_params)
        
        # Create FE solver
        fe_solver = self.create_fe_solver_for_topopt()
        
        QtCore.QMetaObject.invokeMethod(
            self.results_label,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, f"Running {method} topology optimization...\nThis may take several minutes.")
        )
        
        success = False
        error_msg = ""
        u, history, n_feas = None, None, 0
        
        try:
            if method == "DENSITY-MMA":
                u, history, success, error_msg, n_feas = topopt_mma(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    maxMMAIterations=100,
                    timeLimitSecs=3600,
                    move_limit=0.2,
                    kkt_tol=1e-6,
                    objective_tol=1e-4,
                    constraint_tol=1e-4,
                    print_progress=False,
                    plot_progress=False,
                    binarize_topology=True
                )
                
            elif method == "DENSITY-OC":
                u, history, success, error_msg, n_feas = topopt_optimality_criteria(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    maxIterations=250,
                    move=0.2,
                    move_tol=0.05,
                    rel_conv_tol=1e-4,
                    print_progress=False,
                    plot_progress=False,
                    binarize_topology=True
                )
                
            elif method == "PARETO":
                u, history, success, error_msg, n_feas = topopt_pareto(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    rel_err=0.02,
                    vol_decr_max=0.05,
                    vol_decr_min=0.0025,
                    min_local_iters=2,
                    max_local_iters=5,
                    print_progress=False,
                    plot_progress=False
                )
                
            elif method == "LEVELSET":
                u, history, success, error_msg, n_feas = topopt_levelset(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    maxIterations=250,
                    numReinit=10000,
                    plot_progress=False,
                    print_progress=False
                )
                
            else:
                error_msg = f"Method {method} not implemented"
                success = False
                
        except Exception as e:
            error_msg = f"Optimization failed with error: {str(e)}"
            success = False
        
        if self.optimization_running:
            QtCore.QMetaObject.invokeMethod(
                self,
                "optimization_completed",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, success),
                QtCore.Q_ARG(str, error_msg if not success else ""),
                QtCore.Q_ARG(object, history),
                QtCore.Q_ARG(object, u),
                QtCore.Q_ARG(object, fe_solver)
            )

    # def print_progress_to_message_box(self, msg):
    #     QtCore.QMetaObject.invokeMethod(
    #         self.parent.message_text,
    #         "append",
    #         QtCore.Qt.QueuedConnection,
    #         QtCore.Q_ARG(str, msg)
    #     )

    @QtCore.pyqtSlot(bool, str, object, object, object)
    def optimization_completed(self, success, error_msg, history, u, fe_solver):
        """Handle optimization completion"""
        self.optimization_running = False
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if success and history is not None:

            
            # Handle different history formats from different solvers
            if 'objective' in history:
                final_objective = history['objective'][-1]
                final_volume = history['volume'][-1]
                n_iterations = len(history['objective'])
            elif 'compliance' in history:  # Level set method
                final_objective = history['compliance'][-1]
                final_volume = history['volume'][-1]
                n_iterations = len(history['compliance'])
            else:
                final_objective = 0.0
                final_volume = 0.0
                n_iterations = 0
            
            results_text = (
                f"Optimization completed successfully!\n\n"
                f"Method: {self.method_combo.currentText()}\n"
                f"Final objective: {final_objective:.6e}\n"
                f"Final volume fraction: {final_volume:.3f}\n"
                f"Iterations: {n_iterations}\n"
                f"Target volume: {self.vol_spinbox.value():.3f}"
            )
            
            self.results_label.setText(results_text)
            
            self.parent.message_text.append("Structural topology optimization completed successfully")
            self.parent.message_text.append(f"Method: {self.method_combo.currentText()}")
            self.parent.message_text.append(f"Final objective: {final_objective:.6e}, Volume fraction: {final_volume:.3f}")
            
            self.parent.update_LivVar('topopt.structural_performed', True)
            self.parent.set_sidebar_icon("Structural TopOpt", "check")
            
            self.parent.topopt_results = {
                'method': self.method_combo.currentText(),
                'volume_fraction': final_volume,
                'objective': final_objective,
                'iterations': n_iterations,
                'converged': True,
                'history': history,
                'displacement': u,
                'fe_solver': fe_solver
            }
            
            self.visualize_optimized_topology(fe_solver)
            
        else:
            self.results_label.setText(f"Optimization failed:\n{error_msg}")
            self.parent.message_text.append(f"Structural topology optimization failed: {error_msg}")

    def apply_topopt_constraints_to_params(self, to_params):
        """Apply topology optimization constraints"""
        constraints = self.parent.topopt_constraints
        
        other_constraints = constraints.get('other', {})
        if other_constraints.get('connected_topology', False):
            to_params.ENSURE_CONNECTED_TOPOLOGY = True
        
        if other_constraints.get('keep_fixed_faces', False):
            analysis_window = AnalysisWindow(self.parent)
            boundary_nodes, boundary_points, tolerance = analysis_window.get_boundary_mapping_data()
            constrained_elements = analysis_window.map_triangles_to_elements(
                list(self.parent.constrained_triangles), boundary_nodes, boundary_points, tolerance
            )
            to_params.ElemsToKeep = list(constrained_elements)

    def create_fe_solver_for_topopt(self):
        """Create FE solver for topology optimization"""
        import hex_structural_fea
        
        mesh = self.parent.hex_mesh
        analysis_window = AnalysisWindow(self.parent)
        analysis_window.prepare_mesh_for_analysis(mesh, "structural")
        
        boundary_nodes, boundary_points, tolerance = analysis_window.get_boundary_mapping_data()
        
        fixed_nodes = {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
        
        constrained_elements = analysis_window.map_triangles_to_elements(
            list(self.parent.constrained_triangles), boundary_nodes, boundary_points, tolerance
        )
        
        constrained_nodes = set()
        for elem_id in constrained_elements:
            constrained_nodes.update(mesh.elemArray[elem_id])
        
        for constraint in self.parent.constraint_data:
            constraint_type = constraint['type']
            if constraint_type == 'Fixed XYZ':
                fixed_nodes['xyz'].update(constrained_nodes)
            elif constraint_type == 'Fixed X':
                fixed_nodes['x'].update(constrained_nodes)
            elif constraint_type == 'Fixed Y':
                fixed_nodes['y'].update(constrained_nodes)
            elif constraint_type == 'Fixed Z':
                fixed_nodes['z'].update(constrained_nodes)
        
        load_nodes_groups = []
        load_forces = []
        
        for force_info in self.parent.force_data:
            force_elements = analysis_window.map_triangles_to_elements(
                force_info['triangles'], boundary_nodes, boundary_points, tolerance
            )
            
            force_nodes = set()
            for elem_id in force_elements:
                force_nodes.update(mesh.elemArray[elem_id])
            
            if force_nodes:
                load_nodes_groups.append(list(force_nodes))
                load_forces.append([force_info['force_x'], force_info['force_y'], force_info['force_z']])
        
        load_data = {
            'load_nodes_groups': load_nodes_groups,
            'load_forces': load_forces
        }
        
        mesh_processed, mat_prop, bc = analysis_window.process_data_for_solver(
            mesh, fixed_nodes, load_data, self.parent.applied_material['properties']
        )
        
        solver = analysis_window.get_solver()
        
        fe_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh_processed,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            rtol=1e-8
        )
        
        return fe_solver
    
    def check_prerequisites(self):
        """Check if all prerequisites for optimization are met"""
        if self.parent.hex_mesh is None:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return False
        
        if self.parent.applied_material is None:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return False
        
        has_forces = len(self.parent.force_data) > 0
        has_constraints = len(self.parent.constrained_triangles) > 0
        
        if not (has_forces and has_constraints):
            QtWidgets.QMessageBox.warning(
                self, 
                "Incomplete Loads", 
                "Please apply both forces and constraints for structural optimization."
            )
            return False
        
        if self.parent.topopt_constraints is None:
            QtWidgets.QMessageBox.warning(self, "No Constraints", "Please define topology optimization constraints first.")
            return False
        
        return True
    
    @QtCore.pyqtSlot(bool, str, object, object, object)
    def optimization_completed(self, success, error_msg, history, u, fe_solver):
        """Handle optimization completion"""
        self.optimization_running = False
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if success and history is not None:
            
            final_objective = history['objective'][-1]
            final_volume = history['volume'][-1]
            n_iterations = len(history['objective'])
            
            results_text = (
                f"Optimization completed successfully!\n\n"
                f"Final objective: {final_objective:.6e}\n"
                f"Final volume fraction: {final_volume:.3f}\n"
                f"Iterations: {n_iterations}\n"
                f"Target volume: {self.vol_spinbox.value():.3f}"
            )
            
            self.results_label.setText(results_text)
            
            self.parent.message_text.append("Structural topology optimization completed successfully")
            self.parent.message_text.append(f"Final objective: {final_objective:.6e}, Volume fraction: {final_volume:.3f}")
            
            self.parent.update_LivVar('topopt.structural_performed', True)
            self.parent.set_sidebar_icon("Structural TopOpt", "check")
            
            self.parent.topopt_results = {
                'method': self.method_combo.currentText(),
                'volume_fraction': final_volume,
                'objective': final_objective,
                'iterations': n_iterations,
                'converged': True,
                'history': history,
                'displacement': u,
                'fe_solver': fe_solver
            }
            
            self.visualize_optimized_topology(fe_solver)
            
        else:
            self.results_label.setText(f"Optimization failed:\n{error_msg}")
            self.parent.message_text.append(f"Structural topology optimization failed: {error_msg}")
    
    def visualize_optimized_topology(self, fe_solver):
        """Visualize the optimized topology using FE solver's plot_mesh method"""
        # Clear existing visualization except geometry info
        for name in list(self.parent.plotter.actors.keys()):
            if name != 'geometry_info':
                self.parent.plotter.remove_actor(name, reset_camera=False)
        
        # Use FE solver's plot_mesh method
        fe_solver.plot_mesh(plotter=self.parent.plotter)
        self.parent.plotter.render()
        
        self.parent.message_text.append("Optimized topology visualized")
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_running:
            self.optimization_running = False
            self.results_label.setText("Optimization stopped by user")
            self.parent.message_text.append("Structural topology optimization stopped by user")
            
            self.optimize_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        """Handle window close event"""
        if self.optimization_running:
            reply = QtWidgets.QMessageBox.question(
                self, 
                'Optimization Running', 
                'Optimization is still running. Do you want to stop it and close?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.stop_optimization()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
#----------------------------------------------------------------------------
class DisplayOptionsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Display Options")
        self.setFixedSize(300, 680)
        self.parent = parent
        
        # Initialize display state
        self.display_state = {
            'geometry': 'InitialDesign',
            'field': 'None',
            'display_on': 'Geometry',
            'x_cutting_percent': 0,
            'y_cutting_percent': 0,
            'z_cutting_percent': 0,
            'eigen_number': 1,
            'show_bounding_box': False,
            'show_triangles': False,
            'show_text': True,
            'scale_deformation': False,
            'show_transparent_geometry': False,
            'show_axis': True,
            'show_structural_loads': False,
            'show_thermal_loads': False,
            'show_topopt_constraints': False,
            'show_non_design_parts': False
        }
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Geometry dropdown
        geometry_layout = QtWidgets.QHBoxLayout()
        geometry_layout.addWidget(QtWidgets.QLabel("Geometry"))
        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItems(["InitialDesign"])
        geometry_layout.addWidget(self.geometry_combo)
        layout.addLayout(geometry_layout)
        
        # Field dropdown
        field_layout = QtWidgets.QHBoxLayout()
        field_layout.addWidget(QtWidgets.QLabel("Field"))
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.addItems(["None"])
        field_layout.addWidget(self.field_combo)
        layout.addLayout(field_layout)
        
        # Display On dropdown
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.addWidget(QtWidgets.QLabel("Display On"))
        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(["Geometry"])
        display_layout.addWidget(self.display_combo)
        layout.addLayout(display_layout)
        
        # Cutting percent controls
        self.x_cutting_spin = self.create_cutting_control(layout, "XCuttingPercent", 0)
        self.y_cutting_spin = self.create_cutting_control(layout, "YCuttingPercent", 0)
        self.z_cutting_spin = self.create_cutting_control(layout, "ZCuttingPercent", 0)
        
        # Eigen Number
        eigen_layout = QtWidgets.QHBoxLayout()
        eigen_layout.addWidget(QtWidgets.QLabel("EigenNumber"))
        self.eigen_spin = QtWidgets.QSpinBox()
        self.eigen_spin.setMinimum(1)
        self.eigen_spin.setMaximum(10)
        self.eigen_spin.setValue(1)
        eigen_layout.addWidget(self.eigen_spin)
        layout.addLayout(eigen_layout)
        
        # Checkboxes
        self.show_bounding_box_checkbox = QtWidgets.QCheckBox("Show bounding box")
        layout.addWidget(self.show_bounding_box_checkbox)
        
        self.show_triangles = QtWidgets.QCheckBox("Show triangles")
        #self.show_triangles.setChecked(True)
        layout.addWidget(self.show_triangles)
        
        self.show_text = QtWidgets.QCheckBox("Show text")
        self.show_text.setChecked(True)
        layout.addWidget(self.show_text)
        
        self.scale_deformation = QtWidgets.QCheckBox("Scale deformation")
        layout.addWidget(self.scale_deformation)
        
        self.show_transparent_geometry = QtWidgets.QCheckBox("Show transparent geometry")
        layout.addWidget(self.show_transparent_geometry)
        
        self.show_axis = QtWidgets.QCheckBox("Show axis")
        self.show_axis.setChecked(True)
        layout.addWidget(self.show_axis)
        
        self.show_structural_loads_checkbox = QtWidgets.QCheckBox("Show structural loads")
        layout.addWidget(self.show_structural_loads_checkbox)
        
        self.show_thermal_loads_checkbox = QtWidgets.QCheckBox("Show thermal loads")
        layout.addWidget(self.show_thermal_loads_checkbox)
        
        self.show_topopt_constraints_checkbox = QtWidgets.QCheckBox("Show TopOpt Constraints")
        layout.addWidget(self.show_topopt_constraints_checkbox)
    
        
        # Animation control
        animate_layout = QtWidgets.QHBoxLayout()
        animate_btn = QtWidgets.QPushButton("Animate for 3 cycles")
        animate_btn.clicked.connect(self.animate_view)
        animate_layout.addWidget(animate_btn)
        layout.addLayout(animate_layout)
        
        # Control buttons
        self.create_control_buttons(layout)
        
    def create_cutting_control(self, layout, label, default_value):
        """Create a cutting percentage control with spinbox"""
        cutting_layout = QtWidgets.QHBoxLayout()
        cutting_layout.addWidget(QtWidgets.QLabel(label))
        
        cutting_spin = QtWidgets.QSpinBox()
        cutting_spin.setMinimum(0)
        cutting_spin.setMaximum(100)
        cutting_spin.setValue(default_value)
        cutting_layout.addWidget(cutting_spin)
        
        layout.addLayout(cutting_layout)
        return cutting_spin
        
    def create_control_buttons(self, layout):
        """Create control buttons at the bottom"""
        button_layout = QtWidgets.QVBoxLayout()
        
        hide_selected_btn = QtWidgets.QPushButton("Hide Selected Part")
        hide_selected_btn.clicked.connect(self.hide_selected_part)
        button_layout.addWidget(hide_selected_btn)
        
        save_image_btn = QtWidgets.QPushButton("Save Image")
        save_image_btn.clicked.connect(self.save_image)
        button_layout.addWidget(save_image_btn)
        
        reset_view_btn = QtWidgets.QPushButton("Reset View")
        reset_view_btn.clicked.connect(self.reset_view)
        button_layout.addWidget(reset_view_btn)
        
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
    def connect_signals(self):
        """Connect all control signals to update methods"""
        self.geometry_combo.currentTextChanged.connect(self.update_display)
        self.field_combo.currentTextChanged.connect(self.update_display)
        self.display_combo.currentTextChanged.connect(self.update_display)
        
        self.x_cutting_spin.valueChanged.connect(self.update_cutting)
        self.y_cutting_spin.valueChanged.connect(self.update_cutting)
        self.z_cutting_spin.valueChanged.connect(self.update_cutting)
        
        # Connect all checkboxes
        for checkbox in [self.show_bounding_box_checkbox, self.show_triangles, self.show_text,
                 self.scale_deformation, self.show_transparent_geometry, self.show_axis,
                 self.show_structural_loads_checkbox, self.show_thermal_loads_checkbox,
                 self.show_topopt_constraints_checkbox]:
            checkbox.stateChanged.connect(self.update_display)
            
        # Set initial states based on current data
        self.update_checkbox_states()

    def update_checkbox_states(self):
        """Update checkbox states and enabled status based on current data"""
        # Structural loads checkbox
        has_structural = (
            (hasattr(self.parent, 'force_data') and bool(self.parent.force_data)) or
            (hasattr(self.parent, 'constrained_triangles') and bool(self.parent.constrained_triangles))
        )

        self.show_structural_loads_checkbox.setEnabled(bool(has_structural))
        if has_structural:
            self.show_structural_loads_checkbox.setChecked(True)
        else:
            self.show_structural_loads_checkbox.setChecked(False)
        
        # Thermal loads checkbox
        has_thermal = (
            hasattr(self.parent, 'thermal_loads_window') and
            self.parent.thermal_loads_window and
            getattr(self.parent.thermal_loads_window, 'thermal_loads', None) and
            any(self.parent.thermal_loads_window.thermal_loads.get(key, []) 
                for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])
        )
        has_thermal = bool(has_thermal)

        self.show_thermal_loads_checkbox.setEnabled(has_thermal)
        if has_thermal:
            self.show_thermal_loads_checkbox.setChecked(True)
        else:
            self.show_thermal_loads_checkbox.setChecked(False)
        
        # TopOpt constraints checkbox
        has_constraints = (
            hasattr(self.parent, 'topopt_constraint_actors') and
            bool(self.parent.topopt_constraint_actors)
        )

        self.show_topopt_constraints_checkbox.setEnabled(has_constraints)
        if has_constraints:
            self.show_topopt_constraints_checkbox.setChecked(True)
        else:
            self.show_topopt_constraints_checkbox.setChecked(False)

    def showEvent(self, event):
        """Update options when window is shown"""
        super().showEvent(event)
        self.update_geometry_options()
        self.update_field_options()
        self.update_checkbox_states()  # Update checkbox states when window opens
            
    def update_display(self):
        """Update display based on current settings"""
        geometry_choice = self.geometry_combo.currentText()
        field_choice = self.field_combo.currentText()
        # Only clear actors if something will be plotted
        should_plot = (
            (geometry_choice == "Initial Design" and self.parent.stl_geom) or
            (geometry_choice == "Mesh" and hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh) or
            (geometry_choice == "FEA Results" and hasattr(self.parent, "fe_solver") and self.parent.fe_solver) or
            (geometry_choice == "Final Design" and hasattr(self.parent, "topopt_results") and self.parent.topopt_results)
        )
        if should_plot:
            for name in list(self.parent.plotter.actors.keys()):
                if name != "geometry_info":
                    self.parent.plotter.remove_actor(name, reset_camera=False)

        # Visualization logic based on field selection
        if geometry_choice == "FEA Results":
            if field_choice == "Displacement" and hasattr(self.parent, "fe_solver") and self.parent.fe_solver:
                self.parent.fe_solver.plot_deformation(plotter=self.parent.plotter)
            elif field_choice == "Von Mises" and hasattr(self.parent, "fe_solver") and self.parent.fe_solver:
                self.parent.fe_solver.plot_vonMisesStress(plotter=self.parent.plotter)
            elif field_choice == "Temperature" and hasattr(self.parent, "thermal_fe_solver") and self.parent.thermal_fe_solver:
                self.parent.thermal_fe_solver.plot_temperature(plotter=self.parent.plotter)

        # Show initial design (STL) if selected
        if geometry_choice == "Initial Design" and self.parent.stl_geom:
            self.parent.stl_geom.plotGeometry(
                show_edges=self.show_triangles.isChecked(),
                show_axes=self.show_axis.isChecked(),
                show_bounding_box=self.show_bounding_box_checkbox.isChecked(),
                plotter=self.parent.plotter
            )
        # Show mesh if selected
        elif geometry_choice == "Mesh" and hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh:
            AnalysisWindow(self.parent).visualize_colored_mesh("structural")
            
        # Show FEA Results if selected
        elif geometry_choice == "FEA Results":
            # Prefer structural FEA if available, else thermal
            if hasattr(self.parent, "fe_solver") and self.parent.fe_solver is not None:
                self.parent.fe_solver.plot_deformation(plotter = self.parent.plotter)
            elif hasattr(self.parent, "thermal_fe_solver") and self.parent.thermal_fe_solver is not None:
                self.parent.thermal_fe_solver.plot_temperature(plotter = self.parent.plotter)

        # Show final design if selected
        elif geometry_choice == "Final Design" and hasattr(self.parent, "topopt_results") and self.parent.topopt_results:
            fe_solver = self.parent.topopt_results.get("fe_solver")
            if fe_solver:
                fe_solver.plot_mesh(plotter=self.parent.plotter)

        # Update bounding box display
        if self.show_bounding_box_checkbox.isChecked():
            self.show_bounding_box(True)
        else:
            self.show_bounding_box(False)
            
        # Update geometry display
        if self.show_triangles.isChecked():
            self.show_mesh_edges(True)
        else:
            self.show_mesh_edges(False)
            
        # Update text display
        if self.show_text.isChecked():
            self.show_geometry_text(True)
        else:
            self.show_geometry_text(False)
            
        # Update axis display
        if self.show_axis.isChecked():
            self.show_axes(True)
        else:
            self.show_axes(False)
            
        # Update transparency
        if self.show_transparent_geometry.isChecked():
            self.set_geometry_transparency(0.5)
        else:
            self.set_geometry_transparency(1.0)
            
        # Update structural loads visibility
        if self.show_structural_loads_checkbox.isChecked():
            self.show_structural_loads(True)
        else:
            self.show_structural_loads(False)
            
        # Update thermal loads visibility
        if self.show_thermal_loads_checkbox.isChecked():
            self.show_thermal_loads(True)
        else:
            self.show_thermal_loads(False)
            
        # Update TopOpt constraints visibility
        if self.show_topopt_constraints_checkbox.isChecked():
            self.show_topopt_constraints(True)
        else:
            self.show_topopt_constraints(False)

        # if self.show_non_design_parts_checkbox.isChecked():
        #     self.show_non_design_parts(True)
        # else:
        #     self.show_non_design_parts(False)
            
        self.parent.plotter.render()
        
    def update_cutting(self):
        """Update cutting planes based on spinbox values"""
        x_percent = self.x_cutting_spin.value() / 100.0
        y_percent = self.y_cutting_spin.value() / 100.0
        z_percent = self.z_cutting_spin.value() / 100.0
        
        if any([x_percent > 0, y_percent > 0, z_percent > 0]):
            self.apply_cutting_planes(x_percent, y_percent, z_percent)
        else:
            self.remove_cutting_planes()
            
    def show_mesh_edges(self, show):
        """Show mesh wireframe for mesh, hide triangle edges for STL."""
        geometry_choice = self.geometry_combo.currentText()
        for name, actor in self.parent.plotter.actors.items():
            if geometry_choice == "Mesh" and "mesh" in name.lower() and hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if hasattr(prop, 'EdgeVisibilityOn'):
                    prop.EdgeVisibilityOn()
                    prop.SetEdgeColor(0, 0, 0)
                    prop.SetLineWidth(1)
            elif geometry_choice == "Initial Design" and hasattr(actor, 'GetProperty'):
                prop = actor.GetProperty()
                if hasattr(prop, 'EdgeVisibilityOff'):
                    prop.EdgeVisibilityOff()

    def show_bounding_box(self, show):
        """Show or hide bounding box"""
        bbox_actor_name = "bounding_box_display"
        if show:
            if bbox_actor_name not in self.parent.plotter.actors:
                if self.parent.stl_geom:
                    bounds = self.parent.stl_geom.get_bounding_box()
                    if bounds:
                        bbox_mesh = pv.Box(bounds=bounds)
                        self.parent.plotter.add_mesh(
                            bbox_mesh,
                            style='wireframe',
                            color='black',
                            line_width=2,
                            name=bbox_actor_name
                        )
        else:
            if bbox_actor_name in self.parent.plotter.actors:
                self.parent.plotter.remove_actor(bbox_actor_name, reset_camera=False)
                    
    def show_geometry_text(self, show):
        """Show or hide geometry information text"""
        if show:
            if "geometry_info" not in self.parent.plotter.actors:
                self.add_geometry_info_text()
        else:
            if "geometry_info" in self.parent.plotter.actors:
                self.parent.plotter.remove_actor("geometry_info")
                
    def add_geometry_info_text(self):
        """Add geometry information text if geometry is loaded"""
        if not self.parent.stl_geom:
            return
            
        area, volume, _, _ = self.parent.stl_geom.compute_mass_properties()
        bounds = self.parent.stl_geom.get_bounding_box()
        length_unit = self.parent.settings.get_length_unit_string()
        
        info_lines = [
            f"Model: {os.path.basename(self.parent.stl_geom.file_path)}",
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
            font="arial"
        )
        
    def show_axes(self, show):
        """Show or hide coordinate axes"""
        if show:
            # Check if axes already exist
            axes_exist = False
            for name in self.parent.plotter.actors.keys():
                if 'axes' in name.lower():
                    axes_exist = True
                    break
            
            if not axes_exist:
                self.parent.plotter.add_axes(interactive=False)
        else:
            # Remove all axes-related actors
            axes_actors = []
            for name in list(self.parent.plotter.actors.keys()):
                if 'axes' in name.lower():
                    axes_actors.append(name)
            
            for axes_name in axes_actors:
                self.parent.plotter.remove_actor(axes_name, reset_camera=False)
                
    def set_geometry_transparency(self, opacity):
        """Set transparency of geometry"""
        for name, actor in self.parent.plotter.actors.items():
            # Skip text actors and other non-geometry actors
            if name == "geometry_info" or not hasattr(actor, 'GetProperty'):
                continue
                
            # Check if it's a 3D actor with opacity capabilities
            prop = actor.GetProperty()
            if hasattr(prop, 'SetOpacity'):
                prop.SetOpacity(opacity)
                
    def show_structural_loads(self, show):
        """Show or hide structural load actors"""
        if hasattr(self.parent, 'force_actors'):
            for actor in self.parent.force_actors:
                if show:
                    self.parent.plotter.add_actor(actor)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
                    
        if hasattr(self.parent, 'constraint_actors'):
            for actor in self.parent.constraint_actors:
                if show:
                    self.parent.plotter.add_actor(actor)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
                    
    def show_thermal_loads(self, show):
        """Show or hide thermal load actors"""
        if hasattr(self.parent, 'thermal_loads_window') and self.parent.thermal_loads_window:
            thermal_actor_lists = ['fixed_temp_actors', 'heat_source_actors', 'total_heat_actors']
            for actor_list_name in thermal_actor_lists:
                if hasattr(self.parent.thermal_loads_window, actor_list_name):
                    actor_list = getattr(self.parent.thermal_loads_window, actor_list_name)
                    for actor in actor_list:
                        if show:
                            self.parent.plotter.add_actor(actor)
                        else:
                            self.parent.plotter.remove_actor(actor, reset_camera=False)
                            
    def show_topopt_constraints(self, show):
        """Show or hide TopOpt constraint actors"""
        if hasattr(self.parent, 'topopt_constraint_actors'):
            for actor_key, actor in self.parent.topopt_constraint_actors.items():
                if actor:
                    if isinstance(actor, list):
                        for a in actor:
                            if show:
                                # Only add if not already present
                                if a not in self.parent.plotter.actors.values():
                                    self.parent.plotter.add_actor(a)
                            else:
                                self.parent.plotter.remove_actor(a, reset_camera=False)
                    else:
                        if show:
                            if actor not in self.parent.plotter.actors.values():
                                self.parent.plotter.add_actor(actor)
                        else:
                            self.parent.plotter.remove_actor(actor, reset_camera=False)
                            
    def apply_cutting_planes(self, x_percent, y_percent, z_percent):
        """Apply cutting planes based on percentages"""
        if not self.parent.stl_geom:
            return
            
        bounds = self.parent.stl_geom.get_bounding_box()
        if not bounds:
            return
            
        # Calculate cutting positions
        x_cut = bounds[0] + (bounds[1] - bounds[0]) * x_percent
        y_cut = bounds[2] + (bounds[3] - bounds[2]) * y_percent
        z_cut = bounds[4] + (bounds[5] - bounds[4]) * z_percent
        
        # Apply cuts to all visible mesh actors
        for name, actor in list(self.parent.plotter.actors.items()):
            if name not in ["geometry_info"] and hasattr(actor, 'GetMapper'):
                if x_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(-1, 0, 0), origin=(x_cut, 0, 0))
                if y_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(0, -1, 0), origin=(0, y_cut, 0))
                if z_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(0, 0, -1), origin=(0, 0, z_cut))
                    
    def remove_cutting_planes(self):
        """Remove all cutting planes"""
        # Reset the view by reloading the current visualization
        self.update_display()
        
    def animate_view(self):
        """Animate the view for 3 cycles"""
        self.parent.plotter.orbit_on_path(n_cycles=3, progress_bar=False)
        
    def hide_selected_part(self):
        """Hide currently selected triangles"""
        if hasattr(self.parent, 'highlight_actor') and self.parent.highlight_actor:
            self.parent.plotter.remove_actor(self.parent.highlight_actor, reset_camera=False)
            self.parent.highlight_actor = None
            self.parent.plotter.render()
            
    def save_image(self):
        """Save current view as image"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        if filename:
            self.parent.plotter.screenshot(filename)
            self.parent.message_text.append(f"Image saved: {os.path.basename(filename)}")
            
    def reset_view(self):
        """Reset the camera view"""
        self.parent.plotter.reset_camera()
        self.parent.plotter.render()
        
    def update_geometry_options(self):
        # """Update geometry dropdown options and select current state."""
        # items = ["Initial Design"]
        # current_index = 0

        # # Check for mesh
        # has_mesh = hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh is not None
        # if has_mesh:
        #     items.append("Mesh")
        #     current_index = 1

        # # Check for FEA results (structural or thermal)
        # has_structural_FEA = hasattr(self.parent, "fe_solver") and self.parent.fe_solver is not None
        # has_thermal_FEA = hasattr(self.parent, "thermal_fe_solver") and self.parent.thermal_fe_solver is not None
        # has_FEA = has_structural_FEA or has_thermal_FEA
        # if has_FEA:
        #     items.append("FEA Results")
        #     if current_index == 0:
        #         current_index = 1

        # # Check for topology optimization
        # has_topopt = hasattr(self.parent, "topopt_results") and self.parent.topopt_results is not None
        # if has_topopt:
        #     items.append("Final Design")
        #     current_index = 2

        # # Update dropdown items only if changed
        # self.geometry_combo.blockSignals(True)
        # self.geometry_combo.clear()
        # self.geometry_combo.addItems(items)
        # self.geometry_combo.setCurrentIndex(current_index)
        # self.geometry_combo.blockSignals(False)
        """Update geometry dropdown options and select current state intelligently."""
        items = ["Initial Design"]
        current_index = 0

        # Check for mesh
        has_mesh = hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh is not None
        if has_mesh:
            items.append("Mesh")
            current_index = 1

        # Check for FEA results (structural or thermal)
        has_structural_FEA = hasattr(self.parent, "fe_solver") and self.parent.fe_solver is not None
        has_thermal_FEA = hasattr(self.parent, "thermal_fe_solver") and self.parent.thermal_fe_solver is not None
        has_FEA = has_structural_FEA or has_thermal_FEA
        if has_FEA:
            items.append("FEA Results")
            current_index = len(items) - 1  # <-- Always select FEA Results if available

        # Check for topology optimization
        has_topopt = hasattr(self.parent, "topopt_results") and self.parent.topopt_results is not None
        if has_topopt:
            items.append("Final Design")
            current_index = len(items) - 1  # <-- Select Final Design if available

        # Update dropdown items only if changed
        self.geometry_combo.blockSignals(True)
        self.geometry_combo.clear()
        self.geometry_combo.addItems(items)
        self.geometry_combo.setCurrentIndex(current_index)
        self.geometry_combo.blockSignals(False)
        
                
    # def update_field_options(self):
    #     """Update field dropdown options based on available analysis data"""
    #     current_items = [self.field_combo.itemText(i) for i in range(self.field_combo.count())]
        
    #     # Add structural analysis fields
    #     if hasattr(self.parent, 'analysis_results'):
    #         fields_to_add = ["Displacement", "Stress", "Strain"]
    #         for field in fields_to_add:
    #             if field not in current_items:
    #                 self.field_combo.addItem(field)
                    
    #     # Add thermal analysis fields
    #     if hasattr(self.parent, 'thermal_results'):
    #         thermal_fields = ["Temperature", "HeatFlux"]
    #         for field in thermal_fields:
    #             if field not in current_items:
    #                 self.field_combo.addItem(field)

    def update_field_options(self):
        """Update field dropdown options based on available analysis data"""
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        # Always add these options
        self.field_combo.addItems(["Displacement", "Von Mises", "Temperature"])
        self.field_combo.blockSignals(False)
                    
    def showEvent(self, event):
        """Update options when window is shown"""
        super().showEvent(event)
        self.update_geometry_options()
        self.update_field_options()
#----------------------------------------------------------------------------
class ProjectsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Projects")
        self.setFixedSize(200, 120)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Buttons
        for text, method in [("Save Project", self.save_project), 
                           ("Load Project", self.load_project),
                           ("Close", self.close)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(method)
            layout.addWidget(btn)

    def save_project(self):
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "No Data", "No geometry loaded to save.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", "", "PyTO Files (*.pyto)")
        
        if not filename:
            return
            
        # Clean thermal loads data
        thermal_loads_data = {}
        if getattr(self.parent, 'thermal_loads_window', None):
            for load_type, load_list in self.parent.thermal_loads_window.thermal_loads.items():
                thermal_loads_data[load_type] = []
                for load_data in load_list:
                    clean_load = {'triangles': load_data.get('triangles', []),
                                'type': load_data.get('type', '')}
                    # Add thermal value based on type
                    for key in ['temperature', 'heat_flux', 'total_heat']:
                        if key in load_data:
                            clean_load[key] = load_data[key]
                    thermal_loads_data[load_type].append(clean_load)
        
        # Clean force data
        clean_force_data = [{
            key: force_info.get(key, 0 if 'force' in key else (0 if key == 'load_set' else ''))
            for key in ['triangles', 'force_x', 'force_y', 'force_z', 'load_set', 'type']
        } for force_info in self.parent.force_data]
        
        project_data = {
            'version': '2025.01',
            'stl_file_path': self.parent.stl_geom.file_path,
            'settings': {
                'unit_system': self.parent.settings.unit_system,
                'temperature_unit': self.parent.settings.temperature_unit,
                'angle_unit': self.parent.settings.angle_unit
            },
            'material_data': getattr(self.parent, 'applied_material', None),
            'force_data': clean_force_data,
            'constraint_data': self.parent.constraint_data, 
            'constrained_triangles': list(self.parent.constrained_triangles),
            'thermal_loads': thermal_loads_data,
            'topopt_constraints': getattr(self.parent, 'topopt_constraints', None)
        }
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            self.parent.message_text.append(f"Project saved: {os.path.basename(filename)}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving project: {str(e)}")
            self.parent.message_text.append(f"Save failed: {str(e)}")

    def load_project(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "PyTO Files (*.pyto)")
        
        if not filename:
            return
            
        import json
        try:
            with open(filename, 'r') as f:
                project_data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Error loading project: {str(e)}")
            return
        
        # Restore settings
        if 'settings' in project_data:
            s = project_data['settings']
            self.parent.settings.update_settings(
                s.get('unit_system', 'MKS'),
                s.get('temperature_unit', 'Kelvin'),
                s.get('angle_unit', 'Degree'))
        
        # Load geometry
        stl_path = project_data.get('stl_file_path')
        if stl_path and os.path.exists(stl_path):
            self.load_geometry(stl_path)
        else:
            self.parent.message_text.append(f"Warning: STL file not found: {stl_path}")
        
        # Restore material
        if project_data.get('material_data'):
            self.parent.applied_material = project_data['material_data']
            self.parent.update_LivVar('material_defined', True)
            self.parent.set_sidebar_icon("Material", "check")
            self.parent.set_sidebar_icon("Structural Loads", "arrow")
        
        # Restore structural loads
        self.parent.force_data = project_data.get('force_data', [])
        self.parent.constraint_data = project_data.get('constraint_data', [])
        self.parent.constrained_triangles = set(project_data.get('constrained_triangles', []))
        
        # Recreate structural visualizations
        self.restore_structural_loads()
        
        # Restore thermal loads
        if project_data.get('thermal_loads'):
            self.restore_thermal_loads(project_data['thermal_loads'])
        
        # Restore TopOpt constraints
        if project_data.get('topopt_constraints'):
            self.restore_topopt_constraints(project_data['topopt_constraints'])
        
        self.parent.message_text.append(f"Project loaded with all visualizations: {os.path.basename(filename)}")
        self.close()

    def load_geometry(self, file_path):
        """Load geometry using the same approach as GeometryWindow"""
        stl_geom = STLGeom(file_path)
        stl_geom.plotGeometry(show_edges=False, show_axes=False, show_bounding_box=False, plotter=self.parent.plotter)
        
        area, volume, _, _ = stl_geom.compute_mass_properties()
        bounds = stl_geom.get_bounding_box()
        length_unit = self.parent.settings.get_length_unit_string()

        info_lines = [
            f"Model: {os.path.basename(file_path)}",
            f"Volume: {volume:.2e} {length_unit}³",
            f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
            f"Surface Area: {area:.2e} {length_unit}²"
        ]

        self.parent.plotter.remove_actor("geometry_info")
        self.parent.plotter.add_text("\n".join(info_lines), position="upper_left", font_size=12, 
                                   color="black", name="geometry_info", font="arial")

        self.parent.stl_geom = stl_geom
        self.parent.plotter.disable_picking()
        self.parent.plotter.enable_point_picking(callback=self.parent.on_left_button_press, use_picker=True, 
                                                picker='cell', show_message=False, left_clicking=True, show_point=False)
        self.parent.plotter.iren.add_observer("RightButtonPressEvent", self.parent.on_right_button_press)
        
        self.parent.update_LivVar('geometry_loaded', True)
        self.parent.set_sidebar_icon("Geometry", "check")
        self.parent.set_sidebar_icon("Material", "arrow")
        self.parent.message_text.append(f"Geometry loaded from project: {os.path.basename(file_path)}")

    def restore_structural_loads(self):
        """Restore structural loads and visualizations"""
        # Create structural loads window if needed
        if not getattr(self.parent, 'structural_loads_window', None):
            self.parent.structural_loads_window = StructuralLoadsWindow(self.parent)
        
        # Recreate constraint visualizations
        if self.parent.constrained_triangles:
            self.parent.structural_loads_window.visualize_constraints()
        
        # Recreate force visualizations
        for force_info in self.parent.force_data:
            force_info['triangle_data'] = self.recreate_triangle_data(force_info['triangles'])
            config = self.parent.structural_loads_window.LOAD_TYPES["Force"]
            
            for axis in ['X', 'Y', 'Z']:
                force_value = force_info.get(f'force_{axis.lower()}', 0)
                if force_value != 0:
                    direction = config["directions"][axis]
                    self.parent.structural_loads_window.visualize_force_arrows(
                        force_info['triangle_data'], force_value, direction, config["color"], axis)
        
        # Update structural loads state
        if self.parent.force_data:
            self.parent.update_LivVar('structural_loads.forces_applied', True)
        if self.parent.constrained_triangles:
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        
        if (self.parent.LivVar['structural_loads']['forces_applied'] and 
            self.parent.LivVar['structural_loads']['fixed_constraints']):
            self.parent.LivVar['structural_loads']['applied'] = True
            self.parent.set_sidebar_icon("Structural Loads", "check")
            self.parent.set_sidebar_icon("Analysis", "arrow")
            self.parent.set_sidebar_icon("TopOpt Constraints", "arrow")

    def restore_thermal_loads(self, thermal_loads):
        """Restore thermal loads and visualizations"""
        # Create thermal loads window if needed
        if not getattr(self.parent, 'thermal_loads_window', None):
            self.parent.thermal_loads_window = ThermalLoadsWindow(self.parent)
        
        self.parent.thermal_loads_window.thermal_loads = thermal_loads
        
        # Recreate visualizations for each thermal load type
        thermal_types = {
            'fixed_temps': ('Temperature', 'temperature'),
            'heat_sources': ('Heat Flux', 'heat_flux'),
            'total_heat_sources': ('Total Heat', 'total_heat')
        }
        
        for load_key, (type_name, value_key) in thermal_types.items():
            for load in thermal_loads.get(load_key, []):
                config = self.parent.thermal_loads_window.THERMAL_TYPES[type_name]
                load['triangle_data'] = self.recreate_triangle_data(load['triangles'])
                self.parent.thermal_loads_window.create_thermal_visualization(
                    load['triangle_data'], load[value_key], config)
        
        # Update thermal state
        if any(thermal_loads.get(key, []) for key in thermal_types.keys()):
            self.parent.update_LivVar('thermal_loads.applied', True)
            self.parent.set_sidebar_icon("Thermal Loads", "check")
            self.parent.set_sidebar_icon("Analysis", "arrow")
            self.parent.set_sidebar_icon("TopOpt Constraints", "arrow")

    def restore_topopt_constraints(self, constraints):
        """Restore TopOpt constraints and visualizations"""
        self.parent.topopt_constraints = constraints
        
        # Create TopOpt constraints window if needed
        if not getattr(self.parent, 'topopt_constraints_window', None):
            self.parent.topopt_constraints_window = TopOptConstraintsWindow(self.parent)
        
        # Update widgets to match saved constraints
        for category, constraint_data in constraints.items():
            for constraint_key, constraint_value in constraint_data.items():
                if isinstance(constraint_value, bool):
                    widget_name = f'{constraint_key}_check'
                    if widget_name in self.parent.topopt_constraints_window.widgets:
                        self.parent.topopt_constraints_window.widgets[widget_name].setChecked(constraint_value)
                elif isinstance(constraint_value, dict):
                    check_widget = f'{constraint_key}_check'
                    if check_widget in self.parent.topopt_constraints_window.widgets:
                        self.parent.topopt_constraints_window.widgets[check_widget].setChecked(
                            constraint_value.get('enabled', False))
                    
                    if 'value' in constraint_value:
                        for widget_type in ['combo', 'spin']:
                            value_widget = f'{constraint_key}_{widget_type}'
                            if value_widget in self.parent.topopt_constraints_window.widgets:
                                widget = self.parent.topopt_constraints_window.widgets[value_widget]
                                if widget_type == 'combo':
                                    widget.setCurrentText(str(constraint_value['value']))
                                else:
                                    widget.setValue(constraint_value['value'])
                                break
        
        self.parent.topopt_constraints_window.update_visualizations()
        self.parent.update_LivVar('topopt.constraints_defined', True)
        self.parent.set_sidebar_icon("TopOpt Constraints", "check")
        self.parent.set_sidebar_icon("Structural TopOpt", "arrow")
        self.parent.set_sidebar_icon("Thermal TopOpt", "arrow")

    def recreate_triangle_data(self, triangle_indices):
        """Recreate triangle data structure from saved indices"""
        triangle_data = []
        for tri_idx in triangle_indices:
            triangle_vertices = self.parent.stl_geom.mesh.vectors[tri_idx]
            center = np.mean(triangle_vertices, axis=0)
            v1, v2 = triangle_vertices[1] - triangle_vertices[0], triangle_vertices[2] - triangle_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            triangle_data.append({
                'index': tri_idx,
                'center': center,
                'normal': normal,
                'vertices': triangle_vertices
            })
        return triangle_data
#----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())