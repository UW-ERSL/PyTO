import sys
sys.path.append('./src') #assuming the pyTO src files is in the parent directory
import vtk
import math
import numpy as np
import os
import json
import time
import threading
import bound_cond
import mat_lib
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import QMetaObject, Q_ARG, Qt, QObject
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt
from queue import Queue
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt
from stl_reader import STLGeom
from hex_mesher import Mesher
import hex_structural_fea as fea
import linear_solvers as lin_solv
import jax
import hex_thermal_fea
import traceback
from topopt_density_mma import topopt_mma
from topopt_density_oc import topopt_optimality_criteria
from topopt_pareto import topopt_pareto
from topopt_levelset import topopt_levelset
from topopt_common import TOParams
from hex_structural_fea import StructFEA
from hex_structural_examples import *    
import linear_solvers as lin_solv
import jax
   
# Enable JAX double precision
jax.config.update("jax_enable_x64", True)
            

'''

To do:
1. Topopt fails
2. 

'''
#---------------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.settings = Settings()
        self.stl_geom = None
        self.stl_filepath = None    
        self.constraint_actors = []
        self.force_actors = []
        self.material_data = None
        self.structuralBC = None   
        self.analysis = Analysis()
        self.topopt_constraints = None
        self.optimization_params = None
        self.thermal_optimization_params = None
        self.results_actor = None
        self.scalar_bar = None
        self.analysis_results = None
        self.fixed_temp_actors = []
        self.heat_flux_actors = []
        self.heat_source_actors = []
        self.total_heat_flux_actors = []
        self.convection_actors = []
        self.radiation_actors = []
        self.internal_heat_actors = []
        self.initialize_display_settings()

        # LivVar - Live Variable to track UI state
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
        
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        # Set size policy for the main widget
        self.main_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        
        # Initialize thermal loads
        self.thermal_loads = {
            "fixed_temps": [],      # List of (node_ids, temperature)
            "heat_sources": [],     # List of (node_ids, heat_value)
            "convection": [],       # List of (node_ids, h_coeff, ambient_temp)
            "radiation": [],        # List of (node_ids, emissivity, ambient_temp)
            "internal_heat": []     # List of (node_ids, heat_generation)
        }

    

        # Modify h_layout to use proper stretch factors
        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.setSpacing(10)
        self.h_layout.setContentsMargins(10, 10, 10, 10)
        
        # VTK Setup with size policy
        self.vtk_frame = QtWidgets.QFrame()
        self.vtk_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.vtk_layout = QtWidgets.QVBoxLayout(self.vtk_frame)
        self.vtk_layout.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_layout.addWidget(self.vtkWidget)
        self.h_layout.addWidget(self.vtk_frame, stretch=4)
        
        # Sidebar with fixed width and scrollable content
        self.setup_sidebar()
        self.sidebar.setFixedWidth(250)  # Fixed width for consistency
        self.h_layout.addWidget(self.sidebar, stretch=0)
        
        # Message Frame with proportional height
        self.setup_message_frame()
        
        # Add layouts to main layout with proper stretch
        self.main_layout.addLayout(self.h_layout, stretch=4)
        self.main_layout.addWidget(self.message_frame, stretch=1)
        
        # Status Bar
        self.setup_status_bar()
        
        self.setup_vtk()

    def initialize_display_settings(self):
        """Initialize display settings with default values"""
        self.display_settings = {
            'geometry': 'InitialDesign',
            'field': 'None',
            'x_cutting': 0,
            'y_cutting': 0,
            'z_cutting': 0,
            'eigen_number': 1,
            'show_bounding_box': False,
            'show_mesh': True,
            'show_results': True,
            'show_geometry': True,
            'show_triangles': False,
            'show_text': True,
            'scale_deformation': True,
            'show_transparent': False,
            'show_axis': True,
            'show_structural_loads': True,
            'show_thermal_loads': True,
            'show_topopt_constraints': True,
            'show_non_design': True
        }
        self.cutting_plane_actors = []
        self.hidden_indices = set()


    def open_display_options(self):
        """Open the enhanced display options dialog"""
        dialog = DisplayOptionsWindow(self)
        dialog.show()

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
        
        # Update current workflow step based on the change
        if key == 'geometry_loaded' and value:
            self.LivVar['current_step'] = 'geometry_loaded'
            # Update availability of next steps
            self.update_button_icon("Material", "arrow")
            self.update_button_icon("Structural Loads", "arrow")
            self.update_button_icon("Thermal Loads", "arrow")
        elif key == 'material_defined' and value:
            self.LivVar['current_step'] = 'material_defined'
            # Material is complete, loads can be defined
            self.update_button_icon("Material", "check")
        # Change this block to enable TopOpt Constraints after loads are applied
        elif key == 'structural_loads.applied' and value:
            self.LivVar['current_step'] = 'loads_applied'
            self.update_button_icon("Structural Loads", "check")
            self.update_button_icon("Analysis", "arrow")
            # Enable TopOpt Constraints immediately after structural loads are applied
            self.update_button_icon("TopOpt Constraints", "arrow")
            self.message_text.append("Structural loads applied. You can now define TopOpt constraints.")
        # Change this block to enable TopOpt Constraints after loads are applied
        elif key == 'thermal_loads.applied' and value:
            self.LivVar['current_step'] = 'loads_applied'
            self.update_button_icon("Thermal Loads", "check")
            self.update_button_icon("Analysis", "arrow")
            # Enable TopOpt Constraints immediately after thermal loads are applied
            self.update_button_icon("TopOpt Constraints", "arrow")
            self.message_text.append("Thermal loads applied. You can now define TopOpt constraints.")
        elif key == 'mesh_generated' and value:
            self.LivVar['current_step'] = 'mesh_generated'
        elif key == 'analysis.performed' and value:
            self.LivVar['current_step'] = 'analysis_performed'
            self.update_button_icon("Analysis", "check")
        elif key == 'topopt.constraints_defined' and value:
            self.LivVar['current_step'] = 'topopt_ready'
            self.update_button_icon("TopOpt Constraints", "check")
            self.update_button_icon("Structural TopOpt", "arrow")
            self.update_button_icon("Thermal TopOpt", "arrow")
        elif key == 'topopt.structural_performed' and value:
            self.update_button_icon("Structural TopOpt", "check")
        elif key == 'topopt.thermal_performed' and value:
            self.update_button_icon("Thermal TopOpt", "check")
        
        # Map state keys to button names for UI updates
        button_map = {
            'geometry_loaded': "Geometry",
            'material_defined': "Material",
            'structural_loads.applied': "Structural Loads",
            'thermal_loads.applied': "Thermal Loads",
            'mesh_generated': "Analysis",
            'analysis.performed': "Analysis",
            'topopt.constraints_defined': "TopOpt Constraints",
            'topopt.structural_performed': "Structural TopOpt",
            'topopt.thermal_performed': "Thermal TopOpt"
        }
        
        # Update button icon if there's a corresponding button
        if key in button_map:
            button_name = button_map[key]
            icon_type = "check" if value else "cross"
            self.update_button_icon(button_name, icon_type)
            
        # Log state change to message box
        #self.message_text.append(f"Status update: {key.replace('_', ' ').replace('.', ' - ')} = {value}")
        
        # Print current workflow step
        #self.message_text.append(f"{self.LivVar['current_step']}")

    
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
            return False, "You need to load geometry first"
            
        elif target_step == 'loads' and not self.LivVar['material_defined']:
            return False, "You need to define material properties first"
            
        elif target_step == 'analysis':
            # Check if either structural OR thermal loads have been applied
            loads_applied = (self.LivVar['structural_loads']['applied'] or 
                            self.LivVar['thermal_loads']['applied'])
            if not loads_applied:
                return False, "You need to apply either structural or thermal loads first"
        
        # Change this to allow TopOpt Constraints after loads are applied
        elif target_step == 'topopt_constraints':
            # Check if either structural OR thermal loads have been applied
            loads_applied = (self.LivVar['structural_loads']['applied'] or 
                            self.LivVar['thermal_loads']['applied'])
            if not loads_applied:
                return False, "You need to apply either structural or thermal loads first"
            
        elif target_step == 'topopt' and not self.LivVar['topopt']['constraints_defined']:
            return False, "You need to define topology optimization constraints first"
            
        return True, "Ready"
    
    
    
    def setup_sidebar(self):
        # Create a scroll area for the sidebar
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        sidebar_content = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(5)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Store buttons for dynamic updates
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
            ("TopOpt Constraints", "cross"),  # Not available until analysis is done
            ("Structural TopOpt", "cross"),  # Not available until constraints defined
            ("Thermal TopOpt", "cross"),  # Not available until constraints defined
            ("TopOpt Results", "cross"),  # Not available until optimization is done
            ("Projects", "arrow"),  # Always available
            ("Help", "arrow")  # Always available
        ]

        # Add buttons to the sidebar
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
            btn.setEnabled(True)  # Enable or disable buttons dynamically if needed
            btn.clicked.connect(lambda checked, name=text: self.sidebar_button_clicked(name))
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons[text] = btn  # Store button reference for updates
        
        sidebar_layout.addStretch()
        scroll.setWidget(sidebar_content)
        
        # Create sidebar container
        self.sidebar = QtWidgets.QFrame()
        sidebar_main_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_main_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_main_layout.addWidget(scroll)

    def sidebar_button_clicked(self, name):
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
            self.open_structural_loads()
        elif name == "Thermal Loads":
            self.open_thermal_loads_window()
        elif name == "Display Options":
            self.open_display_options()
        elif name == "Analysis":
            self.open_analysis_window()
        elif name == "TopOpt Constraints":
            self.open_topopt_constraints_window()
        elif name == "Structural TopOpt":
            self.open_structural_topopt_window()
        elif name == "Thermal TopOpt":
            self.open_thermal_topopt_window()
        elif name == "Projects":
            dialog = ProjectsWindow(self)
            dialog.exec_()
        
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
        elif icon_type == "cross":
            icon_file = os.path.join(base_path, "cross.png")
        elif icon_type == "check":
            icon_file = os.path.join(base_path, "check.png")
        if not os.path.exists(icon_file):
            print(f"Icon file not found: {icon_file}")
        return QIcon(icon_file)

    def update_button_icon(self, button_name, icon_type):
        """
        Dynamically update the icon of a specific sidebar button.
        
        Parameters:
        -----------
        button_name : str
            Name of the button to update
        icon_type : str
            Type of icon to apply: "arrow" (blue), "check" (green), or "cross" (red)
        """
        if button_name in self.sidebar_buttons:
            button = self.sidebar_buttons[button_name]
            button.setIcon(self.get_icon(icon_type))

    def setup_message_frame(self):
        self.message_frame = QtWidgets.QFrame()
        self.message_frame.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        message_layout = QtWidgets.QVBoxLayout(self.message_frame)
        
        self.message_text = QtWidgets.QTextEdit()
        # Updated fixed height to match standard size
        self.message_text.setFixedHeight(162)
        self.message_text.setStyleSheet("""
            QTextEdit {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
        """)
        self.message_text.setReadOnly(True)
        self.message_text.setText("Welcome to pyTO!")
        
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
        
        version_label = QtWidgets.QLabel("pyTO GUI Version 2025.01. ")
        build_label = QtWidgets.QLabel("GUI Build Date 3.15.2025. ")
        license_label = QtWidgets.QLabel(
            "This is an academic license, and should not be used for commercial purposes."
        )
        license_label.setStyleSheet("color: red;")
        
        status_bar.addWidget(version_label)
        status_bar.addWidget(build_label)
        status_bar.addWidget(license_label)

    def on_geometry_loaded(self):
        # Change the icon for the "Geometry" button to a green checkmark
        self.update_button_icon("Geometry", "check")
        
        # Update icons for next steps to show they're available
        self.update_button_icon("Material", "arrow")
        self.update_button_icon("Structural Loads", "arrow")
        self.update_button_icon("Thermal Loads", "arrow")

    def setup_vtk(self):
        """Modified setup_vtk to handle multiple rendering layers"""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetLayer(0)  # Main geometry renderer on base layer
        self.renderer.SetBackground(1, 1, 1)
        
        render_window = self.vtkWidget.GetRenderWindow()
        render_window.SetNumberOfLayers(2)  # Enable layered rendering
        render_window.AddRenderer(self.renderer)
        
        self.interactor = render_window.GetInteractor()
        
        # Setup picker and interaction style
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.AddObserver("InteractionEvent", self.on_interaction)
        self.interactor.SetInteractorStyle(style)
        
        # Add coordinate axes with increased size (50% larger)
        axes = vtk.vtkAxesActor()
        
        # Set standard shaft and tip lengths first
        axes.SetShaftTypeToCylinder()
        
        # Scale the shaft and cone radii by 50%
        axes.SetCylinderRadius(axes.GetCylinderRadius() * 1.5)
        axes.SetConeRadius(axes.GetConeRadius() * 1.5)
        
        # Keep the standard normalized lengths to maintain correct positioning
        # Don't modify these as it can cause positioning issues
        axes.SetNormalizedShaftLength(0.8, 0.8, 0.8)
        axes.SetNormalizedTipLength(0.2, 0.2, 0.2)
        
        # Increase font size by 50%
        for i, axis in enumerate(['X', 'Y', 'Z']):
            caption_property = axes.GetXAxisCaptionActor2D().GetCaptionTextProperty() if i == 0 else \
                            axes.GetYAxisCaptionActor2D().GetCaptionTextProperty() if i == 1 else \
                            axes.GetZAxisCaptionActor2D().GetCaptionTextProperty()
            caption_property.SetFontSize(int(caption_property.GetFontSize() * 1.5))
            caption_property.SetFontFamilyToArial()
            caption_property.SetBold(True)
        
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.25, 0.25)  # Increase viewport size by ~25%
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOn()
        
        # Other observers
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.interactor.AddObserver("RightButtonPressEvent", self.on_right_button_press)
        
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.interactor.Initialize()

    def on_interaction(self, obj, event):
        if hasattr(self, 'highlight_poly_data'): 
            self.update_highlights()
    
    def open_material_window(self):
            ready, message = self.check_workflow_readiness('material')
            if ready:
                dialog = MaterialWindow(self)
                dialog.exec_()
            else:
                QtWidgets.QMessageBox.warning(self, "Workflow Error", message)
    
    def open_structural_loads(self):
        ready, message = self.check_workflow_readiness('loads')
        if ready:
            dialog = StructuralLoadsWindow(self)
            dialog.show()  # Use show() instead of exec_() to allow main window interaction
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def open_thermal_loads_window(self):
        """Open the thermal loads window"""
        ready, message = self.check_workflow_readiness('loads')
        if ready:
            dialog = ThermalLoadsWindow(self)
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def open_thermal_analysis_window(self):
        """Open the analysis window and focus on thermal analysis"""
        ready, message = self.check_workflow_readiness('analysis')
        if ready:
            dialog = AnalysisWindow(self)
            # Focus on thermal analysis by making the thermal button visible/highlighted
            if hasattr(dialog, 'thermal_button'):
                dialog.thermal_button.setFocus()
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

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
        
        # Update the sidebar icon to "cross" for Structural Loads
        self.update_button_icon("Structural Loads", "cross")

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
            self.update_highlights()  
            
            self.interactor.GetInteractorStyle().OnLeftButtonDown()

    # def on_right_button_press(self, obj, event):
    #     click_pos = self.interactor.GetEventPosition()
    #     self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
    #     cell_id = self.picker.GetCellId()
    #     if cell_id >= 0: 
    #         self.clear_selections()
        
        
    #     self.interactor.GetInteractorStyle().OnRightButtonDown()

    def on_right_button_press(self, obj, event):
        """Handle right button press for deselecting faces"""
        try:
            click_pos = self.interactor.GetEventPosition()
            self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            
            cell_id = self.picker.GetCellId()
            if cell_id >= 0 and self.stl_geom:
                # Store current state of all highlights
                current_highlights = self.stl_geom.tri_highlight.copy()
                
                # First reset all highlights
                self.stl_geom.tri_highlight = [False] * len(self.stl_geom.tri_highlight)
                
                # Then highlight only the clicked triangle and its connected faces
                _, _ = self.stl_geom.highlight_triangles_recursive(
                    cell_id,
                    depth=500,
                    cutoff_angle_degrees=30
                )
                
                # Get the faces that would be highlighted from this click
                faces_to_deselect = [i for i, highlight in enumerate(self.stl_geom.tri_highlight) if highlight]
                
                # Restore original highlights
                self.stl_geom.tri_highlight = current_highlights
                
                # Deselect only the faces that were in the clicked group
                for face_id in faces_to_deselect:
                    self.stl_geom.tri_highlight[face_id] = False
                
                # Update the visualization
                self.update_highlights()
        
        except Exception as e:
            print(f"Error in deselection: {str(e)}")
        
        self.interactor.GetInteractorStyle().OnRightButtonDown()

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self, self)
        dialog.exec_()

    def load_stl_file(self, file_path):
        """Load an STL file by creating and using a GeometryWindow"""
        try:
            # Create a geometry window
            geometry_window = GeometryWindow(self, self)
            
            # Set the file path before loading
            self.stl_filepath = file_path
            
            # Call the load_stl_file method
            geometry_window.load_stl_file(file_path)
            
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load STL file: {str(e)}")
            return False

    def open_analysis_window(self):
        ready, message = self.check_workflow_readiness('analysis')
        if ready:
            dialog = AnalysisWindow(self)
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def open_topopt_constraints_window(self):
        """Open the topology optimization constraints window"""
        ready, message = self.check_workflow_readiness('topopt_constraints')
        if ready:
            # Create the dialog
            dialog = TopOptConstraintsWindow(self)
            
            # If we have previously saved constraints, restore the UI state
            if hasattr(self, 'topopt_constraints') and self.topopt_constraints:
                self.update_constraints_window_from_dict(dialog, self.topopt_constraints)
            
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def update_constraints_window_from_dict(self, window, constraints_dict):
        """
        Update the constraints window UI based on loaded constraints dictionary.
        
        Parameters:
        - window: The TopOptConstraintsWindow instance to update
        - constraints_dict: Dictionary containing constraint settings
        """
        # Manufacturing constraints
        if 'manufacturing' in constraints_dict:
            mfg = constraints_dict['manufacturing']
            
            # Extrude
            if 'extrude' in mfg:
                window.extrude_check.setChecked(mfg['extrude']['enabled'])
                window.extrude_combo.setCurrentText(mfg['extrude']['direction'])
            
            # AM Build
            if 'am_build' in mfg:
                window.am_build_check.setChecked(mfg['am_build']['enabled'])
                window.am_build_combo.setCurrentText(mfg['am_build']['direction'])
            
            # Draw Direction
            if 'draw_direction' in mfg:
                window.draw_direction_check.setChecked(mfg['draw_direction']['enabled'])
                window.draw_direction_combo.setCurrentText(mfg['draw_direction']['direction'])
            
            # Cyclic Symmetry
            if 'cyclic_symmetry' in mfg:
                window.cyclic_sym_check.setChecked(mfg['cyclic_symmetry']['enabled'])
                window.cyclic_sym_combo.setCurrentText(mfg['cyclic_symmetry']['angle'])
        
        # Patterns
        if 'patterns' in constraints_dict:
            patterns = constraints_dict['patterns']
            
            # X Grid
            if 'x_grid' in patterns:
                window.x_grid_check.setChecked(patterns['x_grid']['enabled'])
                window.x_grid_spin.setValue(patterns['x_grid']['value'])
            
            # Y Grid
            if 'y_grid' in patterns:
                window.y_grid_check.setChecked(patterns['y_grid']['enabled'])
                window.y_grid_spin.setValue(patterns['y_grid']['value'])
            
            # Z Grid
            if 'z_grid' in patterns:
                window.z_grid_check.setChecked(patterns['z_grid']['enabled'])
                window.z_grid_spin.setValue(patterns['z_grid']['value'])
        
        # Performance constraints
        if 'performance' in constraints_dict:
            perf = constraints_dict['performance']
            
            # Stress Safety
            if 'stress_safety' in perf:
                window.stress_safety_check.setChecked(perf['stress_safety']['enabled'])
                window.stress_safety_spin.setValue(perf['stress_safety']['value'])
            
            # Max Displacement
            if 'max_displacement' in perf:
                window.max_disp_check.setChecked(perf['max_displacement']['enabled'])
                window.max_disp_spin.setValue(perf['max_displacement']['value'])
            
            # Min Frequency
            if 'min_frequency' in perf:
                window.min_freq_check.setChecked(perf['min_frequency']['enabled'])
                window.min_freq_spin.setValue(perf['min_frequency']['value'])
            
            # Max Temperature
            if 'max_temperature' in perf:
                window.max_temp_check.setChecked(perf['max_temperature']['enabled'])
                window.max_temp_spin.setValue(perf['max_temperature']['value'])
            
            # Min Feature
            if 'min_feature' in perf:
                window.min_feat_check.setChecked(perf['min_feature']['enabled'])
                window.min_feat_spin.setValue(perf['min_feature']['value'])
            
            # Max Feature
            if 'max_feature' in perf:
                window.max_feat_check.setChecked(perf['max_feature']['enabled'])
                window.max_feat_spin.setValue(perf['max_feature']['value'])
        
        # Symmetry constraints
        if 'symmetry' in constraints_dict:
            sym = constraints_dict['symmetry']
            
            # X Symmetry
            if 'x_symmetry' in sym:
                window.x_symmetry_check.setChecked(sym['x_symmetry'])
            
            # Y Symmetry
            if 'y_symmetry' in sym:
                window.y_symmetry_check.setChecked(sym['y_symmetry'])
            
            # Z Symmetry
            if 'z_symmetry' in sym:
                window.z_symmetry_check.setChecked(sym['z_symmetry'])
        
        # Other constraints
        if 'other' in constraints_dict:
            other = constraints_dict['other']
            
            # Connected Topology
            if 'connected_topology' in other:
                window.connected_topology_check.setChecked(other['connected_topology'])
            
            # Keep Fixed Faces
            if 'keep_fixed_faces' in other:
                window.keep_fixed_faces_check.setChecked(other['keep_fixed_faces'])
        
        # Update visualizations if needed
        window.update_extrude_visualization()
        window.update_draw_direction_visualization()
        window.update_cyclic_symmetry_visualization()
        window.update_grid_pattern_visualization()
        window.update_symmetry_visualization()

    def initialize_constraint_actors(self):
        """
        Initialize the dictionary for storing topological optimization constraint visualization actors.
        Should be called in the parent window class.
        """
        # Create a dictionary to store all constraint visualization actors
        self.topopt_constraint_actors = {
            'extrude': None,
            'am_build': None,
            'draw_direction': None,
            'cyclic_symmetry': None,
            'bounding_box': None,
            'grid_patterns': {
                'x_grid': [],
                'y_grid': [],
                'z_grid': []
            },
            'symmetry': {
                'x_symmetry': None,
                'y_symmetry': None,
                'z_symmetry': None
            }
        }


    def open_optimize_topology_window(self):
        dialog = OptimizeTopologyWindow(self)
        dialog.show()

    def update_optimization_results(self, dialog, u, history, elapsed_time):
        """Update UI with topology optimization results"""
        self.message_text.append(f"Optimization completed in {elapsed_time:.2f} seconds")
        self.message_text.append(f"Final compliance: {history['compliance'][-1]:.6g}")
        self.message_text.append(f"Final volume fraction: {history['volume'][-1]:.3f}")
        
        # Store results
        self.topopt_results = {
            'displacement': u,
            'history': history,
            'elapsed_time': elapsed_time
        }
        
        # Visualize the optimized topology
        self.visualize_optimized_topology(u)
        
        # Stop the optimization process in the dialog
        dialog.stop_optimization()

    def visualize_optimized_topology(self, u=None):
        """Visualize the optimized topology"""
        if not hasattr(self, 'stl_geom') or self.stl_geom is None or not hasattr(self.stl_geom, 'mesh'):
            self.message_text.append("Error: No valid mesh to visualize")
            return
            
        # Import plots module
        import plots
        
        # Clear previous visualization
        self.renderer.RemoveAllViewProps()
        
        # Generate title
        title = f"Optimized topology (Vol.Frac: {self.optimization_params['volume_fraction']})"
        
        # Get density values from the mesh
        mesh = self.stl_geom.mesh
        
        # Add mesh visualization to renderer
        actor = plots.get_mesh_actor(mesh, u, show_edges=True)
        self.renderer.AddActor(actor)
        
        # Add density isosurface
        iso_actor = plots.get_isocontour_actor(mesh, binarize=self.optimization_params['smooth_surface'])
        if iso_actor:
            self.renderer.AddActor(iso_actor)
        
        # Reset camera and render
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def open_structural_topopt_window(self):
        ready, message = self.check_workflow_readiness('topopt')
        if ready:
            dialog = OptimizeTopologyWindow(self)
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def update_optimization_results(self, optimizer_window, u, history, elapsed_time):
        """Update UI with topology optimization results."""
        # Update progress bar to 100%
        if hasattr(optimizer_window, 'progress_bar'):
            optimizer_window.progress_bar.setValue(100)
        
        # Enable the optimizer window buttons again
        optimizer_window.optimize_button.setEnabled(True)
        optimizer_window.stop_button.setEnabled(False)
        optimizer_window.export_button.setEnabled(True)
        
        # Update optimization state
        optimizer_window.optimization_running = False
        
        # Display results in the 3D view
        if hasattr(self, 'stl_geom') and self.stl_geom and hasattr(self.stl_geom, 'mesh'):
            # Update densities in the mesh
            self.stl_geom.mesh.densities = u
            
            # Display the optimized result
            optimizer_window.visualization.display_results(
                fe_solver=self.stl_geom.mesh, 
                history=history, 
                elapsed_time=elapsed_time
            )
        
        # Log completion
        self.message_text.append(f"Topology optimization completed in {elapsed_time:.2f} seconds")
    
    def open_thermal_topopt_window(self):
        ready, message = self.check_workflow_readiness('topopt')
        if ready:
            dialog = ThermalTopOptWindow(self)
            dialog.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def reset_visualization(self):
        """Reset visualization state"""
        if hasattr(self, 'results_actor') and self.results_actor:
            self.renderer.RemoveActor(self.results_actor)
            self.results_actor = None
        
        if hasattr(self, 'scalar_bar') and self.scalar_bar:
            self.renderer.RemoveActor(self.scalar_bar)
            self.scalar_bar = None
            
        if hasattr(self, 'mesh_actor'):
            self.mesh_actor.SetVisibility(True)
            
        self.vtkWidget.GetRenderWindow().Render()

#---------------------------------------------------------------------------------
class UnitsWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Units")
        self.resize(200, 200)
        self.settings = settings
        self.parent = parent  # Store parent reference

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
        old_unit_system = self.settings.unit_system
        new_unit_system = self.unit_system.currentText()
        
        # Update settings in parent
        self.settings.update_settings(
            self.unit_system.currentText(),
            self.temperature_unit.currentText(),
            self.angle_unit.currentText()
        )
        
        # Apply unit conversion to displayed geometry metrics
        if hasattr(self.parent, 'stl_geom') and self.parent.stl_geom and old_unit_system != new_unit_system:
            self.update_geometry_info(old_unit_system, new_unit_system)
            
        # Update any spinbox values in open windows
        self.update_open_windows(old_unit_system, new_unit_system)
        
        # Update the Units button icon to show a green checkmark
        if hasattr(self.parent, 'update_button_icon'):
            self.parent.update_button_icon("Units", "check")
        
        # Add a message to the parent's message text
        if hasattr(self.parent, 'message_text'):
            self.parent.message_text.append(f"Units updated: {self.unit_system.currentText()}, " + 
                                      f"{self.temperature_unit.currentText()}, {self.angle_unit.currentText()}")
        
        self.accept()
        
    def update_geometry_info(self, old_unit_system, new_unit_system):
        """Update geometry information display with new unit system"""
        # Update geometry display text if it exists
        if hasattr(self.parent, 'text_actor'):
            # Get current metrics and update with new unit system
            metrics = None
            if hasattr(self.parent, 'calculate_geometry_metrics'):
                metrics = self.parent.calculate_geometry_metrics()
            
            if metrics:
                # Convert length and volume
                length_unit = self.settings.get_length_unit_string()
                converted_length = self.settings.convert_length(metrics['length'], from_system=old_unit_system)
                
                # Volume conversion depends on unit system (cubic conversion)
                volume_conversion = pow(self.settings.unit_conversions["length"][new_unit_system] / 
                                      self.settings.unit_conversions["length"][old_unit_system], 3)
                converted_volume = metrics['volume'] * volume_conversion
                
                # Update display text
                info_text = (
                    f"{metrics['model']}\n"
                    f"Length: {converted_length:.2f} ({length_unit})\n"
                    f"Volume: {converted_volume:.2e} ({length_unit}^3)"
                )
                self.parent.text_actor.SetInput(info_text)
                self.parent.vtkWidget.GetRenderWindow().Render()
    
    def update_open_windows(self, old_unit_system, new_unit_system):
        """Update values in any open dialog windows"""
        # Find all child windows that might need unit conversion
        for child in self.parent.findChildren(QtWidgets.QDialog):
            # Skip self to avoid recursion
            if child is self:
                continue
                
            # Handle specific window types
            if isinstance(child, StructuralLoadsWindow):
                self.update_structural_loads_window(child, old_unit_system, new_unit_system)
            elif isinstance(child, ThermalLoadsWindow):
                self.update_thermal_loads_window(child, old_unit_system, new_unit_system)
            elif isinstance(child, MaterialWindow):
                self.update_material_window(child, old_unit_system, new_unit_system)
    
    def update_structural_loads_window(self, window, old_unit_system, new_unit_system):
        """Update force values in structural loads window"""
        if hasattr(window, 'x_force_spin'):
            # Convert force values
            window.x_force_spin.setValue(
                self.settings.convert_force(window.x_force_spin.value(), 
                                           from_system=old_unit_system, 
                                           to_system=new_unit_system)
            )
            window.y_force_spin.setValue(
                self.settings.convert_force(window.y_force_spin.value(), 
                                           from_system=old_unit_system, 
                                           to_system=new_unit_system)
            )
            window.z_force_spin.setValue(
                self.settings.convert_force(window.z_force_spin.value(), 
                                           from_system=old_unit_system, 
                                           to_system=new_unit_system)
            )
    
    def update_thermal_loads_window(self, window, old_unit_system, new_unit_system):
        """Update heat flux values in thermal loads window"""
        # Heat flux conversion (W/m² to W/mm² or W/in²)
        if hasattr(window, 'heat_flux_value_spin'):
            # Area conversion factor
            area_conversion = (self.settings.unit_conversions["length"][old_unit_system] / 
                              self.settings.unit_conversions["length"][new_unit_system]) ** 2
                              
            window.heat_flux_value_spin.setValue(
                window.heat_flux_value_spin.value() * area_conversion
            )
            
        # Total heat doesn't need conversion (Watts are the same in all unit systems)
        
        # For internal heat generation (W/m³), cube the conversion
        if hasattr(window, 'internal_heat_value_spin'):
            volume_conversion = (self.settings.unit_conversions["length"][old_unit_system] / 
                               self.settings.unit_conversions["length"][new_unit_system]) ** 3
                               
            window.internal_heat_value_spin.setValue(
                window.internal_heat_value_spin.value() * volume_conversion
            )
    
    def update_material_window(self, window, old_unit_system, new_unit_system):
        """Update material property values with unit conversion"""
        if not hasattr(window, 'young_input'):
            return
            
        # Convert Young's modulus (stress units)
        try:
            young_value = float(window.young_input.text())
            converted_young = self.settings.convert_stress(young_value, 
                                                         from_system=old_unit_system, 
                                                         to_system=new_unit_system)
            window.young_input.setText(f"{converted_young:.2e}")
        except ValueError:
            pass
            
        # Convert yield strength (stress units)
        try:
            yield_value = float(window.yield_input.text())
            converted_yield = self.settings.convert_stress(yield_value,
                                                        from_system=old_unit_system,
                                                        to_system=new_unit_system)
            window.yield_input.setText(f"{converted_yield:.2e}")
        except ValueError:
            pass
#---------------------------------------------------------------------------------   
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
        """Update settings with user preferences"""
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
        """Return the appropriate length unit string"""
        if self.unit_system == "MKS":
            return "m"
        elif self.unit_system == "mmKS":
            return "mm"
        elif self.unit_system == "IPS":
            return "in"
        return "m"  # Default
        
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
        """Return the temperature unit symbol"""
        if self.temperature_unit == "Celsius":
            return "°C"
        elif self.temperature_unit == "Fahrenheit":
            return "°F"
        else:  # Kelvin
            return "K"
#---------------------------------------------------------------------------------
class GeometryWindow(QtWidgets.QDialog):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.setWindowTitle("Geometry")
        self.resize(200, 100)
        self.main_window = main_window
        
        # Store references to main window components needed for rendering
        self.renderer = main_window.renderer
        self.vtkWidget = main_window.vtkWidget
        self.interactor = main_window.interactor
        self.message_text = main_window.message_text

        # Setup text actors first before trying to access them
        self.setup_geometry_info()

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Load button
        load_button = QtWidgets.QPushButton("Load Geometry")
        load_button.clicked.connect(self.load_geometry)
        layout.addWidget(load_button)
        
        # Close button
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
                self.main_window.stl_filepath = file_path
                self.load_stl_file(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load geometry: {str(e)}")
        self.close()

    def setup_geometry_info(self):
        """Setup the geometry information text overlay"""
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.text_actor.GetPositionCoordinate().SetValue(0.02, 0.91)  # Top left (x, y)
        self.text_actor.GetTextProperty().SetFontSize(20)  # Increased font size
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)  # Black text
        self.text_actor.GetTextProperty().SetFontFamilyToArial()
        self.text_actor.GetTextProperty().SetBold(True)
        self.renderer.AddActor2D(self.text_actor)

    def calculate_geometry_metrics(self):
        """Calculate geometry metrics for the loaded STL file"""
        if not self.main_window.stl_geom:
            return None

        # Get filename without path
        filename = os.path.basename(self.main_window.stl_geom.file_path)
        
        # Calculate volume and center of mass
        volume = 0
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3

        for vertices in self.main_window.stl_geom.mesh.vectors:
            # Update bounding box
            for vertex in vertices:
                for i in range(3):
                    min_coords[i] = min(min_coords[i], vertex[i])
                    max_coords[i] = max(max_coords[i], vertex[i])
            
            # Calculate volume contribution of this triangle
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            normal = np.cross(v1, v2)
            volume += np.dot(vertices[0], normal) / 6.0

        # Calculate length (maximum dimension)
        dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
        max_length = max(dimensions)

        return {
            'model': filename,
            'length': abs(max_length),
            'volume': abs(volume)
        }

    def update_geometry_info(self):
        """Update the geometry information display"""
        if not self.main_window.stl_geom:
            self.text_actor.SetInput("")
            return

        metrics = self.calculate_geometry_metrics()
        if metrics:
            # Get unit strings
            length_unit = self.main_window.settings.get_length_unit_string()
            
            info_text = (
                f"{metrics['model']}\n"
                f"Length: {metrics['length']:.2f} ({length_unit})\n"
                f"Volume: {metrics['volume']:.2e} ({length_unit}^3)"
            )
            self.text_actor.SetInput(info_text)

    def load_stl_file(self, file_path):
        """Load an STL file and display it in the main window with feature edges"""
        self.main_window.stl_filepath = file_path
        self.main_window.stl_geom = STLGeom(file_path)
        
        # Create vtkPolyData
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        for vertices in self.main_window.stl_geom.mesh.vectors:
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
        
        # Create mapper and main actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        self.main_window.stl_actor = vtk.vtkActor()
        self.main_window.stl_actor.SetMapper(mapper)
        self.main_window.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        
        # Extract only feature edges
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.BoundaryEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.FeatureEdgesOn()
        featureEdges.SetFeatureAngle(90)
        featureEdges.Update()
        
        edgeMapper = vtk.vtkPolyDataMapper()
        edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
        
        self.main_window.edge_actor = vtk.vtkActor()
        self.main_window.edge_actor.SetMapper(edgeMapper)
        self.main_window.edge_actor.GetProperty().SetColor(0, 0, 0)
        self.main_window.edge_actor.GetProperty().SetLineWidth(3)
        
        # Remove existing actors and add new ones
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor2D(self.text_actor)
        self.renderer.AddActor(self.main_window.stl_actor)
        self.renderer.AddActor(self.main_window.edge_actor)
        
        # Setup highlight actor
        self.main_window.highlight_poly_data = vtk.vtkPolyData()
        highlight_mapper = vtk.vtkPolyDataMapper()
        highlight_mapper.SetInputData(self.main_window.highlight_poly_data)
        
        self.main_window.highlight_actor = vtk.vtkActor()
        self.main_window.highlight_actor.SetMapper(highlight_mapper)
        self.main_window.highlight_actor.GetProperty().SetColor(1, 0, 0)
        self.main_window.highlight_actor.GetProperty().SetOpacity(0.6)
        
        self.renderer.AddActor(self.main_window.highlight_actor)
        
        # Configure depth peeling for proper transparency
        render_window = self.vtkWidget.GetRenderWindow()
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)
        self.renderer.UseDepthPeelingOn()
        self.renderer.SetMaximumNumberOfPeels(100)
        
        # Setup interactions (use main window's method)
        self.interactor.AddObserver("LeftButtonPressEvent", self.main_window.on_left_button_press)
        
        # Update geometry information
        self.update_geometry_info()

        # Get filename without path for display
        filename = os.path.basename(file_path)
        
        # Update model title
        # self.model_title_actor.SetInput(f"{filename}")
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        self.message_text.setText(f"Model loaded with {self.main_window.stl_geom.stl_n_triangles} triangles")

        # Update UI state in main window
        self.main_window.update_LivVar('geometry_loaded', True)

        # Call the function to update the sidebar
        self.main_window.on_geometry_loaded()
#---------------------------------------------------------------------------------
class DisplayOptionsWindow(QtWidgets.QDialog):
    """
    Display Options Window for setting visualization parameters.
    """
    def __init__(self, parent):
            super().__init__(parent)
            self.setWindowTitle("Display Options")
            self.resize(300, 500)
            self.parent = parent
            
            layout = QtWidgets.QVBoxLayout(self)
            layout.setSpacing(10)
            
            # Geometry dropdown
            geom_layout = QtWidgets.QHBoxLayout()
            geom_label = QtWidgets.QLabel("Geometry")
            self.geom_combo = QtWidgets.QComboBox()
            self.geom_combo.addItems(["None", "InitialDesign", "OptimizedDesign", "Both", "Mesh"])
            geom_layout.addWidget(geom_label)
            geom_layout.addWidget(self.geom_combo)
            layout.addLayout(geom_layout)
            
            # Field dropdown
            field_layout = QtWidgets.QHBoxLayout()
            field_label = QtWidgets.QLabel("Field")
            self.field_combo = QtWidgets.QComboBox()
            self.field_combo.addItem("None")  # Always have "None" as the first option
            # Other field options will be populated dynamically based on available analysis results
            field_layout.addWidget(field_label)
            field_layout.addWidget(self.field_combo)
            layout.addLayout(field_layout)
            
            # Display mode dropdown (geometry or mesh)
            display_mode_layout = QtWidgets.QHBoxLayout()
            display_mode_label = QtWidgets.QLabel("Display On")
            self.display_mode_combo = QtWidgets.QComboBox()
            self.display_mode_combo.addItems(["Geometry", "Mesh"])
            display_mode_layout.addWidget(display_mode_label)
            display_mode_layout.addWidget(self.display_mode_combo)
            layout.addLayout(display_mode_layout)
            
            # Cutting Percent spinboxes (X, Y, Z)
            # X Cutting Percent
            x_cutting_layout = QtWidgets.QHBoxLayout()
            x_cutting_label = QtWidgets.QLabel("XCuttingPercent")
            self.x_cutting_spin = QtWidgets.QSpinBox()
            self.x_cutting_spin.setRange(0, 100)
            self.x_cutting_spin.setValue(0)
            x_cutting_layout.addWidget(x_cutting_label)
            x_cutting_layout.addWidget(self.x_cutting_spin)
            layout.addLayout(x_cutting_layout)
            
            # Y Cutting Percent
            y_cutting_layout = QtWidgets.QHBoxLayout()
            y_cutting_label = QtWidgets.QLabel("YCuttingPercent")
            self.y_cutting_spin = QtWidgets.QSpinBox()
            self.y_cutting_spin.setRange(0, 100)
            self.y_cutting_spin.setValue(0)
            y_cutting_layout.addWidget(y_cutting_label)
            y_cutting_layout.addWidget(self.y_cutting_spin)
            layout.addLayout(y_cutting_layout)
            
            # Z Cutting Percent
            z_cutting_layout = QtWidgets.QHBoxLayout()
            z_cutting_label = QtWidgets.QLabel("ZCuttingPercent")
            self.z_cutting_spin = QtWidgets.QSpinBox()
            self.z_cutting_spin.setRange(0, 100)
            self.z_cutting_spin.setValue(0)
            z_cutting_layout.addWidget(z_cutting_label)
            z_cutting_layout.addWidget(self.z_cutting_spin)
            layout.addLayout(z_cutting_layout)
            
            # Eigen Number
            eigen_layout = QtWidgets.QHBoxLayout()
            eigen_label = QtWidgets.QLabel("EigenNumber")
            self.eigen_spin = QtWidgets.QSpinBox()
            self.eigen_spin.setRange(1, 10)
            self.eigen_spin.setValue(1)
            eigen_layout.addWidget(eigen_label)
            eigen_layout.addWidget(self.eigen_spin)
            layout.addLayout(eigen_layout)
            
            # Checkboxes
            self.show_bounding_box = QtWidgets.QCheckBox("Show bounding box")
            layout.addWidget(self.show_bounding_box)
            
            self.show_triangles = QtWidgets.QCheckBox("Show triangles")
            layout.addWidget(self.show_triangles)
            
            self.show_text = QtWidgets.QCheckBox("Show text")
            self.show_text.setChecked(True)
            layout.addWidget(self.show_text)
            
            self.scale_deformation = QtWidgets.QCheckBox("Scale deformation")
            self.scale_deformation.setChecked(True)
            layout.addWidget(self.scale_deformation)
            
            self.show_transparent = QtWidgets.QCheckBox("Show transparent geometry")
            layout.addWidget(self.show_transparent)
            
            self.show_axis = QtWidgets.QCheckBox("Show axis")
            self.show_axis.setChecked(True)
            layout.addWidget(self.show_axis)
            
            self.show_structural_loads = QtWidgets.QCheckBox("Show structural loads")
            self.show_structural_loads.setChecked(True)
            layout.addWidget(self.show_structural_loads)
            
            self.show_thermal_loads = QtWidgets.QCheckBox("Show thermal loads")
            layout.addWidget(self.show_thermal_loads)
            
            self.show_topopt_constraints = QtWidgets.QCheckBox("Show TopOpt Constraints")
            self.show_topopt_constraints.setChecked(True)
            layout.addWidget(self.show_topopt_constraints)
            
            self.show_non_design = QtWidgets.QCheckBox("Show non-design parts")
            self.show_non_design.setChecked(True)
            layout.addWidget(self.show_non_design)
            
            # Action buttons
            self.animate_button = QtWidgets.QPushButton("Animate for 3 cycles")
            self.animate_button.clicked.connect(self.animate_view)
            layout.addWidget(self.animate_button)
            
            self.hide_selected_button = QtWidgets.QPushButton("Hide Selected Part")
            self.hide_selected_button.clicked.connect(self.hide_selected_part)
            layout.addWidget(self.hide_selected_button)
            
            self.save_image_button = QtWidgets.QPushButton("Save Image")
            self.save_image_button.clicked.connect(self.save_image)
            layout.addWidget(self.save_image_button)
            
            self.reset_view_button = QtWidgets.QPushButton("Reset View")
            self.reset_view_button.clicked.connect(self.reset_view)
            layout.addWidget(self.reset_view_button)
            
            self.close_button = QtWidgets.QPushButton("Close")
            self.close_button.clicked.connect(self.close)
            layout.addWidget(self.close_button)
            
            # Initially block all signals to prevent actions when loading
            self.blockAllSignals(True)
            
            # Update field options based on available analysis results
            self.update_field_options()
            
            # Load current settings without applying them
            self.load_current_settings(apply_immediately=False)
            
            # Set the enabled state of controls that depend on analysis results
            self.update_controls_enabled_state()
            
            # Now connect controls to handlers
            self.blockAllSignals(False)
            self.connect_controls()

    def showEvent(self, event):
        """Override showEvent to update field options when dialog is shown"""
        # Update field options based on available analysis results
        self.update_field_options()
        
        # Update enabled state of controls
        self.update_controls_enabled_state()
        
        # Call parent implementation
        super().showEvent(event)

    def blockAllSignals(self, block):
        """Block or unblock signals for all interactive widgets"""
        widgets = [
            self.geom_combo, self.field_combo, self.display_mode_combo,
            self.x_cutting_spin, self.y_cutting_spin, self.z_cutting_spin,
            self.eigen_spin, self.show_bounding_box, self.show_triangles,
            self.show_text, self.scale_deformation, self.show_transparent,
            self.show_axis, self.show_structural_loads, self.show_thermal_loads,
            self.show_topopt_constraints, self.show_non_design
        ]
        
        for widget in widgets:
            widget.blockSignals(block)

    def update_field_options(self):
        """Update the field dropdown options based on available analysis results"""
        # Store current selection if any
        current_selection = self.field_combo.currentText()
        
        # Clear all items except "None"
        while self.field_combo.count() > 1:
            self.field_combo.removeItem(1)
        
        # Check for available analysis results
        has_structural_results = hasattr(self.parent, 'analysis_results') and self.parent.analysis_results is not None
        has_thermal_results = hasattr(self.parent, 'thermal_results') and self.parent.thermal_results is not None
        
        # Add appropriate field options
        if has_structural_results:
            self.field_combo.addItem("Displacement")
            self.field_combo.addItem("Von Mises Stress")
            
        if has_thermal_results:
            self.field_combo.addItem("Temperature")
        
        # Try to restore previous selection if it exists in the new list
        index = self.field_combo.findText(current_selection)
        if index >= 0:
            self.field_combo.setCurrentIndex(index)
        else:
            # If no previous selection or it's no longer available, make a smart default choice
            if has_thermal_results and not has_structural_results:
                # Only thermal results available, select Temperature
                temp_index = self.field_combo.findText("Temperature")
                if temp_index >= 0:
                    self.field_combo.setCurrentIndex(temp_index)
            elif has_structural_results and not has_thermal_results:
                # Only structural results available, select Displacement
                disp_index = self.field_combo.findText("Displacement")
                if disp_index >= 0:
                    self.field_combo.setCurrentIndex(disp_index)
            else:
                # Default to "None" or choose based on most recent analysis
                self.field_combo.setCurrentIndex(0)  # Default to "None"

    def update_controls_enabled_state(self):
        """Update the enabled state of controls based on available data"""
        # Check if we have geometry
        has_geometry = hasattr(self.parent, 'stl_geom') and self.parent.stl_geom is not None
        
        # Check if we have mesh
        has_mesh = hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor is not None
        
        # Check for available analysis results
        has_results = hasattr(self.parent, 'analysis_results') and self.parent.analysis_results is not None
        has_thermal = hasattr(self.parent, 'thermal_results') and self.parent.thermal_results is not None
        
        # Enable/disable controls based on available data
        self.geom_combo.setEnabled(has_geometry or has_mesh)
        self.display_mode_combo.setEnabled(has_geometry and has_mesh)
        
        # Enable scale deformation only if we have displacement results
        self.scale_deformation.setEnabled(has_results)
        
        # Enable animate button only if we have results and scale deformation is checked
        self.animate_button.setEnabled(has_results and self.scale_deformation.isChecked())
        
        # Enable structural loads toggle only if we have structural loads
        # This is the problematic line - force_actors is a list, not a boolean
        has_structural_loads = hasattr(self.parent, 'force_actors') and len(self.parent.force_actors) > 0
        self.show_structural_loads.setEnabled(has_structural_loads)
        
        # Enable thermal loads toggle only if we have thermal loads
        has_thermal_loads = (hasattr(self.parent, 'fixed_temp_actors') and len(self.parent.fixed_temp_actors) > 0) or \
                            (hasattr(self.parent, 'heat_flux_actors') and len(self.parent.heat_flux_actors) > 0) or \
                            (hasattr(self.parent, 'convection_actors') and len(self.parent.convection_actors) > 0)
        self.show_thermal_loads.setEnabled(has_thermal_loads)
        
        # Enable topopt constraints toggle only if we have topopt constraints
        has_topopt = hasattr(self.parent, 'topopt_constraint_actors') and self.parent.topopt_constraint_actors
        self.show_topopt_constraints.setEnabled(has_topopt)
        
        # Enable non-design toggle only if we have non-design parts
        has_non_design = hasattr(self.parent, 'non_design_actors') and self.parent.non_design_actors
        self.show_non_design.setEnabled(has_non_design)

    def connect_controls(self):
        """Connect controls to update handlers"""
        # Connect dropdowns
        self.geom_combo.currentIndexChanged.connect(self.update_geometry_display)
        self.field_combo.currentIndexChanged.connect(self.update_field_display)  
        self.display_mode_combo.currentIndexChanged.connect(self.update_display)
        
        # Connect spinboxes
        self.x_cutting_spin.valueChanged.connect(self.update_cutting_planes)
        self.y_cutting_spin.valueChanged.connect(self.update_cutting_planes)
        self.z_cutting_spin.valueChanged.connect(self.update_cutting_planes)
        self.eigen_spin.valueChanged.connect(self.update_display)
        
        # Connect checkboxes
        self.show_bounding_box.stateChanged.connect(lambda: self.toggle_feature('bounding_box'))
        self.show_triangles.stateChanged.connect(lambda: self.toggle_feature('triangles'))
        self.show_text.stateChanged.connect(lambda: self.toggle_feature('text'))
        self.scale_deformation.stateChanged.connect(self.handle_scale_deformation_changed)
        self.show_transparent.stateChanged.connect(lambda: self.toggle_feature('transparent'))
        self.show_axis.stateChanged.connect(lambda: self.toggle_feature('axis'))
        self.show_structural_loads.stateChanged.connect(lambda: self.toggle_feature('structural_loads'))
        self.show_thermal_loads.stateChanged.connect(lambda: self.toggle_feature('thermal_loads'))
        self.show_topopt_constraints.stateChanged.connect(lambda: self.toggle_feature('topopt_constraints'))
        self.show_non_design.stateChanged.connect(lambda: self.toggle_feature('non_design'))
    
    def handle_scale_deformation_changed(self):
        """Handle changes to the scale deformation checkbox"""
        is_checked = self.scale_deformation.isChecked()
        
        # Update display settings
        if hasattr(self.parent, 'display_settings'):
            self.parent.display_settings['scale_deformation'] = is_checked
        
        # Enable/disable animate button based on checkbox state and available results
        has_results = hasattr(self.parent, 'analysis_results') and self.parent.analysis_results is not None
        self.animate_button.setEnabled(is_checked and has_results)
        
        # Toggle the feature
        self.toggle_feature('scale_deformation')
    
    def load_current_settings(self, apply_immediately=False):
        """
        Load current display settings from parent without immediately applying changes
        
        Parameters:
        -----------
        apply_immediately : bool
            Whether to apply the settings to the visualization immediately
        """
        # Initialize default settings if not present in parent
        if not hasattr(self.parent, 'display_settings'):
            self.parent.display_settings = {
                'geometry': 'InitialDesign',
                'field': 'None',
                'display_mode': 'Geometry',  # Default to display on geometry
                'x_cutting': 0,
                'y_cutting': 0,
                'z_cutting': 0,
                'eigen_number': 1,
                'show_bounding_box': False,
                'show_results': False,
                'show_triangles': False,
                'show_text': True,
                'scale_deformation': True,
                'show_transparent': False,
                'show_axis': True,
                'show_structural_loads': True,
                'show_thermal_loads': False,
                'show_topopt_constraints': True,
                'show_non_design': True,
            }
            
        settings = self.parent.display_settings
        
        # Block signals to prevent auto-update while setting values
        self.blockAllSignals(True)
        
        # Set dropdowns
        if 'geometry' in settings:
            # If "None" is selected or previously had "show_mesh" enabled, select the appropriate option
            if settings.get('geometry') == 'None':
                self.geom_combo.setCurrentText("None")
            elif settings.get('show_mesh', False) and settings['geometry'] == 'InitialDesign':
                self.geom_combo.setCurrentText("Mesh")
            else:
                index = self.geom_combo.findText(settings['geometry'])
                if index >= 0:
                    self.geom_combo.setCurrentIndex(index)
        
        if 'field' in settings:
            index = self.field_combo.findText(settings['field'])
            if index >= 0:
                self.field_combo.setCurrentIndex(index)
                
        if 'display_mode' in settings:
            index = self.display_mode_combo.findText(settings['display_mode'])
            if index >= 0:
                self.display_mode_combo.setCurrentIndex(index)
        
        # Set spinboxes
        if 'x_cutting' in settings:
            self.x_cutting_spin.setValue(settings['x_cutting'])
        if 'y_cutting' in settings:
            self.y_cutting_spin.setValue(settings['y_cutting'])
        if 'z_cutting' in settings:
            self.z_cutting_spin.setValue(settings['z_cutting'])
        if 'eigen_number' in settings:
            self.eigen_spin.setValue(settings['eigen_number'])
        
        # Set checkboxes
        if 'show_bounding_box' in settings:
            self.show_bounding_box.setChecked(settings['show_bounding_box'])
        if 'show_triangles' in settings:
            self.show_triangles.setChecked(settings['show_triangles'])
        if 'show_text' in settings:
            self.show_text.setChecked(settings['show_text'])
        if 'scale_deformation' in settings:
            self.scale_deformation.setChecked(settings['scale_deformation'])
        if 'show_transparent' in settings:
            self.show_transparent.setChecked(settings['show_transparent'])
        if 'show_axis' in settings:
            self.show_axis.setChecked(settings['show_axis'])
        if 'show_structural_loads' in settings:
            self.show_structural_loads.setChecked(settings['show_structural_loads'])
        if 'show_thermal_loads' in settings:
            self.show_thermal_loads.setChecked(settings['show_thermal_loads'])
        if 'show_topopt_constraints' in settings:
            self.show_topopt_constraints.setChecked(settings['show_topopt_constraints'])
        if 'show_non_design' in settings:
            self.show_non_design.setChecked(settings['show_non_design'])

        # Restore signals
        self.blockAllSignals(False)

        # Only apply changes if requested
        if apply_immediately:
            self.update_visualization()

    def update_geometry_display(self):
        """Update the display based on geometry dropdown selection"""
        selected_geometry = self.geom_combo.currentText()
        
        # First, hide all geometry and results actors to start with a clean slate
        if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
            self.parent.stl_actor.SetVisibility(False)
        
        if hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor:
            self.parent.mesh_actor.SetVisibility(False)
            
        # Update settings
        if hasattr(self.parent, 'display_settings'):
            if selected_geometry == "None":
                # When "None" is selected, hide all geometry and mesh
                self.parent.display_settings['geometry'] = 'None'
                self.parent.display_settings['show_mesh'] = False
                
                # Results actors are already hidden above, but for clarity:
                if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
                    self.parent.results_actor.SetVisibility(False)
                    
                if hasattr(self.parent, 'results_mesh_actor') and self.parent.results_mesh_actor:
                    self.parent.results_mesh_actor.SetVisibility(False)
                    
                # Hide scalar bar if it exists
                if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar:
                    self.parent.scalar_bar.SetVisibility(False)
                
                # Hide structural loads
                if hasattr(self.parent, 'force_actors') and self.parent.force_actors:
                    for actor in self.parent.force_actors:
                        if actor:  # Check if actor is not None
                            actor.SetVisibility(False)
                            
                if hasattr(self.parent, 'constraint_actors') and self.parent.constraint_actors:
                    for actor in self.parent.constraint_actors:
                        if actor:  # Check if actor is not None
                            actor.SetVisibility(False)
                
                # Hide thermal loads
                thermal_actor_lists = [
                    'fixed_temp_actors',
                    'heat_flux_actors',
                    'total_heat_actors',
                    'heat_source_actors',
                    'convection_actors',
                    'radiation_actors',
                    'internal_heat_actors'
                ]
                
                for actor_list_name in thermal_actor_lists:
                    if hasattr(self.parent, actor_list_name):
                        actors = getattr(self.parent, actor_list_name)
                        if actors and isinstance(actors, list):
                            for actor in actors:
                                if actor:  # Check if actor is not None
                                    actor.SetVisibility(False)
                
                # Hide topopt constraints
                if hasattr(self.parent, 'topopt_constraint_actors') and self.parent.topopt_constraint_actors:
                    for actor in self.parent.topopt_constraint_actors:
                        if actor:  # Check if actor is not None
                            actor.SetVisibility(False)
                            
                # Hide non-design parts
                if hasattr(self.parent, 'non_design_actors') and self.parent.non_design_actors:
                    for actor in self.parent.non_design_actors:
                        if actor:  # Check if actor is not None
                            actor.SetVisibility(False)
                            
                # Update checkbox states in the UI to reflect the actual visibility
                # Block signals temporarily to prevent recursive updates
                self.blockAllSignals(True)
                self.parent.display_settings['show_structural_loads'] = False
                self.parent.display_settings['show_thermal_loads'] = False
                self.parent.display_settings['show_topopt_constraints'] = False
                self.parent.display_settings['show_non_design'] = False
                
                self.show_structural_loads.setChecked(False)
                self.show_thermal_loads.setChecked(False)
                self.show_topopt_constraints.setChecked(False)
                self.show_non_design.setChecked(False)
                self.blockAllSignals(False)
                
                # Add message to log
                if hasattr(self.parent, 'message_text'):
                    self.parent.message_text.append("Hiding all geometry and visualization elements")
                
            elif selected_geometry == "Mesh":
                # When "Mesh" is selected, show mesh only
                self.parent.display_settings['geometry'] = 'InitialDesign'
                self.parent.display_settings['show_mesh'] = True

                # Hide geometry
                if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                    self.parent.stl_actor.SetVisibility(False)
                
                # Set mesh visibility
                if hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor:
                    self.parent.mesh_actor.SetVisibility(True)
                
                
                # Check if results should be shown (only if field is not None)
                show_results = self.field_combo.currentText() != "None"
                if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
                    self.parent.results_actor.SetVisibility(show_results)
                
                if hasattr(self.parent, 'results_mesh_actor') and self.parent.results_mesh_actor:
                    self.parent.results_mesh_actor.SetVisibility(show_results)
                    
                # Show scalar bar if results are visible
                if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar:
                    self.parent.scalar_bar.SetVisibility(show_results)
                
                # Restore state of loads visibility to match checkboxes
                self.toggle_feature('structural_loads')
                self.toggle_feature('thermal_loads')
                self.toggle_feature('topopt_constraints')
                self.toggle_feature('non_design')
                
            else:
                # For other geometry selections
                self.parent.display_settings['geometry'] = selected_geometry
                self.parent.display_settings['show_mesh'] = False
                
                # Show geometry, hide mesh
                if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                    self.parent.stl_actor.SetVisibility(True)
                
                if hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor:
                    self.parent.mesh_actor.SetVisibility(False)
                
                # Check if results should be shown
                show_results = self.field_combo.currentText() != "None"
                
                # Set visibility of results actors based on results flag
                if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
                    self.parent.results_actor.SetVisibility(show_results)
                
                if hasattr(self.parent, 'results_mesh_actor') and self.parent.results_mesh_actor:
                    self.parent.results_mesh_actor.SetVisibility(False)  # Always hide mesh results when showing geometry
                
                # Show or hide scalar bar based on results visibility
                if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar:
                    self.parent.scalar_bar.SetVisibility(show_results)
                
                # If showing results and not transparent, hide original geometry
                if show_results and not self.show_transparent.isChecked() and hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                    self.parent.stl_actor.SetVisibility(False)
                
                # Restore state of loads visibility to match checkboxes
                self.toggle_feature('structural_loads')
                self.toggle_feature('thermal_loads')
                self.toggle_feature('topopt_constraints')
                self.toggle_feature('non_design')
                
                # Add message to log
                if hasattr(self.parent, 'message_text'):
                    self.parent.message_text.append(f"Showing {selected_geometry}")
        
        # Render the changes
        self.parent.vtkWidget.GetRenderWindow().Render()
        
        # Update settings
        self.update_display()

    def update_field_display(self):
        """Update the display based on field dropdown selection"""
        selected_field = self.field_combo.currentText()
        
        # Update settings
        if hasattr(self.parent, 'display_settings'):
            self.parent.display_settings['field'] = selected_field
            
            # Toggle result visibility based on field selection
            show_results = selected_field != "None"
            self.parent.display_settings['show_results'] = show_results
            
            # Toggle results visibility
            self.toggle_results_visibility(show_results)
            
            # Apply field visualization if a field is selected and results should be shown
            if show_results:
                self.apply_field_visualization(selected_field)
            
            # If geometry is set to None, keep everything hidden regardless of field selection
            if self.geom_combo.currentText() == "None":
                if hasattr(self.parent, 'stl_actor'):
                    self.parent.stl_actor.SetVisibility(False)
                if hasattr(self.parent, 'mesh_actor'):
                    self.parent.mesh_actor.SetVisibility(False)
                if hasattr(self.parent, 'results_actor'):
                    self.parent.results_actor.SetVisibility(False)
                if hasattr(self.parent, 'results_mesh_actor'):
                    self.parent.results_mesh_actor.SetVisibility(False)
                if hasattr(self.parent, 'scalar_bar'):
                    self.parent.scalar_bar.SetVisibility(False)
            else:
                # If not showing results but geometry is selected, make sure geometry is visible
                if not show_results and self.geom_combo.currentText() != "Mesh":
                    if hasattr(self.parent, 'stl_actor'):
                        self.parent.stl_actor.SetVisibility(True)
                elif not show_results and self.geom_combo.currentText() == "Mesh":
                    if hasattr(self.parent, 'mesh_actor'):
                        self.parent.mesh_actor.SetVisibility(True)
        
        # Render the changes
        self.parent.vtkWidget.GetRenderWindow().Render()
        
        # Add message to log
        if hasattr(self.parent, 'message_text'):
            if selected_field == "None":
                self.parent.message_text.append("Hiding results")
            else:
                self.parent.message_text.append(f"Showing {selected_field} field")

    def update_display(self):
        """Update the display based on current settings"""
        # Get current settings
        settings = {
            'geometry': self.geom_combo.currentText(),
            'field': self.field_combo.currentText(),
            'display_mode': self.display_mode_combo.currentText(),
            'x_cutting': self.x_cutting_spin.value(),
            'y_cutting': self.y_cutting_spin.value(),
            'z_cutting': self.z_cutting_spin.value(),
            'eigen_number': self.eigen_spin.value(),
            'show_bounding_box': self.show_bounding_box.isChecked(),
            'show_triangles': self.show_triangles.isChecked(),
            'show_text': self.show_text.isChecked(),
            'show_mesh': self.geom_combo.currentText() == "Mesh",  # Derive from geometry dropdown
            'scale_deformation': self.scale_deformation.isChecked(),
            'show_transparent': self.show_transparent.isChecked(),
            'show_axis': self.show_axis.isChecked(),
            'show_structural_loads': self.show_structural_loads.isChecked(),
            'show_thermal_loads': self.show_thermal_loads.isChecked(),
            'show_topopt_constraints': self.show_topopt_constraints.isChecked(),
            'show_non_design': self.show_non_design.isChecked(),
            'show_results': self.field_combo.currentText() != "None"  # Derive from field dropdown
        }
        
        # Save settings to parent
        self.parent.display_settings = settings
        
        # Update enabled state of controls
        self.update_controls_enabled_state()
        
        # Update visualization
        self.update_visualization()
    
    def toggle_mesh(self, show):
        """Toggle mesh visibility - now controlled through geometry dropdown"""
        if hasattr(self.parent, 'mesh_actor'):
            # Set mesh visibility based on parameter
            self.parent.mesh_actor.SetVisibility(show)
            
            # IMPORTANT: If showing mesh, explicitly hide geometry
            if show and hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                self.parent.stl_actor.SetVisibility(False)
            
            # Update display settings
            if hasattr(self.parent, 'display_settings'):
                self.parent.display_settings['show_mesh'] = show
            
            # Update geometry dropdown to reflect mesh state if needed
            current_selection = self.geom_combo.currentText()
            if show and current_selection != "Mesh":
                self.geom_combo.setCurrentText("Mesh")
            elif not show and current_selection == "Mesh":
                self.geom_combo.setCurrentText("InitialDesign")
            
            # Add message to log
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"{'Showing mesh only (geometry hidden)' if show else 'Hiding mesh'}")
            
            # Render to show changes immediately
            self.parent.vtkWidget.GetRenderWindow().Render()

    def update_cutting_planes(self):
        """Update cutting planes based on spinbox values"""
        x_percent = self.x_cutting_spin.value()
        y_percent = self.y_cutting_spin.value()
        z_percent = self.z_cutting_spin.value()
        
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
            
        # Example implementation to create cutting planes
        try:
            # Get model bounds
            points = np.array(self.parent.stl_geom.mesh.points)
            bounds = [
                points[:,0].min(), points[:,0].max(),
                points[:,1].min(), points[:,1].max(),
                points[:,2].min(), points[:,2].max()
            ]
            
            # Calculate cutting plane positions
            x_pos = bounds[0] + (bounds[1] - bounds[0]) * (x_percent / 100.0)
            y_pos = bounds[2] + (bounds[3] - bounds[2]) * (y_percent / 100.0)
            z_pos = bounds[4] + (bounds[5] - bounds[4]) * (z_percent / 100.0)
            
            # Create or update cutting planes in VTK
            self.update_vtk_cutting_planes(x_pos, y_pos, z_pos, x_percent, y_percent, z_percent)
            
            # Save to settings
            self.parent.display_settings['x_cutting'] = x_percent
            self.parent.display_settings['y_cutting'] = y_percent
            self.parent.display_settings['z_cutting'] = z_percent
            
        except Exception as e:
            print(f"Error updating cutting planes: {str(e)}")
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"Error updating cutting planes: {str(e)}")

    def update_vtk_cutting_planes(self, x_pos, y_pos, z_pos, x_percent, y_percent, z_percent):
        """Create or update VTK cutting planes"""
        # Initialize cutting plane actors if not exists
        if not hasattr(self.parent, 'cutting_plane_actors'):
            self.parent.cutting_plane_actors = []
        
        # Remove existing cutting planes
        for actor in self.parent.cutting_plane_actors:
            self.parent.renderer.RemoveActor(actor)
        
        self.parent.cutting_plane_actors = []
        
        # Only create planes for non-zero percentages
        if x_percent > 0:
            self.create_cutting_plane(x_pos, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        
        if y_percent > 0:
            self.create_cutting_plane(y_pos, [0, 1, 0], [1, 0, 0], [0, 0, 1])
            
        if z_percent > 0:
            self.create_cutting_plane(z_pos, [0, 0, 1], [1, 0, 0], [0, 1, 0])
            
        # Update display
        self.parent.vtkWidget.GetRenderWindow().Render()

    def create_cutting_plane(self, position, normal, x_axis, y_axis):
        """Create a single cutting plane actor"""
        try:
            # Get model bounds
            points = np.array(self.parent.stl_geom.mesh.points)
            bounds = [
                points[:,0].min(), points[:,0].max(),
                points[:,1].min(), points[:,1].max(),
                points[:,2].min(), points[:,2].max()
            ]
            
            # Calculate plane size based on model bounds
            plane_size = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4]
            ) * 1.2  # Make plane slightly larger than model
            
            # Create plane
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(
                position - x_axis[0] * plane_size/2 - y_axis[0] * plane_size/2,
                position - x_axis[1] * plane_size/2 - y_axis[1] * plane_size/2,
                position - x_axis[2] * plane_size/2 - y_axis[2] * plane_size/2
            )
            plane.SetPoint1(
                position + x_axis[0] * plane_size/2 - y_axis[0] * plane_size/2,
                position + x_axis[1] * plane_size/2 - y_axis[1] * plane_size/2,
                position + x_axis[2] * plane_size/2 - y_axis[2] * plane_size/2
            )
            plane.SetPoint2(
                position - x_axis[0] * plane_size/2 + y_axis[0] * plane_size/2,
                position - x_axis[1] * plane_size/2 + y_axis[1] * plane_size/2,
                position - x_axis[2] * plane_size/2 + y_axis[2] * plane_size/2
            )
            plane.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.2)
            actor.GetProperty().SetColor(0.8, 0.8, 0.9)
            
            # Add to renderer and store reference
            self.parent.renderer.AddActor(actor)
            self.parent.cutting_plane_actors.append(actor)
            
            return actor
        except Exception as e:
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"Error creating cutting plane: {str(e)}")
            return None

    def toggle_feature(self, feature_name):
        """Toggle visibility of a specific visualization feature"""
        # Update the parent's display settings
        if hasattr(self.parent, 'display_settings'):
            setting_key = f'show_{feature_name}'
            if setting_key in self.parent.display_settings:
                if feature_name == 'bounding_box':
                    is_checked = self.show_bounding_box.isChecked()
                    self.parent.display_settings[setting_key] = is_checked
                    self.toggle_bounding_box(is_checked)
                elif feature_name == 'triangles':
                    is_checked = self.show_triangles.isChecked()
                    self.parent.display_settings[setting_key] = is_checked
                    self.toggle_triangles(is_checked)
                elif feature_name == 'results':
                    is_checked = self.field_combo.currentText() != "None"
                    self.parent.display_settings[setting_key] = is_checked
                    self.toggle_results_visibility(is_checked)
                else:
                    # Generic handling for other features
                    checkbox = getattr(self, f'show_{feature_name}', None)
                    if checkbox:
                        self.parent.display_settings[setting_key] = checkbox.isChecked()
                        
        # Handle specific features
        if feature_name == 'axis':
            if hasattr(self.parent, 'axes_widget'):
                self.parent.axes_widget.SetEnabled(self.show_axis.isChecked())
        
        elif feature_name == 'structural_loads':
            if hasattr(self.parent, 'force_actors'):
                for actor in self.parent.force_actors:
                    actor.SetVisibility(self.show_structural_loads.isChecked())
            
            if hasattr(self.parent, 'constraint_actors'):
                for actor in self.parent.constraint_actors:
                    actor.SetVisibility(self.show_structural_loads.isChecked())
        
        elif feature_name == 'text':
            if hasattr(self.parent, 'text_actor'):
                self.parent.text_actor.SetVisibility(self.show_text.isChecked())
        
        elif feature_name == 'triangles' or feature_name == 'edges':
            if hasattr(self.parent, 'toggle_edges'):
                self.parent.toggle_edges(self.show_triangles.isChecked())
            else:
                self.toggle_triangles(self.show_triangles.isChecked())
        
        elif feature_name == 'transparent':
            if hasattr(self.parent, 'stl_actor'):
                # Get transparency state
                is_transparent = self.show_transparent.isChecked()
                
                # Check if any results are visible
                has_visible_results = hasattr(self.parent, 'results_actor') and \
                                    self.parent.results_actor and \
                                    self.parent.results_actor.GetVisibility()
                
                # Only adjust if geometry should be visible
                if is_transparent:
                    # Show original geometry with transparency when checkbox is checked
                    self.parent.stl_actor.SetVisibility(True)
                    self.parent.stl_actor.GetProperty().SetOpacity(0.6)
                    self.parent.stl_actor.GetProperty().SetColor(0.78, 0.86, 1.0)
                    
                    # If results are visible, adjust their opacity too
                    if has_visible_results:
                        self.parent.results_actor.GetProperty().SetOpacity(0.8)
                else:
                    # Return to normal opacity
                    self.parent.stl_actor.GetProperty().SetOpacity(1.0)
                    self.parent.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                    # Turn off edge visibility
                    self.parent.stl_actor.GetProperty().EdgeVisibilityOff()
                    
                    # If results are being displayed, hide the original geometry
                    if has_visible_results:
                        self.parent.stl_actor.SetVisibility(False)
                    else:
                        self.parent.stl_actor.SetVisibility(True)
                        
                # Render to show changes immediately
                self.parent.vtkWidget.GetRenderWindow().Render()
        
        elif feature_name == 'bounding_box':
            self.toggle_bounding_box(self.show_bounding_box.isChecked())
                
        elif feature_name == 'topopt_constraints':
            # Implement logic to show/hide optimization constraints
            if hasattr(self.parent, 'topopt_constraint_actors'):
                for actor in self.parent.topopt_constraint_actors:
                    actor.SetVisibility(self.show_topopt_constraints.isChecked())
                    
        elif feature_name == 'thermal_loads':
            # Handle all types of thermal load visualization actors
            is_checked = self.show_thermal_loads.isChecked()
            self.parent.display_settings['show_thermal_loads'] = is_checked
            
            # Create a comprehensive list of all thermal load actor lists to check
            thermal_actor_lists = [
                'fixed_temp_actors',
                'heat_source_actors',
                'total_heat_actors', 
                'convection_actors',
                'radiation_actors',
                'internal_heat_actors'
            ]
            
            # Loop through each thermal actor list and update visibility
            for actor_list_name in thermal_actor_lists:
                if hasattr(self.parent, actor_list_name):
                    actors = getattr(self.parent, actor_list_name)
                    if actors and isinstance(actors, list):
                        for actor in actors:
                            if actor:  # Make sure it's not None
                                actor.SetVisibility(is_checked)
                                    
            # Add message to log
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"{'Showing' if is_checked else 'Hiding'} thermal loads")
                    
        elif feature_name == 'non_design':
            # Implement logic to show/hide non-design parts
            if hasattr(self.parent, 'non_design_actors'):
                for actor in self.parent.non_design_actors:
                    actor.SetVisibility(self.show_non_design.isChecked())
                    
        elif feature_name == 'scale_deformation':
            # Handle scale deformation toggle
            is_checked = self.scale_deformation.isChecked()
            
            # Check if we have an active field that can be affected by scale deformation
            has_displacement_field = self.field_combo.currentText() == "Displacement"
            
            # Apply scaling only if we're showing displacement field
            if has_displacement_field and hasattr(self.parent, 'analysis_results'):
                if is_checked and hasattr(self.parent, 'apply_displacement_scaling'):
                    # Call the parent's method to apply scaling
                    self.parent.apply_displacement_scaling()
                elif not is_checked and hasattr(self.parent, 'reset_displacement_scaling'):
                    # Call the parent's method to reset scaling
                    self.parent.reset_displacement_scaling()
                    
            # Update the enable state of the animate button
            has_results = hasattr(self.parent, 'analysis_results') and self.parent.analysis_results is not None
            self.animate_button.setEnabled(is_checked and has_results)
                    
        # Render to show changes
        self.parent.vtkWidget.GetRenderWindow().Render()
            
    def toggle_triangles(self, show):
        """Toggle triangles/edges visibility"""
        if hasattr(self.parent, 'stl_actor'):
            if show:
                self.parent.stl_actor.GetProperty().SetRepresentationToWireframe()
            else:
                self.parent.stl_actor.GetProperty().SetRepresentationToSurface()
                
            # Render to show changes immediately
            self.parent.vtkWidget.GetRenderWindow().Render()

    def toggle_results_visibility(self, show):
        """Toggle visibility of analysis results"""
        # Check if the geometry selection is "None" - if so, don't show anything
        if self.geom_combo.currentText() == "None":
            # Even if results should be shown, don't show them if geometry is set to None
            show = False
        
        if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
            self.parent.results_actor.SetVisibility(show)
            
            # Also toggle scalar bar visibility
            if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar:
                self.parent.scalar_bar.SetVisibility(show)
            
            # Update display settings
            if hasattr(self.parent, 'display_settings'):
                self.parent.display_settings['show_results'] = show
            
            # Add message to log
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"{'Showing' if show else 'Hiding'} analysis results")
            
            # If results are shown, hide original geometry unless transparency is enabled
            if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                if show and not self.show_transparent.isChecked():
                    self.parent.stl_actor.SetVisibility(False)
                elif not show:
                    self.parent.stl_actor.SetVisibility(True)
                    
            # Render to show changes immediately
            self.parent.vtkWidget.GetRenderWindow().Render()
    
    def toggle_geometry_visibility(self, show):
        """Toggle geometry visibility"""
        # Check if the geometry selection is "None" - if so, don't show anything
        if self.geom_combo.currentText() == "None":
            show = False
            
        if hasattr(self.parent, 'stl_actor'):
            self.parent.stl_actor.SetVisibility(show)
            
            # Update display settings
            if hasattr(self.parent, 'display_settings'):
                self.parent.display_settings['show_geometry'] = show
            
            # If transparency is enabled and both geometry and results are visible,
            # apply transparency to the geometry
            if show and self.show_transparent.isChecked() and hasattr(self.parent, 'results_actor') and \
               self.parent.results_actor and self.parent.results_actor.GetVisibility():
                self.parent.stl_actor.GetProperty().SetOpacity(0.5)
            else:
                # Otherwise use full opacity
                self.parent.stl_actor.GetProperty().SetOpacity(1.0)
            
            # Render to show changes immediately
            self.parent.vtkWidget.GetRenderWindow().Render()
            
            # Add message to log
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"{'Showing' if show else 'Hiding'} geometry")
    
    def toggle_bounding_box(self, show):
        """Toggle bounding box visibility"""
        if not hasattr(self.parent, 'bounding_box_actor'):
            # Create bounding box if it doesn't exist
            if hasattr(self.parent, 'stl_geom') and self.parent.stl_geom is not None:
                try:
                    # Get model bounds
                    points = np.array(self.parent.stl_geom.mesh.points)
                    bounds = [
                        points[:,0].min(), points[:,0].max(),
                        points[:,1].min(), points[:,1].max(),
                        points[:,2].min(), points[:,2].max()
                    ]
                    
                    # Create outline source
                    outline = vtk.vtkOutlineSource()
                    outline.SetBounds(bounds)
                    outline.Update()
                    
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(outline.GetOutputPort())
                    
                    self.parent.bounding_box_actor = vtk.vtkActor()
                    self.parent.bounding_box_actor.SetMapper(mapper)
                    self.parent.bounding_box_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                    self.parent.bounding_box_actor.GetProperty().SetLineWidth(2.0)
                    
                    self.parent.renderer.AddActor(self.parent.bounding_box_actor)
                except Exception as e:
                    print(f"Error creating bounding box: {str(e)}")
                    if hasattr(self.parent, 'message_text'):
                        self.parent.message_text.append(f"Error creating bounding box: {str(e)}")
                    return
        
        # Set visibility based on checkbox state
        if hasattr(self.parent, 'bounding_box_actor'):
            self.parent.bounding_box_actor.SetVisibility(show)
            self.parent.vtkWidget.GetRenderWindow().Render()

    def update_visualization(self):
        """Update all visualization elements based on current settings"""
        # Check if geometry is "None" - special case to hide everything
        if self.geom_combo.currentText() == "None":
            # Hide all geometry and results
            if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                self.parent.stl_actor.SetVisibility(False)
            if hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor:
                self.parent.mesh_actor.SetVisibility(False)
            if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
                self.parent.results_actor.SetVisibility(False)
            if hasattr(self.parent, 'results_mesh_actor') and self.parent.results_mesh_actor:
                self.parent.results_mesh_actor.SetVisibility(False)
            if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar:
                self.parent.scalar_bar.SetVisibility(False)
                    
            # Add message to log
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append("Hiding all geometry and results")
                    
            # Skip the rest of the visualization updates
            self.parent.vtkWidget.GetRenderWindow().Render()
            return
                
        # Check if mesh should be shown (based on geometry dropdown)
        show_mesh = self.geom_combo.currentText() == "Mesh"
        self.toggle_mesh(show_mesh)
            
        # Apply field visualization if selected
        selected_field = self.field_combo.currentText()
        if selected_field != "None":
            self.apply_field_visualization(selected_field)
        else:
            # No field selected, hide result actors
            self.toggle_results_visibility(False)
            
        # Update all toggleable features
        feature_list = [
            'bounding_box', 'triangles', 'text', 'scale_deformation',
            'transparent', 'axis', 'structural_loads', 'thermal_loads',
            'topopt_constraints', 'non_design'
        ]
            
        for feature in feature_list:
            self.toggle_feature(feature)
                
        # Update cutting planes
        self.update_cutting_planes()
            
        # Render the changes
        self.parent.vtkWidget.GetRenderWindow().Render()
        
    def apply_field_visualization(self, field_name):
        """Apply a specific field visualization by calling parent visualization method"""
        try:
            # Check if we're set to "None" geometry - if so, don't show anything
            if self.geom_combo.currentText() == "None":
                return
            
            # Get the display mode (geometry or mesh)
            display_mode = self.display_mode_combo.currentText()
                
            # Call the parent's existing visualization method based on field type
            if field_name == "Displacement" and hasattr(self.parent, 'visualize_displacement'):
                self.parent.visualize_displacement(display_mode=display_mode)
                
            elif field_name == "Von Mises Stress" and hasattr(self.parent, 'visualize_stress'):
                self.parent.visualize_stress(display_mode=display_mode)
                
            elif field_name == "Temperature" and hasattr(self.parent, 'visualize_temperature'):
                self.parent.visualize_temperature(display_mode=display_mode)
            else:
                # Fallback if the parent doesn't have specific visualization methods
                if field_name == "Displacement" and hasattr(self.parent, 'analysis_results'):
                    if hasattr(self.parent, 'apply_field_visualization'):
                        self.parent.apply_field_visualization(field_name, display_mode)
                    elif hasattr(self.parent, 'update_result_display'):
                        self.parent.update_result_display(field_name)
                    else:
                        if hasattr(self.parent, 'message_text'):
                            self.parent.message_text.append(f"No visualization method available for {field_name}")
                
            # Apply scale deformation if needed
            if field_name == "Displacement" and self.scale_deformation.isChecked():
                if hasattr(self.parent, 'apply_displacement_scaling'):
                    self.parent.apply_displacement_scaling()
                
        except Exception as e:
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"Error applying {field_name} visualization: {str(e)}")

    def animate_view(self):
        """Animate the view for 3 cycles"""
        # Call the parent's animation method if it exists
        if hasattr(self.parent, 'animate_results'):
            self.parent.animate_results(cycles=3)
        else:
            # Fallback animation implementation
            if hasattr(self.parent, 'analysis_results') and self.parent.analysis_results and self.scale_deformation.isChecked():
                if hasattr(self.parent, 'message_text'):
                    self.parent.message_text.append("Starting animation...")
                
                # Get displacement data
                u = self.parent.analysis_results['displacements']
                delta = self.parent.analysis_results.get('delta')
                
                if delta is None and 'displacements' in self.parent.analysis_results:
                    # Calculate delta (displacement magnitude) if not provided
                    u = self.parent.analysis_results['displacements']
                    delta = []
                    
                    for i in range(len(u) // 3):
                        dx = u[3*i]
                        dy = u[3*i + 1]
                        dz = u[3*i + 2]
                        delta.append(math.sqrt(dx*dx + dy*dy + dz*dz))
                
                # Basic animation parameters
                cycles = 3
                frames_per_cycle = 20
                total_frames = cycles * frames_per_cycle
                
                # Calculate scaling factor based on model size
                if hasattr(self.parent, 'analysis_mesher') and self.parent.analysis_mesher:
                    model_size = np.max(self.parent.analysis_mesher.node_xyz.max(axis=0) - 
                                    self.parent.analysis_mesher.node_xyz.min(axis=0))
                    max_disp = np.max(delta) if hasattr(delta, '__iter__') else delta
                    base_scale = 0.1 * model_size / max_disp if max_disp > 0 else 1.0
                    
                    # Create a timer to handle animation frames
                    self.animation_frame = 0
                    self.animation_timer = QtCore.QTimer()
                    self.animation_timer.timeout.connect(lambda: self.update_animation_frame(
                        total_frames, frames_per_cycle, base_scale, u
                    ))
                    self.animation_timer.start(50)  # Update every 50ms
                else:
                    if hasattr(self.parent, 'message_text'):
                        self.parent.message_text.append("Cannot animate: missing mesh data")
            else:
                if hasattr(self.parent, 'message_text'):
                    self.parent.message_text.append("Animation requires analysis results and scale deformation enabled")

    def update_animation_frame(self, total_frames, frames_per_cycle, base_scale, u):
        """Update a single animation frame"""
        try:
            if not hasattr(self.parent, 'results_actor') or not self.parent.results_actor:
                self.animation_timer.stop()
                return
                
            # Calculate animation progress and scale factor
            self.animation_frame += 1
            cycle_position = (self.animation_frame % frames_per_cycle) / frames_per_cycle
            scale_factor = base_scale * math.sin(cycle_position * 2 * math.pi)
            
            # Update points for deformed mesh
            points = self.parent.results_actor.GetMapper().GetInput().GetPoints()
            
            for i in range(self.parent.analysis_mesher.num_nodes):
                original_pos = self.parent.analysis_mesher.node_xyz[i]
                dx = u[3*i] * scale_factor
                dy = u[3*i + 1] * scale_factor
                dz = u[3*i + 2] * scale_factor
                points.SetPoint(i, 
                    original_pos[0] + dx,
                    original_pos[1] + dy,
                    original_pos[2] + dz
                )
            
            # Update visualization
            self.parent.results_actor.GetMapper().GetInput().Modified()
            self.parent.vtkWidget.GetRenderWindow().Render()
            
            # Stop after all frames
            if self.animation_frame >= total_frames:
                self.animation_timer.stop()
                if hasattr(self.parent, 'message_text'):
                    self.parent.message_text.append("Animation completed")
                
                # Reset to static scaled deformation if scale_deformation is checked
                if self.scale_deformation.isChecked() and hasattr(self.parent, 'apply_displacement_scaling'):
                    self.parent.apply_displacement_scaling()
        except Exception as e:
            self.animation_timer.stop()
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append(f"Error in animation: {str(e)}")

    def hide_selected_part(self):
        """Hide the currently selected part"""
        if hasattr(self.parent, 'hide_selected_part'):
            # Call the parent's method if it exists
            self.parent.hide_selected_part()
        else:
            # Fallback implementation
            if hasattr(self.parent, 'stl_geom') and self.parent.stl_geom:
                selected_indices = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
                
                if selected_indices:
                    # Store hidden indices if not already tracking them
                    if not hasattr(self.parent, 'hidden_indices'):
                        self.parent.hidden_indices = set()
                        
                    # Add current selection to hidden set
                    self.parent.hidden_indices.update(selected_indices)
                    
                    # Clear selection
                    self.parent.stl_geom.tri_highlight = [False] * len(self.parent.stl_geom.tri_highlight)
                    if hasattr(self.parent, 'update_highlights'):
                        self.parent.update_highlights()
                    
                    # Update visibility of STL model
                    self.update_stl_visibility()
                    
                    if hasattr(self.parent, 'message_text'):
                        self.parent.message_text.append(f"Hidden {len(selected_indices)} triangles")
                else:
                    if hasattr(self.parent, 'message_text'):
                        self.parent.message_text.append("No selection to hide")

    def update_stl_visibility(self):
        """Update STL visibility based on hidden indices"""
        if hasattr(self.parent, 'update_stl_visibility'):
            # Call the parent's method if it exists
            self.parent.update_stl_visibility()
        else:
            # Fallback implementation
            if hasattr(self.parent, 'stl_geom') and hasattr(self.parent, 'hidden_indices'):
                # Create a visibility array for triangles
                visibility = vtk.vtkUnsignedCharArray()
                visibility.SetNumberOfComponents(1)
                visibility.SetName("Visibility")
                
                # Set visibility for each triangle
                for i in range(self.parent.stl_geom.stl_n_triangles):
                    if i in self.parent.hidden_indices:
                        visibility.InsertNextValue(0)  # Hidden
                    else:
                        visibility.InsertNextValue(1)  # Visible
                
                # Apply visibility to cell data
                poly_data = self.parent.stl_actor.GetMapper().GetInput()
                poly_data.GetCellData().AddArray(visibility)
                poly_data.GetCellData().SetScalars(visibility)
                
                # Update visualization
                self.parent.stl_actor.GetMapper().Update()
                self.parent.vtkWidget.GetRenderWindow().Render()

    def save_image(self):
        """Save current view as an image file"""
        if hasattr(self.parent, 'save_image'):
            # Call the parent's method if it exists
            self.parent.save_image()
        else:
            # Fallback implementation
            options = QtWidgets.QFileDialog.Options()
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
                options=options
            )
            
            if file_path:
                # Make sure file has proper extension
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path += '.png'
                    
                try:
                    # Create window to image filter
                    window_to_image = vtk.vtkWindowToImageFilter()
                    window_to_image.SetInput(self.parent.vtkWidget.GetRenderWindow())
                    window_to_image.SetInputBufferTypeToRGBA()
                    window_to_image.ReadFrontBufferOff()
                    window_to_image.Update()
                    
                    # Determine writer type based on extension
                    if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                        writer = vtk.vtkJPEGWriter()
                    else:  # Default to PNG
                        writer = vtk.vtkPNGWriter()
                        
                    writer.SetFileName(file_path)
                    writer.SetInputConnection(window_to_image.GetOutputPort())
                    writer.Write()
                    
                    if hasattr(self.parent, 'message_text'):
                        self.parent.message_text.append(f"Image saved to {file_path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def reset_view(self):
        """Reset camera and view settings"""
        if hasattr(self.parent, 'reset_view'):
            # Call the parent's method if it exists
            self.parent.reset_view()
            
            # Update the controls to match the new state
            self.blockAllSignals(True)
            self.load_current_settings(apply_immediately=False)
            self.blockAllSignals(False)
        else:
            # Fallback implementation
            # Block signals during reset
            self.blockAllSignals(True)
            
            # Reset geometry to initial design
            self.geom_combo.setCurrentText("InitialDesign")
            
            # Reset field to None
            self.field_combo.setCurrentIndex(0)
            
            # Reset camera position
            self.parent.renderer.ResetCamera()
            
            # Reset cutting planes
            self.x_cutting_spin.setValue(0)
            self.y_cutting_spin.setValue(0)
            self.z_cutting_spin.setValue(0)
            
            # Reset hidden parts if any
            if hasattr(self.parent, 'hidden_indices'):
                self.parent.hidden_indices = set()
                self.update_stl_visibility()
            
            # Hide any result actors and show original geometry
            if hasattr(self.parent, 'results_actor') and self.parent.results_actor:
                self.parent.results_actor.SetVisibility(False)
                
            if hasattr(self.parent, 'results_mesh_actor') and self.parent.results_mesh_actor:
                self.parent.results_mesh_actor.SetVisibility(False)
                
            if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor:
                self.parent.stl_actor.SetVisibility(True)
                self.parent.stl_actor.GetProperty().SetOpacity(1.0)
                
            # Hide the scalar bar if visible
            if hasattr(self.parent, 'scalar_bar'):
                self.parent.scalar_bar.SetVisibility(False)
                
            # Reset display settings
            if hasattr(self.parent, 'display_settings'):
                self.parent.display_settings = {
                    'geometry': 'InitialDesign',
                    'field': 'None',
                    'display_mode': 'Geometry',
                    'x_cutting': 0,
                    'y_cutting': 0,
                    'z_cutting': 0,
                    'eigen_number': 1,
                    'show_bounding_box': False,
                    'show_results': False,
                    'show_triangles': False,
                    'show_text': True,
                    'scale_deformation': True,
                    'show_transparent': False,
                    'show_axis': True,
                    'show_structural_loads': True,
                    'show_thermal_loads': False,
                    'show_topopt_constraints': True,
                    'show_non_design': True,
                }
                
            # Update checkboxes to match reset settings
            self.show_bounding_box.setChecked(False)
            self.show_triangles.setChecked(False)
            self.show_text.setChecked(True)
            self.scale_deformation.setChecked(True)
            self.show_transparent.setChecked(False)
            self.show_axis.setChecked(True)
            self.show_structural_loads.setChecked(True)
            self.show_thermal_loads.setChecked(False)
            self.show_topopt_constraints.setChecked(True)
            self.show_non_design.setChecked(True)
            
            # Restore signals
            self.blockAllSignals(False)
                
            # Reset view 
            self.parent.vtkWidget.GetRenderWindow().Render()
            if hasattr(self.parent, 'message_text'):
                self.parent.message_text.append("View reset to default")
        

    
#---------------------------------------------------------------------------------
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
        
        # Get unit strings based on current settings
        stress_unit = self.parent.settings.get_stress_unit_string()
        density_unit = f"kg/m³"  # Density typically stays in SI units
        
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
        form_layout.addRow(f"Young's Modulus ({stress_unit}):", self.young_input)
        form_layout.addRow("Poisson ratio ( - ):", self.poisson_input)
        form_layout.addRow(f"Yield Strength ({stress_unit}):", self.yield_input)
        form_layout.addRow(f"Density ({density_unit}):", self.density_input)
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
        
        # Material data dictionary - store values in base units (MKS/Pascal)
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
        
        # Initialize values for current material with correct unit conversions
        self.update_material_for_current_units("AlloySteel")

    def on_material_changed(self, material_name):
        if material_name != "Custom":
            self.update_material_for_current_units(material_name)
        else:
            self.clear_inputs()
    
    def update_material_for_current_units(self, material_name):
        """Update material properties with values converted to current unit system"""
        if material_name not in self.materials_data:
            return
            
        material = self.materials_data[material_name]
        
        # Convert material properties from base units (MKS) to current unit system
        current_unit_system = self.parent.settings.unit_system
        
        # Convert Young's modulus and yield strength to display units
        young_value = float(material["young"])
        yield_value = float(material["yield"])
        
        if current_unit_system != "MKS":
            young_value = self.parent.settings.convert_stress(young_value, from_system="MKS")
            yield_value = self.parent.settings.convert_stress(yield_value, from_system="MKS")
        
        # Update the input fields with converted values
        self.young_input.setText(f"{young_value:.2e}")
        self.poisson_input.setText(material["poisson"])
        self.yield_input.setText(f"{yield_value:.2e}")
        self.density_input.setText(material["density"])
        self.thermal_cond_input.setText(material["thermal_cond"])
        self.thermal_exp_input.setText(material["thermal_exp"])
        self.spec_heat_input.setText(material["spec_heat"])
        self.price_input.setText(material["price"])

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
            current_unit_system = self.parent.settings.unit_system
            
            # Parse values from input fields
            young_input_value = float(self.young_input.text())
            yield_input_value = float(self.yield_input.text())
            
            # Convert stress values back to MKS (Pascal) for internal storage
            if current_unit_system != "MKS":
                young_modulus = self.parent.settings.convert_stress(young_input_value, 
                                                               from_system=current_unit_system, 
                                                               to_system="MKS")
                yield_strength = self.parent.settings.convert_stress(yield_input_value, 
                                                                from_system=current_unit_system, 
                                                                to_system="MKS")
            else:
                young_modulus = young_input_value
                yield_strength = yield_input_value
            
            # Create material data dictionary with values in base units
            material_data = {
                "name": self.material_combo.currentText(),
                "young_modulus": young_modulus,
                "poisson_ratio": float(self.poisson_input.text()),
                "yield_strength": yield_strength,
                "density": float(self.density_input.text()),
                "thermal_conductivity": float(self.thermal_cond_input.text()),
                "thermal_expansion": float(self.thermal_exp_input.text()),
                "specific_heat": float(self.spec_heat_input.text()),
                "price": float(self.price_input.text()),
                "do_not_optimize": self.optimize_check.isChecked()
            }
            
            # Also store display values for UI use
            material_data["display_young_modulus"] = young_input_value
            material_data["display_yield_strength"] = yield_input_value
            material_data["display_unit_system"] = current_unit_system
            
            # Validate inputs
            if not (0 < material_data["poisson_ratio"] < 0.5):
                raise ValueError("Poisson's ratio must be between 0 and 0.5")
            
            # Store material data in parent
            self.parent.material_data = material_data
            
            # Add appropriate unit to the message
            stress_unit = self.parent.settings.get_stress_unit_string()
            self.parent.message_text.append(
                f"Material applied: {material_data['name']} " + 
                f"(E={young_input_value:.2e} {stress_unit}, " +
                f"σ_y={yield_input_value:.2e} {stress_unit})"
            )
            
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
            
            # Update Material button icon to check
            self.parent.update_button_icon("Material", "check")
            
            # Update LivVar state
            self.parent.update_LivVar('material_defined', True)
            
            self.close()
            
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(e))

#---------------------------------------------------------------------------------
class StructuralLoadsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowModality(Qt.NonModal)  # Allow face selection while dialog is open
        self.setWindowTitle("Structural Loads")
        self.parent = parent
        
        # Create main layout
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
        self.selection_combo.addItems(["Coarse Cylinder"])
        selection_layout.addWidget(selection_label)
        selection_layout.addWidget(self.selection_combo)
        layout.addLayout(selection_layout)
        
        # Load Type dropdown
        load_type_layout = QtWidgets.QHBoxLayout()
        load_type_label = QtWidgets.QLabel("Load Type")
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems(["Force", "Torque", "Fixed XYZ", "Fixed X", "Fixed Y", "Fixed Z"])
        load_type_layout.addWidget(load_type_label)
        load_type_layout.addWidget(self.load_type)
        layout.addLayout(load_type_layout)

        # Get force unit string
        force_unit = self.parent.settings.get_force_unit_string()
        
        # Force components
        self.force_group = QtWidgets.QGroupBox()
        force_layout = QtWidgets.QVBoxLayout(self.force_group)
        
        # X Force
        x_force_layout = QtWidgets.QHBoxLayout()
        x_force_label = QtWidgets.QLabel(f"X-Force ({force_unit})")
        self.x_force_spin = QtWidgets.QDoubleSpinBox()
        self.x_force_spin.setRange(-1e6, 1e6)
        self.x_force_spin.setDecimals(1)
        x_force_layout.addWidget(x_force_label)
        x_force_layout.addWidget(self.x_force_spin)
        force_layout.addLayout(x_force_layout)
        
        # Y Force
        y_force_layout = QtWidgets.QHBoxLayout()
        y_force_label = QtWidgets.QLabel(f"Y-Force ({force_unit})")
        self.y_force_spin = QtWidgets.QDoubleSpinBox()
        self.y_force_spin.setRange(-1e6, 1e6)
        self.y_force_spin.setDecimals(1)
        y_force_layout.addWidget(y_force_label)
        y_force_layout.addWidget(self.y_force_spin)
        force_layout.addLayout(y_force_layout)
        
        # Z Force
        z_force_layout = QtWidgets.QHBoxLayout()
        z_force_label = QtWidgets.QLabel(f"Z-Force ({force_unit})")
        self.z_force_spin = QtWidgets.QDoubleSpinBox()
        self.z_force_spin.setRange(-1e6, 1e6)
        self.z_force_spin.setDecimals(1)
        z_force_layout.addWidget(z_force_label)
        z_force_layout.addWidget(self.z_force_spin)
        force_layout.addLayout(z_force_layout)
        
        layout.addWidget(self.force_group)

        # Torque components
        self.torque_group = QtWidgets.QGroupBox()
        torque_layout = QtWidgets.QVBoxLayout(self.torque_group)
        
        # Torque magnitude
        torque_magnitude_layout = QtWidgets.QHBoxLayout()
        torque_magnitude_label = QtWidgets.QLabel(f"Torque Magnitude ({force_unit}·m)")
        self.torque_magnitude_spin = QtWidgets.QDoubleSpinBox()
        self.torque_magnitude_spin.setRange(-1e6, 1e6)
        self.torque_magnitude_spin.setDecimals(1)
        self.torque_magnitude_spin.setValue(100.0)  # Default value
        torque_magnitude_layout.addWidget(torque_magnitude_label)
        torque_magnitude_layout.addWidget(self.torque_magnitude_spin)
        torque_layout.addLayout(torque_magnitude_layout)
        
        # Label for axis information
        axis_info_label = QtWidgets.QLabel("Torque will be applied around the central axis of the selected cylindrical surface")
        axis_info_label.setWordWrap(True)
        torque_layout.addWidget(axis_info_label)
        
        layout.addWidget(self.torque_group)
        self.torque_group.hide()  # Initially hide the torque controls
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_load)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # Connect load type change
        self.load_type.currentTextChanged.connect(self.on_load_type_changed)
        
    def on_load_type_changed(self, load_type):
        # Show/hide force inputs based on load type
        show_force = load_type == "Force"
        show_torque = load_type == "Torque"
        self.force_group.setVisible(show_force)
        self.torque_group.setVisible(show_torque)
        self.adjustSize()

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
        
    def apply_load(self):
        load_type = self.load_type.currentText()
        
        if load_type == "Force":
            self.apply_force()
        elif load_type == "Torque":
            self.apply_torque()
        elif load_type == "Fixed XYZ":
            self.apply_fixed_constraint()
        elif load_type == "Fixed X":
            self.apply_fixed_constraint_x()
        elif load_type == "Fixed Y":
            self.apply_fixed_constraint_y()
        elif load_type == "Fixed Z":
            self.apply_fixed_constraint_z()

    def apply_torque(self):
        """Apply torque to selected cylindrical or planar surfaces with proper unit conversion"""
        if not self.parent.stl_geom:
            return
                
        try:
            # Get torque magnitude from spinbox in current unit system
            torque_magnitude = self.torque_magnitude_spin.value()
            
            # Convert to base units (Newton-meters) if needed
            if self.parent.settings.unit_system != "MKS":
                torque_magnitude = self.parent.settings.convert_force(torque_magnitude, 
                                                        from_system=self.parent.settings.unit_system,
                                                        to_system="MKS") * 1.0  # Multiply by 1.0 meter for torque
            
            if torque_magnitude == 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Torque magnitude cannot be zero")
                return
                
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            if not selected_faces:
                QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
                return
            
            
            # Call the assign_highlighted_triangles_to_group method
            surface_type, average_normal, area, cylinder_axis, axis_point, cylinder_radius = self.parent.stl_geom.assign_highlighted_triangles_to_group(
                group=1, stl_verbose=False
            )
            
            # Check if surface is planar or cylindrical
            if surface_type not in ["PLANAR", "CYLINDER"]:
                QtWidgets.QMessageBox.warning(self, "Error", "Currently, torque can only be applied on planes or cylinders")
                return
            
            # Calculate the direction vector based on surface type
            if surface_type == "PLANAR":
                # For planar surfaces, use the average normal
                direction = average_normal
                # Normalize direction
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    QtWidgets.QMessageBox.warning(self, "Error", "Invalid normal vector")
                    return
                direction = [d / norm for d in direction]
                
            elif surface_type == "CYLINDER":
                # For cylindrical surfaces, use the cylinder axis
                direction = cylinder_axis
                # Normalize direction
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    QtWidgets.QMessageBox.warning(self, "Error", "Invalid cylinder axis")
                    return
                direction = [d / norm for d in direction]
            
            # Store selected face data and torque info for later node selection
            if not hasattr(self.parent, 'torque_faces_groups'):
                self.parent.torque_faces_groups = []
                self.parent.torque_values = []
                self.parent.torque_axis_points = []
                
            # Add this group of faces and its torque information
            self.parent.torque_faces_groups.append(selected_faces)
            
            # Store the direction vector multiplied by torque magnitude
            torque_vector = [torque_magnitude * d for d in direction]
            self.parent.torque_values.append(torque_vector)
            self.parent.torque_axis_points.append(axis_point)

            # Get the appropriate force unit string for display
            force_unit = self.parent.settings.get_force_unit_string()
            
            # Display magnitude in current unit system for messages
            display_magnitude = torque_magnitude
            if self.parent.settings.unit_system != "MKS":
                display_magnitude = self.parent.settings.convert_force(torque_magnitude, 
                                                                from_system="MKS", 
                                                                to_system=self.parent.settings.unit_system)
            
            # Visualization code
            points = np.array(self.parent.stl_geom.mesh.points)
            bbox = [points[:,0].min(), points[:,0].max(),
                    points[:,1].min(), points[:,1].max(),
                    points[:,2].min(), points[:,2].max()]
            geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
            scale_factor = 0.15 * geom_size 
            
            # Create curved arrow for torque visualization
            if not hasattr(self.parent, 'torque_actors'):
                self.parent.torque_actors = []
                
            # Create a circle with an arrow tip
            circle_resolution = 36
            angle_increment = 270.0 / (circle_resolution - 1)  # 270 degree arc
            
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            # # Create a 270-degree arc
            # radius = 0.24 * scale_factor  # Radius of the circle

            # Create a 270-degree arc
            if surface_type == "CYLINDER":
                # For cylindrical surfaces
                radius = 0.8 * cylinder_radius  # 80% of the cylinder radius
            else:
                # For planar surfaces
                model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
                radius = 0.15 * model_size  
            
            # Make sure radius is not too small or too large
            min_radius = 0.05 * scale_factor
            max_radius = 0.24 * scale_factor
            radius = max(min_radius, min(radius, max_radius))
            
            # Start with the center point
            center_id = points.InsertNextPoint(axis_point)
            
            # Add points for the arc
            for i in range(circle_resolution):
                angle = math.radians(i * angle_increment)
                # Calculate point on circle 
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0.0
                points.InsertNextPoint([x, y, z])
            
            # Create a circle actor
            circle_poly = vtk.vtkPolyData()
            circle_poly.SetPoints(points)
            
            # Create lines for the arc
            for i in range(1, circle_resolution):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1 if i < circle_resolution - 1 else i)
                lines.InsertNextCell(line)
                
            circle_poly.SetLines(lines)
            
            # Create arrow tip at end of arc
            arrow_tip = vtk.vtkConeSource()
            arrow_tip.SetHeight(0.2 * scale_factor)
            arrow_tip.SetRadius(0.06 * scale_factor)
            arrow_tip.SetResolution(10)
            
            # align it with the torque axis and position it
            transform = vtk.vtkTransform()
            transform.Identity()
            
            # Calculate the perpendicular vectors to the torque direction
            perpendicular1 = [0, 0, 0]
            perpendicular2 = [0, 0, 0]
            
            # Find perpendicular vectors using cross product
            if abs(direction[2]) < 0.707:
                perpendicular1[0] = -direction[1]
                perpendicular1[1] = direction[0]
                perpendicular1[2] = 0
            else:
                perpendicular1[0] = 0
                perpendicular1[1] = -direction[2]
                perpendicular1[2] = direction[1]
                
            # Normalize perpendicular_1
            norm = math.sqrt(sum(v*v for v in perpendicular1))
            if norm > 1e-6:
                perpendicular1 = [v/norm for v in perpendicular1]
                
            # Calculate perpendicular_2 using cross product of direction and perpendicular1
            perpendicular2[0] = direction[1]*perpendicular1[2] - direction[2]*perpendicular1[1]
            perpendicular2[1] = direction[2]*perpendicular1[0] - direction[0]*perpendicular1[2]
            perpendicular2[2] = direction[0]*perpendicular1[1] - direction[1]*perpendicular1[0]
            
            # Create transformation matrix
            matrix = vtk.vtkMatrix4x4()
            for i in range(3):
                matrix.SetElement(i, 0, perpendicular1[i])
                matrix.SetElement(i, 1, perpendicular2[i])
                matrix.SetElement(i, 2, direction[i])
                matrix.SetElement(i, 3, axis_point[i])
            matrix.SetElement(3, 3, 1)
            
            transform.SetMatrix(matrix)
            
            # Create mapper and actor for the circle
            circle_mapper = vtk.vtkPolyDataMapper()
            circle_mapper.SetInputData(circle_poly)
            
            circle_actor = vtk.vtkActor()
            circle_actor.SetMapper(circle_mapper)
            circle_actor.SetUserTransform(transform)
            circle_actor.GetProperty().SetColor(0, 0.8, 0)  # Green color for torque
            circle_actor.GetProperty().SetLineWidth(3)
            
            # Add the circle actor to the renderer
            self.parent.renderer.AddActor(circle_actor)
            self.parent.torque_actors.append(circle_actor)
            
            # Position and orient the arrow tip at the end of the arc
            arrow_tip_transform = vtk.vtkTransform()
            arrow_tip_transform.Identity()
            
            # Last point on the arc
            last_angle = math.radians(270.0)
            tip_x = radius * math.cos(last_angle)
            tip_y = radius * math.sin(last_angle)
            
            # Apply the main transform
            arrow_tip_transform.Concatenate(transform)
            
            # Position at end of arc
            arrow_tip_transform.Translate(tip_x, tip_y, 0)
            
            # Orient in the tangent direction
            arrow_tip_transform.RotateZ(180)  # Align with tangent direction
            
            # Create mapper and actor for arrow tip
            arrow_tip_mapper = vtk.vtkPolyDataMapper()
            arrow_tip_mapper.SetInputConnection(arrow_tip.GetOutputPort())
            
            arrow_tip_actor = vtk.vtkActor()
            arrow_tip_actor.SetMapper(arrow_tip_mapper)
            arrow_tip_actor.SetUserTransform(arrow_tip_transform)
            arrow_tip_actor.GetProperty().SetColor(0, 0.8, 0)  # Green color matching the arc
            
            # Add the arrow tip actor to the renderer
            self.parent.renderer.AddActor(arrow_tip_actor)
            self.parent.torque_actors.append(arrow_tip_actor)
            
            # Create text label for the torque value
            text_offset = 0.2 * geom_size 
            text_pos = [
                axis_point[0] + text_offset * perpendicular1[0],
                axis_point[1] + text_offset * perpendicular1[1],
                axis_point[2] + text_offset * perpendicular1[2]
            ]
            
            # Create a vtkCaptionActor2D for the torque value text
            caption_actor = vtk.vtkCaptionActor2D()
            
            # Format the torque value with proper units
            torque_text = f"{display_magnitude:.2f} {force_unit}·m"
            caption_actor.SetCaption(torque_text)
            caption_actor.SetAttachmentPoint(text_pos)
            
            # Customize text appearance
            caption_actor.BorderOff()
            caption_actor.LeaderOff()  # No leader line
            
            # Set text properties
            text_prop = caption_actor.GetCaptionTextProperty()
            text_prop.SetColor(0.0, 0.8, 0.0)  # Green text to match torque arrow
            text_prop.SetFontSize(5)
            
            # Scale the text appropriately
            caption_actor.SetWidth(0.15)
            caption_actor.SetHeight(0.05)
            text_prop.SetBold(True)
            text_prop.SetShadow(True)
            
            # Add to renderer and store
            self.parent.renderer.AddActor(caption_actor)
            self.parent.torque_actors.append(caption_actor)
            
            # Release faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"Applied torque of {display_magnitude:.2f} {force_unit}·m to {len(selected_faces)} triangles ({surface_type.lower()} surface)")
            self.parent.update_button_icon("Structural Loads", "check")
            
            # Update LivVar state
            self.parent.update_LivVar('structural_loads.torque_applied', True)
            self.parent.update_LivVar('structural_loads.applied', True)
            self.close()
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
            
    def apply_force(self):
        """Apply force to selected nodes with proper unit conversion"""
        if not self.parent.stl_geom:
            return
                
        try:
            # Get force components from spinboxes in current unit system
            force_x = self.x_force_spin.value()
            force_y = self.y_force_spin.value()
            force_z = self.z_force_spin.value()
            
            # Convert to base units (Newtons) if needed
            if self.parent.settings.unit_system != "MKS":
                force_x = self.parent.settings.convert_force(force_x, 
                                                         from_system=self.parent.settings.unit_system,
                                                         to_system="MKS")
                force_y = self.parent.settings.convert_force(force_y, 
                                                         from_system=self.parent.settings.unit_system,
                                                         to_system="MKS")
                force_z = self.parent.settings.convert_force(force_z, 
                                                         from_system=self.parent.settings.unit_system,
                                                         to_system="MKS")
            
            # Calculate magnitude in converted units
            magnitude = math.sqrt(force_x**2 + force_y**2 + force_z**2)
            if magnitude == 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Force magnitude cannot be zero")
                return
                
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            if not selected_faces:
                QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
                return
            
            # Store selected face data for later node selection
            if not hasattr(self.parent, 'load_faces_groups'):
                self.parent.load_faces_groups = []
                self.parent.load_forces = []
                
            # Add this group of faces and its force
            self.parent.load_faces_groups.append(selected_faces)
            self.parent.load_forces.append([force_x, force_y, force_z])

            # Get the appropriate force unit string for display
            force_unit = self.parent.settings.get_force_unit_string()
            
            # Display magnitude in current unit system for messages
            display_magnitude = magnitude
            if self.parent.settings.unit_system != "MKS":
                display_magnitude = self.parent.settings.convert_force(magnitude, 
                                                                  from_system="MKS", 
                                                                  to_system=self.parent.settings.unit_system)
            
            # Visualization code
            MAX_MARKERS = 5
            THRESHOLD = 25
            
            if len(selected_faces) > THRESHOLD:
                step = len(selected_faces) // MAX_MARKERS
                display_indices = range(0, len(selected_faces), step)[:MAX_MARKERS]
                display_faces = [selected_faces[i] for i in display_indices]
            else:
                display_faces = selected_faces
                    
            # Get bounding box for scaling
            points = np.array(self.parent.stl_geom.mesh.points)
            bbox = [points[:,0].min(), points[:,0].max(),
                    points[:,1].min(), points[:,1].max(),
                    points[:,2].min(), points[:,2].max()]
            geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
            scale_factor = 0.10 * geom_size
            
            # Normalize direction for text placement
            dx, dy, dz = force_x/magnitude, force_y/magnitude, force_z/magnitude
                
            # Create a single text actor to display the force value
            # Calculate a good position for the text (near the first arrow, but offset)
            if display_faces:
                first_triangle = display_faces[0]
                
                # Create arrow for visualization
                arrow = vtk.vtkArrowSource()
                arrow.SetTipLength(0.3)
                arrow.SetTipRadius(0.1)
                arrow.SetShaftRadius(0.03)
                
                # Set up transform
                transform = vtk.vtkTransform()
                transform.Translate(first_triangle['center'])
                
                # Calculate rotation angles for direction
                if abs(dx) > 0 or abs(dy) > 0:
                    angle_z = math.degrees(math.atan2(dy, dx))
                else:
                    angle_z = 0
                    
                angle_y = -math.degrees(math.asin(dz))
                
                transform.RotateZ(angle_z)
                transform.RotateY(angle_y)
                transform.Scale(scale_factor, scale_factor, scale_factor)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(arrow.GetOutputPort())
                
                arrow_actor = vtk.vtkActor()
                arrow_actor.SetMapper(mapper)
                arrow_actor.SetUserTransform(transform)
                arrow_actor.GetProperty().SetColor(1, 0, 0)  # Red for force
                
                # Add to renderer and store
                self.parent.renderer.AddActor(arrow_actor)
                self.parent.force_actors.append(arrow_actor)
                
                # Calculate position for text label - place it near the arrow
                text_offset = 0.12 * geom_size  # Offset from the center
                
                # Calculate the text position to be offset perpendicular to the force direction
                # This ensures the text doesn't overlap with the arrow
                text_pos = [
                    first_triangle['center'][0] + text_offset * (-dy),  # Perpendicular to direction
                    first_triangle['center'][1] + text_offset * (dx),   # Perpendicular to direction
                    first_triangle['center'][2] + text_offset * 0.2    # Small Z offset
                ]
                
                # Create a vtkCaptionActor2D for the force value text
                caption_actor = vtk.vtkCaptionActor2D()
                
                # Format the force components and magnitude
                force_text = f"{display_magnitude:.2f} {force_unit}"
                # if abs(force_x) > 0.01 or abs(force_y) > 0.01 or abs(force_z) > 0.01:
                #     force_text += f"\n({force_x:.1f}, {force_y:.1f}, {force_z:.1f}) N"
                
                caption_actor.SetCaption(force_text)
                caption_actor.SetAttachmentPoint(text_pos)
                
                # Customize text appearance
                caption_actor.BorderOff()
                caption_actor.LeaderOff()  # No leader line
                
                # Set text properties
                text_prop = caption_actor.GetCaptionTextProperty()
                text_prop.SetColor(1.0, 0.0, 0.0)  # Red text to match force arrows
                text_prop.SetFontSize(5)
                
                # Scale the text appropriately
                caption_actor.SetWidth(0.15)
                caption_actor.SetHeight(0.05)
                text_prop.SetBold(True)
                text_prop.SetShadow(True)
                
                # Add to renderer and store
                self.parent.renderer.AddActor(caption_actor)
                self.parent.force_actors.append(caption_actor)
                
            # Add remaining arrows without text
            for triangle in display_faces[1:]:
                # Create arrow for visualization
                arrow = vtk.vtkArrowSource()
                arrow.SetTipLength(0.3)
                arrow.SetTipRadius(0.1)
                arrow.SetShaftRadius(0.03)
                
                # Set up transform
                transform = vtk.vtkTransform()
                transform.Translate(triangle['center'])
                
                # Calculate direction angles
                if abs(dx) > 0 or abs(dy) > 0:
                    angle_z = math.degrees(math.atan2(dy, dx))
                else:
                    angle_z = 0
                    
                angle_y = -math.degrees(math.asin(dz))
                
                transform.RotateZ(angle_z)
                transform.RotateY(angle_y)
                transform.Scale(scale_factor, scale_factor, scale_factor)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(arrow.GetOutputPort())
                
                arrow_actor = vtk.vtkActor()
                arrow_actor.SetMapper(mapper)
                arrow_actor.SetUserTransform(transform)
                arrow_actor.GetProperty().SetColor(1, 0, 0)  # Red for force
                
                # Add to renderer and store
                self.parent.renderer.AddActor(arrow_actor)
                self.parent.force_actors.append(arrow_actor)
                
            # Release forced faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"Applied force of {display_magnitude:.2f} {force_unit} to {len(selected_faces)} triangles")
            self.parent.update_button_icon("Structural Loads", "check")
            # Update LivVar state
            self.parent.update_LivVar('structural_loads.forces_applied', True)
            self.parent.update_LivVar('structural_loads.applied', True)
            self.close()
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
            
    def apply_fixed_constraint(self):
        if not self.parent.stl_geom:
            return
            
        # Get selected faces
        selected_faces = self.parent.stl_geom.store_selected_triangles()
        if not selected_faces:
            QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
            return

        # Create visualization first
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        for triangle in selected_faces:
            vertices = triangle['vertices']
            point_ids = []
            for v in vertices:
                point_ids.append(points.InsertNextPoint(v))
            tri = vtk.vtkTriangle()
            for i in range(3):
                tri.GetPointIds().SetId(i, point_ids[i])
            cells.InsertNextCell(tri)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        constraint_actor = vtk.vtkActor()
        constraint_actor.SetMapper(mapper)
        constraint_actor.GetProperty().SetColor(0, 0, 0)  # Black color
        
        self.parent.renderer.AddActor(constraint_actor)
        self.parent.constraint_actors.append(constraint_actor)
        
        # Store selected face data for later node selection
        if not hasattr(self.parent, 'fixed_faces'):
            self.parent.fixed_faces = []
        
        self.parent.fixed_faces.extend(selected_faces)
        
        # Add debug message
        self.parent.message_text.append(f"\nStored {len(selected_faces)} faces for fixed constraint")
        
        # Release constrained faces from selection
        for triangle in selected_faces:
            idx = triangle['index']
            self.parent.stl_geom.tri_highlight[idx] = False
        self.parent.update_highlights()
        
        # Update display
        self.parent.vtkWidget.GetRenderWindow().Render()
        self.parent.update_button_icon("Structural Loads", "check")
        # Update LivVar state
        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        self.parent.update_LivVar('structural_loads.applied', True)    

    def apply_fixed_constraint_x(self):
        if self.parent.stl_geom:
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            if not selected_faces:
                QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
                return

            # Create points and cells for the fixed faces
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            for triangle in selected_faces:
                vertices = triangle['vertices']
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(cells)

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            
            constraint_actor = vtk.vtkActor()
            constraint_actor.SetMapper(mapper)
            constraint_actor.GetProperty().SetColor(0, 0, 0)  # Set color to black
            
            # Add to renderer and store in constraint_actors
            self.parent.renderer.AddActor(constraint_actor)
            self.parent.constraint_actors.append(constraint_actor)
            
            # Store selected face data for later node selection
            if not hasattr(self.parent, 'fixed_faces_x'):
                self.parent.fixed_faces_x = []
            
            self.parent.fixed_faces_x.extend(selected_faces)
            
            # Release constrained faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            # Update display
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"\nStored {len(selected_faces)} faces for fixed X constraint")
            self.parent.update_button_icon("Structural Loads", "check")
            # Update LivVar state
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
            self.parent.update_LivVar('structural_loads.applied', True)


    def apply_fixed_constraint_y(self):
        if self.parent.stl_geom:
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            if not selected_faces:
                QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
                return

            # Create points and cells for the fixed faces
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            for triangle in selected_faces:
                vertices = triangle['vertices']
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(cells)

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            
            constraint_actor = vtk.vtkActor()
            constraint_actor.SetMapper(mapper)
            constraint_actor.GetProperty().SetColor(0, 0, 0)  # Set color to black
            
            # Add to renderer and store in constraint_actors
            self.parent.renderer.AddActor(constraint_actor)
            self.parent.constraint_actors.append(constraint_actor)
            
            # Store selected face data for later node selection
            if not hasattr(self.parent, 'fixed_faces_y'):
                self.parent.fixed_faces_y = []
            
            self.parent.fixed_faces_y.extend(selected_faces)
            
            # Release constrained faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            # Update display
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"\nStored {len(selected_faces)} faces for fixed Y constraint")
            self.parent.update_button_icon("Structural Loads", "check")
            # Update LivVar state
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
            self.parent.update_LivVar('structural_loads.applied', True)

    def apply_fixed_constraint_z(self):
        if self.parent.stl_geom:
            selected_faces = self.parent.stl_geom.store_selected_triangles()
            if not selected_faces:
                QtWidgets.QMessageBox.warning(self, "Error", "No faces selected")
                return

            # Create points and cells for the fixed faces
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            for triangle in selected_faces:
                vertices = triangle['vertices']
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(cells)

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            
            constraint_actor = vtk.vtkActor()
            constraint_actor.SetMapper(mapper)
            constraint_actor.GetProperty().SetColor(0, 0, 0)  # Set color to black
            
            # Add to renderer and store in constraint_actors
            self.parent.renderer.AddActor(constraint_actor)
            self.parent.constraint_actors.append(constraint_actor)
            
            # Store selected face data for later node selection
            if not hasattr(self.parent, 'fixed_faces_z'):
                self.parent.fixed_faces_z = []
            
            self.parent.fixed_faces_z.extend(selected_faces)
            
            # Release constrained faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            # Update display
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"\nStored {len(selected_faces)} faces for fixed Z constraint")
            self.parent.update_button_icon("Structural Loads", "check")
            # Update LivVar state
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
            self.parent.update_LivVar('structural_loads.applied', True)

#---------------------------------------------------------------------------------

class AnalysisWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Analysis")
        self.resize(300, 350)
        self.parent = parent
        
        # Create main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Load Set
        load_set_layout = QtWidgets.QHBoxLayout()
        load_set_label = QtWidgets.QLabel("Load Set")
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        load_set_layout.addWidget(load_set_label)
        load_set_layout.addWidget(self.load_set_spin)
        layout.addLayout(load_set_layout)
        
        # Mesh Quality
        mesh_quality_layout = QtWidgets.QHBoxLayout()
        mesh_quality_label = QtWidgets.QLabel("Mesh Quality")
        self.mesh_quality_combo = QtWidgets.QComboBox()
        self.mesh_quality_combo.addItems(["Very Coarse", "Coarse", "Normal", "Fine", "Very Fine"])
        mesh_quality_layout.addWidget(mesh_quality_label)
        mesh_quality_layout.addWidget(self.mesh_quality_combo)
        layout.addLayout(mesh_quality_layout)
        
        # Number of Elements
        elements_layout = QtWidgets.QHBoxLayout()
        elements_label = QtWidgets.QLabel("#Elements")
        self.elements_spin = QtWidgets.QSpinBox()
        self.elements_spin.setRange(1000, 1000000)
        self.elements_spin.setValue(10000)  # Default is Very Coarse
        elements_layout.addWidget(elements_label)
        elements_layout.addWidget(self.elements_spin)
        layout.addLayout(elements_layout)
        
        # Connect mesh quality combo box to update elements count
        self.mesh_quality_combo.currentIndexChanged.connect(self.update_elements_count)\
        
        # Solver Type
        solver_layout = QtWidgets.QHBoxLayout()
        solver_label = QtWidgets.QLabel("Solver Type")
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems(["PARDISO", "DPCG", "CG", "PYAMG", "SPSOLVE"])
        solver_layout.addWidget(solver_label) 
        solver_layout.addWidget(self.solver_combo)
        layout.addLayout(solver_layout)
        
        # Solver Tolerance (New addition for solver options)
        """         tolerance_layout = QtWidgets.QHBoxLayout()
        tolerance_label = QtWidgets.QLabel("Solver Tolerance")
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-12, 1e-1)
        self.tolerance_spin.setValue(1e-8)
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        tolerance_layout.addWidget(tolerance_label)
        tolerance_layout.addWidget(self.tolerance_spin)
        layout.addLayout(tolerance_layout)
        
        # Deflation Type
        deflation_layout = QtWidgets.QHBoxLayout()
        deflation_label = QtWidgets.QLabel("Deflation Type")
        self.deflation_combo = QtWidgets.QComboBox()
        self.deflation_combo.addItems(["Rigid", "Flexible"])
        deflation_layout.addWidget(deflation_label)
        deflation_layout.addWidget(self.deflation_combo)
        layout.addLayout(deflation_layout)
        
        # Number of Deflation Groups
        deflation_groups_layout = QtWidgets.QHBoxLayout()
        deflation_groups_label = QtWidgets.QLabel("#Deflation Groups")
        self.deflation_groups_spin = QtWidgets.QSpinBox()
        self.deflation_groups_spin.setRange(1, 1000)
        self.deflation_groups_spin.setValue(100)
        deflation_groups_layout.addWidget(deflation_groups_label)
        deflation_groups_layout.addWidget(self.deflation_groups_spin)
        layout.addLayout(deflation_groups_layout) """
        
        # Include Thermal Effect
        self.thermal_check = QtWidgets.QCheckBox("Include Thermal Effect")
        layout.addWidget(self.thermal_check)
        
        # Zero-strain Temperature
        temp_layout = QtWidgets.QHBoxLayout()
        temp_label = QtWidgets.QLabel("Zero-strain T(K):")
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0, 1000)
        self.temp_spin.setValue(300)
        self.temp_spin.setDecimals(2)
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_spin)
        layout.addLayout(temp_layout)
        
        # Mesh and Analysis buttons
        self.mesh_button = QtWidgets.QPushButton("Generate Mesh")
        self.mesh_button.clicked.connect(self.generate_mesh)
        layout.addWidget(self.mesh_button)
        
        # Analysis buttons
        self.thermal_button = QtWidgets.QPushButton("Thermal Analysis")
        self.thermal_button.clicked.connect(self.run_thermal_analysis)
        layout.addWidget(self.thermal_button)
        
        self.structural_button = QtWidgets.QPushButton("Structural Analysis")
        self.structural_button.clicked.connect(self.run_structural_analysis)
        layout.addWidget(self.structural_button)

    def update_elements_count(self, index):
        """Update the number of elements based on mesh quality selection"""
        element_counts = {
            0: 10000,    # Very Coarse
            1: 25000,    # Coarse
            2: 50000,    # Normal
            3: 75000,    # Fine
            4: 100000    # Very Fine
        }
        
        if index in element_counts:
            self.elements_spin.setValue(element_counts[index])
    
    def generate_mesh(self):
        """Generate mesh from geometry"""
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        try:
            num_elements = self.elements_spin.value()
            # Call the new method that was previously in MainWindow
            self.generate_analysis_mesh(num_elements)
            
            # Debug messages
            self.parent.message_text.append(f"\nMesh generated with {num_elements} elements")
            self.parent.message_text.append(f"Total nodes in mesh: {self.parent.analysis_mesher.num_nodes}")
            
            # Check for boundary nodes
            boundary_nodes = self.parent.analysis_mesher.get_boundary_nodes()
            self.parent.message_text.append(f"Found {len(boundary_nodes)} boundary nodes")
            
            # Check if we have stored any fixed nodes
            if hasattr(self.parent, 'fixed_nodes') and self.parent.fixed_nodes:
                # self.parent.message_text.append(f"Number of fixed nodes: {len(self.parent.fixed_nodes)}")
                self.parent.message_text.append(f"Fixed node indices: {sorted(list(self.parent.fixed_nodes))}")
            else:
                self.parent.message_text.append("No fixed nodes found")

            self.parent.mesh = self.parent.analysis_mesher
                
            self.parent.update_button_icon("Analysis", "check")

            # After mesh generation is successful
            self.parent.update_LivVar('mesh_generated', True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate mesh: {str(e)}")

    def generate_analysis_mesh(self, num_elements):
        if not self.parent.stl_geom:
            return
                
        try:
            # Get material properties from the material data
            if not hasattr(self.parent, 'material_data'):
                QtWidgets.QMessageBox.warning(self, "Error", "No material assigned")
                return
                    
            # Use the mesher to create the mesh
            self.parent.analysis_mesher = Mesher()
            self.parent.analysis_mesher.createMeshFromSTLFile(self.parent.stl_filepath, num_elements)
            self.parent.analysis_mesher.createEdofMatStructural()
            
            # Debug info about mesh
            self.parent.message_text.append(f"\nMesh generated with {num_elements} elements")
            self.parent.message_text.append(f"Total nodes in mesh: {self.parent.analysis_mesher.num_nodes}")
            
            # Get boundary nodes
            boundary_nodes = self.parent.analysis_mesher.get_boundary_nodes()
            boundary_points = self.parent.analysis_mesher.node_xyz[boundary_nodes]
            self.parent.message_text.append(f"Found {len(boundary_nodes)} boundary nodes")
            
            # Helper function to find nodes for faces
            def find_nodes_for_faces(faces):
                if not faces:
                    return set()
                nodes = set()
                for face in faces:
                    distances = self.parent.stl_geom.find_points_triangle_distances_vectorized(boundary_points, face['index'])
                    tolerance = min(self.parent.analysis_mesher.elem_size)*0.9
                    close_points_mask = distances < tolerance
                    nodes.update(boundary_nodes[close_points_mask])
                return nodes
            
            # Process fixed constraint faces
            fixed_nodes_xyz = find_nodes_for_faces(self.parent.fixed_faces if hasattr(self.parent, 'fixed_faces') else None)
            fixed_nodes_x = find_nodes_for_faces(self.parent.fixed_faces_x if hasattr(self.parent, 'fixed_faces_x') else None)
            fixed_nodes_y = find_nodes_for_faces(self.parent.fixed_faces_y if hasattr(self.parent, 'fixed_faces_y') else None)
            fixed_nodes_z = find_nodes_for_faces(self.parent.fixed_faces_z if hasattr(self.parent, 'fixed_faces_z') else None)
            
            # Process load faces
            load_nodes_groups = []
            if hasattr(self.parent, 'load_faces_groups') and hasattr(self.parent, 'load_forces'):
                for i, faces_group in enumerate(self.parent.load_faces_groups):
                    nodes = find_nodes_for_faces(faces_group)
                    load_nodes_groups.append(nodes)
                    force = self.parent.load_forces[i]
                    self.parent.message_text.append(f"\nLoad group {i+1}: {len(nodes)} nodes with force ({force[0]}, {force[1]}, {force[2]})N")

            #process torque faces
            torque_nodes_groups = []
            if hasattr(self.parent, 'torque_faces_groups') and hasattr(self.parent, 'torque_values'):
                for i, faces_group in enumerate(self.parent.torque_faces_groups):
                    nodes = find_nodes_for_faces(faces_group)
                    torque_nodes_groups.append(nodes)
                    torque = self.parent.torque_values[i]
                    axis_point = self.parent.torque_axis_points[i]
                    torque_magnitude = np.linalg.norm(torque)
                    self.parent.message_text.append(f"\nTorque group {i+1}: {len(nodes)} nodes with torque magnitude {torque_magnitude:.2f}N·m")
            
            # Process thermal faces
            thermal_nodes = {'fixed_temps': [], 'heat_sources': [], 'convection': []}
            if hasattr(self.parent, 'thermal_loads'):
                # Process fixed temperatures
                for nodes_list, temp in self.parent.thermal_loads.get('fixed_temps', []):
                    nodes = find_nodes_for_faces([self.parent.stl_geom.get_triangle_data(idx) for idx in nodes_list if idx < self.parent.stl_geom.stl_n_triangles])
                    thermal_nodes['fixed_temps'].append((nodes, temp))
                    self.parent.message_text.append(f"\nFixed temperature: {len(nodes)} nodes at {temp}°C")
                
                # Process heat sources
                for nodes_list, heat_flux in self.parent.thermal_loads.get('heat_sources', []):
                    nodes = find_nodes_for_faces([self.parent.stl_geom.get_triangle_data(idx) for idx in nodes_list if idx < self.parent.stl_geom.stl_n_triangles])
                    thermal_nodes['heat_sources'].append((nodes, heat_flux))
                    self.parent.message_text.append(f"\nHeat flux: {len(nodes)} nodes with {heat_flux}W/m²")
                
                # Process convection
                for nodes_list, h_coeff, ambient_temp in self.parent.thermal_loads.get('convection', []):
                    nodes = find_nodes_for_faces([self.parent.stl_geom.get_triangle_data(idx) for idx in nodes_list if idx < self.parent.stl_geom.stl_n_triangles])
                    thermal_nodes['convection'].append((nodes, h_coeff, ambient_temp))
                    self.parent.message_text.append(f"\nConvection: {len(nodes)} nodes with h={h_coeff}W/m²K, T={ambient_temp}°C")
            
            # Store nodes for later use
            self.parent.fixed_nodes = {
                'xyz': fixed_nodes_xyz,
                'x': fixed_nodes_x,
                'y': fixed_nodes_y,
                'z': fixed_nodes_z
            }
            self.parent.load_nodes_groups = load_nodes_groups
            self.parent.torque_nodes_groups = torque_nodes_groups
            self.parent.thermal_nodes = thermal_nodes
            
            
            # Summary of fixed nodes
            self.parent.message_text.append(f"\nFixed nodes summary:")
            self.parent.message_text.append(f"- XYZ fixed: {len(fixed_nodes_xyz)} nodes")
            self.parent.message_text.append(f"- X fixed: {len(fixed_nodes_x)} nodes")
            self.parent.message_text.append(f"- Y fixed: {len(fixed_nodes_y)} nodes")
            self.parent.message_text.append(f"- Z fixed: {len(fixed_nodes_z)} nodes")

            # Create visualization
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Add points
            for node in self.parent.analysis_mesher.node_xyz:
                points.InsertNextPoint(node)
            
            # Add cells (hex elements)
            for elem in self.parent.analysis_mesher.elemArray:
                hex_cell = vtk.vtkHexahedron()
                for i in range(8):
                    hex_cell.GetPointIds().SetId(i, elem[i])
                cells.InsertNextCell(hex_cell)
            
            # Create mesh structure
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_HEXAHEDRON, cells)
            
            # Create color array for visualization
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            # Set colors for visualization
            for i in range(self.parent.analysis_mesher.num_nodes):
                if i in fixed_nodes_xyz:
                    colors.InsertNextTuple3(255, 0, 0)  # Red for fully fixed
                elif i in fixed_nodes_x:
                    colors.InsertNextTuple3(255, 128, 128)  # Light red for X
                elif i in fixed_nodes_y:
                    colors.InsertNextTuple3(128, 255, 128)  # Light green for Y
                elif i in fixed_nodes_z:
                    colors.InsertNextTuple3(128, 128, 255)  # Light blue for Z
                elif any(i in nodes for nodes in load_nodes_groups):
                    colors.InsertNextTuple3(255, 165, 0)  # Orange for loaded nodes
                elif any(i in nodes for nodes in torque_nodes_groups):
                    colors.InsertNextTuple3(0, 200, 0)
                # Check for thermal boundary conditions
                elif any(i in nodes for nodes, _ in thermal_nodes['fixed_temps']):
                    colors.InsertNextTuple3(0, 0, 255)  # Blue for fixed temperature
                elif any(i in nodes for nodes, _ in thermal_nodes['heat_sources']):
                    colors.InsertNextTuple3(255, 0, 0)  # Red for heat sources
                elif any(i in nodes for nodes, _, _ in thermal_nodes['convection']):
                    colors.InsertNextTuple3(0, 255, 0)  # Green for convection
                else:
                    colors.InsertNextTuple3(200, 200, 200)  # Gray for unfixed/unloaded
            
            mesh.GetPointData().SetScalars(colors)
            
            # Create mapper and actor
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(mesh)
            
            if hasattr(self.parent, 'mesh_actor'):
                self.parent.renderer.RemoveActor(self.parent.mesh_actor)
            
            self.parent.mesh_actor = vtk.vtkActor()
            self.parent.mesh_actor.SetMapper(mapper)
            self.parent.mesh_actor.GetProperty().SetOpacity(1.0)
            self.parent.mesh_actor.GetProperty().EdgeVisibilityOn()
            self.parent.mesh_actor.GetProperty().SetEdgeColor(0, 0, 0)
            
            # Hide STL geometry only if show_mesh is checked
            if hasattr(self.parent, 'stl_actor') and hasattr(self.parent, 'display_settings'):
                if self.parent.display_settings.get('show_mesh', True):
                    self.parent.stl_actor.SetVisibility(False)
                elif self.parent.display_settings.get('show_transparent', False):
                    # Show with transparency if that setting is enabled
                    self.parent.stl_actor.SetVisibility(True)
                    self.parent.stl_actor.GetProperty().SetOpacity(0.6)
                else:
                    self.parent.stl_actor.SetVisibility(True)
            
            # Add mesh actor to renderer
            self.parent.renderer.AddActor(self.parent.mesh_actor)
            
            # Reset camera and render
            self.parent.renderer.ResetCamera()
            self.parent.vtkWidget.GetRenderWindow().Render()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate analysis mesh: {str(e)}")

    @staticmethod
    def ProcessDataforSolver(existing_mesh, fixed_nodes, load_data, youngs_modulus, poissons_ratio):
        """
        Parameters:
        -----------
        existing_mesh : Mesher object
            Pre-generated mesh from GUI
        fixed_nodes : dict
            Dictionary of fixed nodes {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
        load_data : dict
            Dictionary containing load_nodes_groups and load_forces
        youngs_modulus : float
            Young's modulus from material selection
        poissons_ratio : float
            Poisson's ratio from material selection
        """
        mesh = existing_mesh
        
        # Process fixed nodes
        fixed_dofs = []
        for node in fixed_nodes['xyz']:
            fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
            mesh.node_indices[node, 3] = 1
            
        for node in fixed_nodes['x']:
            fixed_dofs.append(3*node)
            mesh.node_indices[node, 3] = 2
            
        for node in fixed_nodes['y']:
            fixed_dofs.append(3*node + 1)
            mesh.node_indices[node, 3] = 3
            
        for node in fixed_nodes['z']:
            fixed_dofs.append(3*node + 2)
            mesh.node_indices[node, 3] = 4
            
        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
        
        # Process loads
        force = np.zeros(3*mesh.num_nodes)
        for nodes, force_vector in zip(load_data['load_nodes_groups'], load_data['load_forces']):
            if nodes:
                force_per_node = np.array(force_vector) / len(nodes)
                for node in nodes:
                    force[3*node:3*node + 3] += force_per_node
                    mesh.node_indices[node, 3] = 5

        # Process loads - initialize with zeros
        # force = np.zeros(3*mesh.num_nodes)
        # Process direct forces if they exist
        # if 'load_nodes_groups' in load_data and 'load_forces' in load_data and load_data['load_nodes_groups'] and load_data['load_forces']:
        #     for nodes, force_vector in zip(load_data['load_nodes_groups'], load_data['load_forces']):
        #         if nodes:
        #             force_per_node = np.array(force_vector) / len(nodes)
        #             for node in nodes:
        #                 force[3*node:3*node + 3] += force_per_node
        #                 mesh.node_indices[node, 3] = 5

        # # Process torque forces
        # if 'torque_nodes_groups' in load_data and 'torque_values' in load_data and 'torque_axis_points' in load_data:
        #     for i, nodes in enumerate(load_data['torque_nodes_groups']):
        #         if nodes:
        #             # Get the torque parameters
        #             torque_vector = np.array(load_data['torque_values'][i])
        #             axis_point = np.array(load_data['torque_axis_points'][i])
        #             torque_magnitude = np.linalg.norm(torque_vector)
                    
        #             # Normalize torque direction
        #             torque_dir = torque_vector / torque_magnitude if torque_magnitude > 0 else np.array([0, 0, 1])
                    
        #             # Process each node in this torque group
        #             for node in nodes:
        #                 # Get node position
        #                 node_pos = mesh.node_xyz[node]
                        
        #                 # Vector from axis point to node
        #                 r_vector = node_pos - axis_point
                        
        #                 # Project r_vector onto plane perpendicular to torque axis
        #                 # by removing the component along the torque direction
        #                 dot_product = np.dot(r_vector, torque_dir)
        #                 r_proj = r_vector - dot_product * torque_dir
        #                 r_norm = np.linalg.norm(r_proj)
                        
        #                 if r_norm > 1e-10:  # Only apply force if node is away from axis
        #                     # Calculate tangential direction (perpendicular to r_proj in the plane)
        #                     # Cross product of torque_dir and r_proj gives tangential direction
        #                     tangent_dir = np.cross(torque_dir, r_proj)
        #                     tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
                            
        #                     # Calculate force magnitude: torque = r × F, so F = torque / r
        #                     force_magnitude = torque_magnitude / (r_norm * len(nodes))
                            
        #                     # Apply force in tangential direction
        #                     force_vector = force_magnitude * tangent_dir
        #                     force[3*node:3*node + 3] += force_vector
                            
        #                     # Mark as torque node for visualization
        #                     mesh.node_indices[node, 3] = 6

        # Create boundary conditions and material properties
        bc = bound_cond.BC(force=force,
                        fixed_dofs=fixed_dofs,
                        dirichlet_values=dirichlet_values)
        mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                                            poissons_ratio=poissons_ratio)
        
        return mesh, mat_prop, bc
    
    @staticmethod
    def ProcessDataforThermalSolver(existing_mesh, thermal_nodes, thermal_conductivity):
        """
        Process data for thermal solver
        
        Parameters:
        -----------
        existing_mesh : Mesher object
            Pre-generated mesh from GUI
        thermal_nodes : dict
            Dictionary containing thermal boundary conditions 
            {'fixed_temps', 'heat_sources', 'convection', 'radiation', 'internal_heat'}
        thermal_conductivity : float
            Thermal conductivity from material selection
        """
        mesh = existing_mesh
        
        # Process fixed temperature nodes
        fixed_dofs = []
        dirichlet_values = []
        
        # Make a copy of node_indices to avoid modifying the original
        node_indices = mesh.node_indices.copy() if hasattr(mesh, 'node_indices') else np.zeros((mesh.num_nodes, 4))
        
        # Process temperature nodes - temperatures are stored in Kelvin internally
        for nodes, temp in thermal_nodes.get('fixed_temps', []):
            for node in nodes:
                if 0 <= node < mesh.num_nodes:
                    fixed_dofs.append(node)
                    dirichlet_values.append(temp)  # temp is already in Kelvin
                    node_indices[node, 3] = 1  # Mark as fixed temperature
        
        # Create heat load vector - explicitly set size to mesh.num_nodes
        force = np.zeros(mesh.num_nodes)
        
        # Add heat sources - these values depend on area, but not on temperature units
        for nodes, heat_flux in thermal_nodes.get('heat_sources', []):
            if nodes:
                heat_per_node = heat_flux / len(nodes)
                for node in nodes:
                    if 0 <= node < mesh.num_nodes:
                        force[node] += heat_per_node
                        node_indices[node, 3] = 2  # Mark as heat source
        
        # Add internal heat generation - these values depend on volume, but not on temperature units
        for nodes, heat_gen in thermal_nodes.get('internal_heat', []):
            if nodes:
                heat_per_node = heat_gen / len(nodes)
                for node in nodes:
                    if 0 <= node < mesh.num_nodes:
                        force[node] += heat_per_node
                        node_indices[node, 3] = 3  # Mark as internal heat
        
        # Set the modified node_indices back to mesh
        mesh.node_indices = node_indices
        
        # Convert arrays to the proper type
        if fixed_dofs:
            fixed_dofs = np.array(fixed_dofs).astype(np.int32)
            dirichlet_values = np.array(dirichlet_values).astype(np.float64)
        else:
            # Provide at least one fixed dof to avoid solver errors
            fixed_dofs = np.array([0]).astype(np.int32)
            dirichlet_values = np.array([300.0]).astype(np.float64)  # Default temperature (K)
        
        # Create boundary conditions
        bc = bound_cond.BC(
            force=force,
            fixed_dofs=fixed_dofs,
            dirichlet_values=dirichlet_values
        )
        
        # Create material properties
        mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity)
        
        # Return the updated mesh, material properties, and boundary conditions
        return mesh, mat_prop, bc

    def run_thermal_analysis(self):
        """Run thermal analysis using existing mesh and thermal loads"""
        try:
            # Check if mesh exists and all required components are present
            if not hasattr(self.parent, 'analysis_mesher'):
                QtWidgets.QMessageBox.warning(self, "Error", "Please generate mesh first")
                return
            
            # Check if thermal loads are defined
            mesh = self.parent.analysis_mesher
            thermal_nodes = self.parent.thermal_nodes if hasattr(self.parent, 'thermal_nodes') else {}
            
            # Print debug information
            self.parent.message_text.append(f"\nThermal analysis setup:")
            
            # Check thermal loads
            fixed_temps = thermal_nodes.get('fixed_temps', [])
            heat_sources = thermal_nodes.get('heat_sources', [])
            
            # Debug thermal load info
            self.parent.message_text.append(f"Fixed temperatures: {len(fixed_temps)}")
            self.parent.message_text.append(f"Heat sources: {len(heat_sources)}")
            
            # Check if thermal loads are defined
            if not fixed_temps and not heat_sources:
                QtWidgets.QMessageBox.warning(self, "Error", "No thermal loads defined. Please define at least one fixed temperature.")
                return
            
            # Check for material data
            if not hasattr(self.parent, 'material_data'):
                QtWidgets.QMessageBox.warning(self, "Error", "No material properties defined")
                return
            
            # Get thermal conductivity from material data
            thermal_conductivity = self.parent.material_data.get('thermal_conductivity', 50.0)
            
            # Process mesh and create boundary conditions
            self.parent.message_text.append("\nProcessing thermal boundary conditions...")
            
            # Make sure mesh has edofMatThermal
            if not hasattr(mesh, 'edofMatThermal'):
                mesh.createEdofMatThermal()
            
            # Process thermal boundary conditions
            mesh, mat_prop, bc = self.ProcessDataforThermalSolver(
                existing_mesh=mesh,
                thermal_nodes=thermal_nodes,
                thermal_conductivity=thermal_conductivity
            )
            
            # Debug info for processed data
            self.parent.message_text.append(f"Fixed temperature nodes: {len(bc.fixed_dofs)}")
            self.parent.message_text.append(f"Heat load vector: {np.count_nonzero(bc.force)} nonzero entries")
            
            # Create thermal FE solver
            self.parent.message_text.append("Creating thermal solver...")
            
            # Create solver with properly processed mesh
            fe_solver = hex_thermal_fea.ThermalFEA(
                mesh=mesh,
                mat_prop=mat_prop,
                bc=bc,
                solver=lin_solv.Solvers.PARDISO
            )
            
            # Run thermal analysis
            self.parent.message_text.append("Running thermal analysis...")
            

            # Run analysis with proper error handling
            startTime = time.time()
            try:
                u = np.asarray(fe_solver.solve())
                solve_time = time.time() - startTime
                
                # Convert temperatures according to user units
                if self.parent.settings.temperature_unit == "Celsius":
                    display_temps = u - 273.15
                    unit_symbol = "°C"
                elif self.parent.settings.temperature_unit == "Fahrenheit":
                    display_temps = (u - 273.15) * 9/5 + 32
                    unit_symbol = "°F"
                else:  # Kelvin
                    display_temps = u
                    unit_symbol = "K"
                
                # Process results
                max_temp = np.max(display_temps)
                min_temp = np.min(display_temps)
                avg_temp = np.mean(display_temps)
                
                # Print results
                self.parent.message_text.append('-----------------------------')
                self.parent.message_text.append(f"Nodes: {mesh.num_nodes}")
                self.parent.message_text.append(f'Solver: {fe_solver.solver.name}')
                self.parent.message_text.append(f"FEA time: {solve_time:.4f} seconds")
                self.parent.message_text.append(f'Max temperature: {max_temp:.2f} {unit_symbol}')
                self.parent.message_text.append(f'Min temperature: {min_temp:.2f} {unit_symbol}')
                self.parent.message_text.append(f'Avg temperature: {avg_temp:.2f} {unit_symbol}')
                self.parent.message_text.append('-----------------------------')
                
                # Store results and the mesh for visualization
                self.parent.thermal_results = {
                    'temperatures': u,  # Store in Kelvin
                    'display_temperatures': display_temps,  # Store in display units
                    'max_temperature': max_temp,
                    'min_temperature': min_temp,
                    'avg_temperature': avg_temp,
                    'unit_symbol': unit_symbol,
                    'mesh': mesh
                }
                
                # Update button status
                self.parent.update_button_icon("Analysis", "check")
                
                # Update LivVar state
                self.parent.update_LivVar('analysis.thermal', True)
                self.parent.update_LivVar('analysis.performed', True)
                
                # Visualize the results
                self.visualize_thermal_results()
                
            except Exception as e:
                self.parent.message_text.append(f"Error during solver execution: {str(e)}")
                raise  # Re-raise for outer exception handler
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to run thermal analysis: {str(e)}")
            traceback.print_exc()

    def visualize_thermal_results(self):
        """Visualize thermal analysis results"""
        try:
            if not hasattr(self.parent, 'thermal_results'):
                QtWidgets.QMessageBox.warning(self, "Error", "No thermal analysis results available")
                return
            
            # Get temperature results in the display units
            display_temps = self.parent.thermal_results['display_temperatures']
            unit_symbol = self.parent.thermal_results['unit_symbol']
            
            # Create points for visualization
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Add points
            for i in range(self.parent.analysis_mesher.num_nodes):
                points.InsertNextPoint(self.parent.analysis_mesher.node_xyz[i])
            
            # Add hex elements
            for elem in self.parent.analysis_mesher.elemArray:
                hex_cell = vtk.vtkHexahedron()
                for i in range(8):
                    hex_cell.GetPointIds().SetId(i, elem[i])
                cells.InsertNextCell(hex_cell)
            
            # Create mesh structure
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_HEXAHEDRON, cells)
            
            # Add temperature as scalars
            scalars = vtk.vtkFloatArray()
            scalars.SetNumberOfComponents(1)
            scalars.SetName("Temperature")
            
            for temp in display_temps:
                scalars.InsertNextValue(temp)
            
            mesh.GetPointData().SetScalars(scalars)
            
            # Create mapper with color mapping
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(mesh)
            mapper.SetScalarRange(np.min(display_temps), np.max(display_temps))
            
            # Create custom color lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue (cold) to red (hot)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfTableValues(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            # Create actor for temperature visualization
            if hasattr(self.parent, 'results_actor'):
                self.parent.renderer.RemoveActor(self.parent.results_actor)
            
            self.parent.results_actor = vtk.vtkActor()
            self.parent.results_actor.SetMapper(mapper)
            self.parent.results_actor.GetProperty().EdgeVisibilityOn()
            self.parent.results_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.parent.results_actor.GetProperty().SetLineWidth(1)
            
            # Remove any existing scalar bar
            if hasattr(self.parent, 'scalar_bar'):
                self.parent.renderer.RemoveActor(self.parent.scalar_bar)

            # Create a new scalar bar
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(mapper.GetLookupTable())
            scalar_bar.SetTitle(f"Temperature ({unit_symbol})")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetLabelFormat("%.1f")  # 1 decimal place

            # Set position and dimensions
            scalar_bar.SetPosition(0.85, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)

            # UNCONSTRAIN FONT SIZES
            scalar_bar.UnconstrainedFontSizeOn()  # This is CRUCIAL

            # Create title text property
            title_text_prop = vtk.vtkTextProperty()
            title_text_prop.SetFontFamilyToArial()
            title_text_prop.SetFontSize(22)  # Much larger
            title_text_prop.SetBold(True)
            title_text_prop.SetColor(0, 0, 0)

            # Create label text property
            label_text_prop = vtk.vtkTextProperty()
            label_text_prop.SetFontFamilyToArial()
            label_text_prop.SetFontSize(18)  # Smaller than title
            label_text_prop.SetBold(False)
            label_text_prop.SetColor(0, 0, 0)

            # Apply the text properties
            scalar_bar.SetTitleTextProperty(title_text_prop)
            scalar_bar.SetLabelTextProperty(label_text_prop)

            # Save and add actor
            self.parent.scalar_bar = scalar_bar
            self.parent.renderer.AddActor(scalar_bar)
            
            # Hide original mesh
            if hasattr(self.parent, 'mesh_actor'):
                self.parent.mesh_actor.SetVisibility(False)
            
            # Hide original geometry
            if hasattr(self.parent, 'stl_actor'):
                self.parent.stl_actor.SetVisibility(False)
            
            # Add actors to renderer
            self.parent.renderer.AddActor(self.parent.results_actor)
            self.parent.renderer.AddActor(self.parent.scalar_bar)
            
            # Reset camera and render
            self.parent.renderer.ResetCamera()
            self.parent.vtkWidget.GetRenderWindow().Render()
            
            # Add results summary to message box
            max_temp = self.parent.thermal_results['max_temperature']
            min_temp = self.parent.thermal_results['min_temperature']
            self.parent.message_text.append(f"Temperature range: {min_temp:.2f} to {max_temp:.2f} {unit_symbol}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to visualize thermal results: {str(e)}")
            print(f"Visualization error: {str(e)}")
        

    def run_structural_analysis(self):
        """Run structural analysis using existing mesh and boundary conditions"""
        try:
            # Check if mesh exists and all required components are present
            if not hasattr(self.parent, 'analysis_mesher'):
                QtWidgets.QMessageBox.warning(self, "Error", "Please generate mesh first")
                return
                    
            if not hasattr(self.parent, 'material_props'):
                # Create material properties if they don't exist but material data is available
                if hasattr(self.parent, 'material_data'):
                    self.parent.material_props = mat_lib.StructuralMaterial(
                        youngs_modulus=self.parent.material_data['young_modulus'],
                        poissons_ratio=self.parent.material_data['poisson_ratio']
                    )
                else:
                    QtWidgets.QMessageBox.warning(self, "Error", "No material properties defined")
                    return

            if not hasattr(self.parent, 'boundary_conditions'):
                # Process boundary conditions if they haven't been created
                try:
                    mesh = self.parent.analysis_mesher
                    fixed_nodes = self.parent.fixed_nodes
                    load_data = {
                        'load_nodes_groups': self.parent.load_nodes_groups,
                        'load_forces': self.parent.load_forces
                    }

                    # # Add torque data if it exists
                    # if hasattr(self.parent, 'torque_nodes_groups') and self.parent.torque_nodes_groups:
                    #     load_data['torque_nodes_groups'] = self.parent.torque_nodes_groups
                    #     load_data['torque_values'] = self.parent.torque_values
                    #     load_data['torque_axis_points'] = self.parent.torque_axis_points

                    #     # Debug information
                    # self.parent.message_text.append(f"\nIncluding {len(self.parent.torque_nodes_groups)} torque constraints")
                    # for i, nodes in enumerate(self.parent.torque_nodes_groups):
                    #     torque_mag = np.linalg.norm(self.parent.torque_values[i])
                    #     self.parent.message_text.append(f"  - Torque {i+1}: {len(nodes)} nodes with magnitude {torque_mag:.2f} N·m")
                    
                    # Process mesh and create boundary conditions
                    _, _, self.parent.boundary_conditions = self.ProcessDataforSolver(
                            existing_mesh=mesh,
                            fixed_nodes=fixed_nodes,
                            load_data=load_data,
                            youngs_modulus=self.parent.material_data['young_modulus'],
                            poissons_ratio=self.parent.material_data['poisson_ratio']
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Error", f"Failed to create boundary conditions: {str(e)}")
                    return

            # Create FE solver using the data from generate_analysis_mesh
            fe_solver = fea.StructFEA(
                mesh=self.parent.analysis_mesher,
                mat_prop=self.parent.material_props,
                bc=self.parent.boundary_conditions,
                solver=lin_solv.Solvers.PARDISO
            )

            # Run analysis
            self.parent.message_text.append("\nRunning structural analysis...")
            startTime = time.time()
            u = np.asarray(fe_solver.solve())

            # Compute stresses through post-processing
            fe_solver.postprocess()
            von_mises_stress = np.asarray(fe_solver.vonMisesStress)
            max_stress = np.max(von_mises_stress)
            
            # Calculate displacements
            delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
            deltaMax = np.max(delta)
            nDOF = 3*fe_solver.mesh.num_nodes
            
            # Get the appropriate length unit
            length_unit = self.parent.settings.get_length_unit_string()
            stress_unit = self.parent.settings.get_stress_unit_string()
            
            # Convert values for display if needed
            if self.parent.settings.unit_system != "MKS":
                display_deltaMax = self.parent.settings.convert_length(deltaMax, from_system="MKS")
                display_maxStress = self.parent.settings.convert_stress(max_stress, from_system="MKS")
            else:
                display_deltaMax = deltaMax
                display_maxStress = max_stress
                
            # Print results
            self.parent.message_text.append('-----------------------------')
            self.parent.message_text.append(f"nDof: {nDOF}")
            self.parent.message_text.append(f'Solver: {fe_solver.solver.name}')
            self.parent.message_text.append(f"FEA time: {time.time() - startTime:.4f} seconds")
            self.parent.message_text.append(f'Max displacement: {display_deltaMax:.6f} {length_unit}')
            self.parent.message_text.append(f'Max von Mises stress: {display_maxStress:.6f} {stress_unit}')
            self.parent.message_text.append('-----------------------------')
            
            # Store results in MKS units (for internal use) and display units
            self.parent.analysis_results = {
                'displacements': u,
                'max_displacement': deltaMax,
                'display_max_displacement': display_deltaMax,
                'delta': delta,
                'length_unit': length_unit,
                'von_mises': von_mises_stress,
                'max_stress': max_stress,
                'display_max_stress': display_maxStress,
                'stress_unit': stress_unit
            }
            
            # Update button status
            self.parent.update_button_icon("Analysis", "check")

            # After analysis is complete
            self.parent.update_LivVar('analysis.structural', True)
            self.parent.update_LivVar('analysis.performed', True)

            self.visualize_results()   

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            print(f"Detailed error: {str(e)}")


    def visualize_results(self):
        """Visualize structural analysis results with node displacements"""
        try:
            if not hasattr(self.parent, 'analysis_results'):
                QtWidgets.QMessageBox.warning(self, "Error", "No analysis results available")
                return

            # Get displacement results
            u = self.parent.analysis_results['displacements']
            delta = self.parent.analysis_results['delta']
            
            # Calculate scaling factor based on model size
            model_size = np.max(self.parent.analysis_mesher.node_xyz.max(axis=0) - 
                            self.parent.analysis_mesher.node_xyz.min(axis=0))
            max_disp = np.max(delta)
            scale_factor = 0.1 * model_size / max_disp if max_disp > 0 else 1.0
            
            # Create points for deformed mesh
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Add points with scaled displacements
            for i in range(self.parent.analysis_mesher.num_nodes):
                original_pos = self.parent.analysis_mesher.node_xyz[i]
                dx = u[3*i] * scale_factor
                dy = u[3*i + 1] * scale_factor                             
                dz = u[3*i + 2] * scale_factor
                points.InsertNextPoint(
                    original_pos[0] + dx,
                    original_pos[1] + dy,
                    original_pos[2] + dz
                )
            
            # Add hex elements
            for elem in self.parent.analysis_mesher.elemArray:
                hex_cell = vtk.vtkHexahedron()
                for i in range(8):
                    hex_cell.GetPointIds().SetId(i, elem[i])
                cells.InsertNextCell(hex_cell)
            
            # Create mesh structure
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_HEXAHEDRON, cells)
            
            # Add displacement magnitude as scalars
            scalars = vtk.vtkFloatArray()
            scalars.SetNumberOfComponents(1)
            scalars.SetName("Displacement")
            for d in delta:
                scalars.InsertNextValue(d)
            
            mesh.GetPointData().SetScalars(scalars)
            
            # Create mapper with improved color mapping
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(mesh)
            mapper.SetScalarRange(0, np.max(delta))
            
            # Create custom color lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue to red
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfTableValues(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            # Create actor for deformed mesh
            if hasattr(self.parent, 'results_actor'):
                self.parent.renderer.RemoveActor(self.parent.results_actor)
            
            self.parent.results_actor = vtk.vtkActor()
            self.parent.results_actor.SetMapper(mapper)
            self.parent.results_actor.GetProperty().EdgeVisibilityOn()
            self.parent.results_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.parent.results_actor.GetProperty().SetLineWidth(1)
            
            # Create enhanced scalar bar
            if hasattr(self.parent, 'scalar_bar'):
                self.parent.renderer.RemoveActor(self.parent.scalar_bar)
            
            # Get unit string for the scalar bar title
            length_unit = self.parent.settings.get_length_unit_string()
            
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(mapper.GetLookupTable())
            scalar_bar.SetTitle(f"Displacement ({length_unit})")  # Use the proper length unit
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetPosition(0.85, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)

            # UNCONSTRAIN FONT SIZES
            scalar_bar.UnconstrainedFontSizeOn()
            
            # Create title text property
            title_text_prop = vtk.vtkTextProperty()
            title_text_prop.SetFontFamilyToArial()
            title_text_prop.SetFontSize(22)
            title_text_prop.SetBold(True)
            title_text_prop.SetColor(0, 0, 0)

            # Create label text property
            label_text_prop = vtk.vtkTextProperty()
            label_text_prop.SetFontFamilyToArial()
            label_text_prop.SetFontSize(18)
            label_text_prop.SetBold(False)
            label_text_prop.SetColor(0, 0, 0)

            # Apply the text properties
            scalar_bar.SetTitleTextProperty(title_text_prop)
            scalar_bar.SetLabelTextProperty(label_text_prop)

            # Save and add actor
            self.parent.scalar_bar = scalar_bar
            self.parent.renderer.AddActor(scalar_bar)

            # Hide original mesh
            if hasattr(self.parent, 'mesh_actor'):
                self.parent.mesh_actor.SetVisibility(False)
            
            # Add actors to renderer
            self.parent.renderer.AddActor(self.parent.results_actor)
            self.parent.renderer.AddActor(self.parent.scalar_bar)
            
            # Reset camera and render
            self.parent.renderer.ResetCamera()
            self.parent.vtkWidget.GetRenderWindow().Render()
            
            # Add results summary with proper units
            length_unit = self.parent.settings.get_length_unit_string()
            display_max_disp = self.parent.analysis_results.get('display_max_displacement', np.max(delta))
            
            self.parent.message_text.append(f"Maximum displacement: {display_max_disp:.6f} {length_unit}")
            self.parent.message_text.append(f"Scale factor: {scale_factor:.2f}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to visualize results: {str(e)}")

    def visualize_stress(self, display_mode="Geometry"):
        """Visualize von Mises stress from structural analysis"""
        try:
            if not hasattr(self, 'analysis_results') or self.analysis_results is None:
                QtWidgets.QMessageBox.warning(self, "Error", "No analysis results available")
                return
                
            if not hasattr(self, 'analysis_mesher'):
                QtWidgets.QMessageBox.warning(self, "Error", "No analysis mesh available")
                return
                
            if 'von_mises' not in self.analysis_results:
                QtWidgets.QMessageBox.warning(self, "Error", "Von Mises stress not computed")
                return
                
            # Get stress results
            von_mises = self.analysis_results['von_mises']
            stress_unit = self.analysis_results['stress_unit']
            
            # Create points for undeformed mesh (stress visualization typically shown on original mesh)
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Add points
            for i in range(self.analysis_mesher.num_nodes):
                points.InsertNextPoint(self.analysis_mesher.node_xyz[i])
                
            # Add hex elements
            for elem in self.analysis_mesher.elemArray:
                hex_cell = vtk.vtkHexahedron()
                for i in range(8):
                    hex_cell.GetPointIds().SetId(i, elem[i])
                cells.InsertNextCell(hex_cell)
                
            # Create mesh structure
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_HEXAHEDRON, cells)
            
            # Map element-based von Mises stresses to points
            # We need to convert from element-based values to node-based values
            scalars = vtk.vtkFloatArray()
            scalars.SetNumberOfComponents(1)
            scalars.SetName("Von Mises Stress")
            
            # Create node-to-element mapping for averaging
            node_to_elements = [[] for _ in range(self.analysis_mesher.num_nodes)]
            for i, elem in enumerate(self.analysis_mesher.elemArray):
                for node_idx in elem:
                    node_to_elements[node_idx].append(i)
            
            # Compute average stress for each node
            for i in range(self.analysis_mesher.num_nodes):
                if node_to_elements[i]:
                    # Average the stress values from all elements connected to this node
                    avg_stress = np.mean([von_mises[e] for e in node_to_elements[i]])
                    scalars.InsertNextValue(avg_stress)
                else:
                    scalars.InsertNextValue(0.0)  # For nodes not connected to any element
            
            mesh.GetPointData().SetScalars(scalars)
            
            # Create mapper with improved color mapping
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(mesh)
            mapper.SetScalarRange(0, np.max(von_mises))
            
            # Create custom color lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue to red
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfTableValues(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            # Create actor for visualization
            if hasattr(self, 'results_actor'):
                self.renderer.RemoveActor(self.results_actor)
                
            self.results_actor = vtk.vtkActor()
            self.results_actor.SetMapper(mapper)
            self.results_actor.GetProperty().EdgeVisibilityOn()
            self.results_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.results_actor.GetProperty().SetLineWidth(1)
            
            # Create enhanced scalar bar
            if hasattr(self, 'scalar_bar'):
                self.renderer.RemoveActor(self.scalar_bar)
                
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(mapper.GetLookupTable())
            scalar_bar.SetTitle(f"Von Mises Stress ({stress_unit})")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetPosition(0.85, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)

            # UNCONSTRAIN FONT SIZES
            scalar_bar.UnconstrainedFontSizeOn()
            
            # Create title text property
            title_text_prop = vtk.vtkTextProperty()
            title_text_prop.SetFontFamilyToArial()
            title_text_prop.SetFontSize(22)
            title_text_prop.SetBold(True)
            title_text_prop.SetColor(0, 0, 0)

            # Create label text property
            label_text_prop = vtk.vtkTextProperty()
            label_text_prop.SetFontFamilyToArial()
            label_text_prop.SetFontSize(18)
            label_text_prop.SetBold(False)
            label_text_prop.SetColor(0, 0, 0)

            # Apply the text properties
            scalar_bar.SetTitleTextProperty(title_text_prop)
            scalar_bar.SetLabelTextProperty(label_text_prop)

            # Save and add actor
            self.scalar_bar = scalar_bar
            self.renderer.AddActor(scalar_bar)
            
            # Hide original mesh
            if hasattr(self, 'mesh_actor'):
                self.mesh_actor.SetVisibility(False)
                
            # Hide original geometry
            if hasattr(self, 'stl_actor'):
                self.stl_actor.SetVisibility(False)
                
            # Add actors to renderer
            self.renderer.AddActor(self.results_actor)
            self.renderer.AddActor(self.scalar_bar)
            
            # Reset camera and render
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
            
            # Add results summary with proper units
            max_stress = self.analysis_results.get('display_max_stress', np.max(von_mises))
            self.message_text.append(f"Maximum von Mises stress: {max_stress:.6f} {stress_unit}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to visualize von Mises stress: {str(e)}")
            print(f"Error visualizing von Mises stress: {str(e)}")
            traceback.print_exc()

#---------------------------------------------------------------------------------
class Analysis:
    def __init__(self):
        self.load_set = 0
        self.mesh_quality = "Very Coarse"
        self.num_elements = 10000
        self.deflation_type = "Rigid"
        self.num_deflation_groups = 100
        self.num_modes = 1
        self.include_thermal = True
        self.zero_strain_temp = 300
        self.remesh = False
#---------------------------------------------------------------------------------
class TopOptConstraintsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Constraints")
        self.resize(300, 500)

        # Set window to be non-modal to maintain interactivity with the viewer
        self.setWindowModality(Qt.NonModal)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Create scrollable area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        # Container widget for scroll area
        container = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout(container)
        self.form_layout.setSpacing(8)
        
        # Manufacturing constraints
        self.extrude_check = QtWidgets.QCheckBox("Extrude")
        self.extrude_combo = QtWidgets.QComboBox()
        self.extrude_combo.addItems(["XDir", "YDir", "ZDir"])
        self.form_layout.addRow(self.extrude_check, self.extrude_combo)
        # Connect changes to update visualization
        self.extrude_check.stateChanged.connect(self.update_extrude_visualization)
        self.extrude_combo.currentTextChanged.connect(self.update_extrude_visualization)
        
        self.am_build_check = QtWidgets.QCheckBox("AM Build")
        self.am_build_combo = QtWidgets.QComboBox()
        self.am_build_combo.addItems(["+XDir", "+YDir", "+ZDir", "-XDir", "-YDir", "-ZDir"])
        self.form_layout.addRow(self.am_build_check, self.am_build_combo)
        
        self.draw_direction_check = QtWidgets.QCheckBox("DrawDirection")
        self.draw_direction_combo = QtWidgets.QComboBox()
        self.draw_direction_combo.addItems(["XDir", "YDir", "ZDir"])
        self.form_layout.addRow(self.draw_direction_check, self.draw_direction_combo)

        self.draw_direction_check.stateChanged.connect(self.update_draw_direction_visualization)
        self.draw_direction_combo.currentTextChanged.connect(self.update_draw_direction_visualization)
        
        self.cyclic_sym_check = QtWidgets.QCheckBox("CyclicSym(Z)")
        self.cyclic_sym_combo = QtWidgets.QComboBox()
        self.cyclic_sym_combo.addItems([
            "(2) 180 deg",  # 360/2 = 180°
            "(3) 120 deg",  # 360/3 = 120°
            "(4) 90 deg",   # 360/4 = 90°
            "(5) 72 deg",   # 360/5 = 72°
            "(6) 60 deg",   # 360/6 = 60°
            "(7) 51 deg",   # 360/7 ≈ 51.4°
            "(8) 45 deg"    # 360/8 = 45°
        ])
        self.form_layout.addRow(self.cyclic_sym_check, self.cyclic_sym_combo)
        # Then connect the signals:
        self.cyclic_sym_check.stateChanged.connect(self.update_cyclic_symmetry_visualization)
        self.cyclic_sym_combo.currentTextChanged.connect(self.update_cyclic_symmetry_visualization)

         
        # self.connect_grid_pattern_events()
        
        # def connect_cyclic_symmetry_events(self):
        #     """Connect cyclic symmetry checkboxes and comboboxes to visualization updates"""
        #     self.cyclic_sym_check.stateChanged.connect(self.update_cyclic_symmetry_visualization)
        #     self.cyclic_sym_combo.currentTextChanged.connect(self.update_cyclic_symmetry_visualization)
        
        # Pattern constraints
        self.x_grid_check = QtWidgets.QCheckBox("XGridPattern")
        self.x_grid_spin = QtWidgets.QSpinBox()
        self.x_grid_spin.setRange(1, 10)
        self.x_grid_spin.setValue(2)
        self.form_layout.addRow(self.x_grid_check, self.x_grid_spin)
        
        self.y_grid_check = QtWidgets.QCheckBox("YGridPattern")
        self.y_grid_spin = QtWidgets.QSpinBox()
        self.y_grid_spin.setRange(1, 10)
        self.y_grid_spin.setValue(2)
        self.form_layout.addRow(self.y_grid_check, self.y_grid_spin)
        
        self.z_grid_check = QtWidgets.QCheckBox("ZGridPattern")
        self.z_grid_spin = QtWidgets.QSpinBox()
        self.z_grid_spin.setRange(1, 10)
        self.z_grid_spin.setValue(2)
        self.form_layout.addRow(self.z_grid_check, self.z_grid_spin)

        # def connect_grid_pattern_events(self):
        """Connect grid pattern checkboxes and spinboxes to visualization updates"""
        self.x_grid_check.stateChanged.connect(self.update_grid_pattern_visualization)
        self.x_grid_spin.valueChanged.connect(self.update_grid_pattern_visualization)
        
        self.y_grid_check.stateChanged.connect(self.update_grid_pattern_visualization)
        self.y_grid_spin.valueChanged.connect(self.update_grid_pattern_visualization)
        
        self.z_grid_check.stateChanged.connect(self.update_grid_pattern_visualization)
        self.z_grid_spin.valueChanged.connect(self.update_grid_pattern_visualization)
        
        # Performance constraints
        self.stress_safety_check = QtWidgets.QCheckBox("StressSafety")
        self.stress_safety_spin = QtWidgets.QDoubleSpinBox()
        self.stress_safety_spin.setRange(0.1, 10.0)
        self.stress_safety_spin.setValue(1.00)
        self.stress_safety_spin.setSingleStep(0.1)
        self.form_layout.addRow(self.stress_safety_check, self.stress_safety_spin)
        
        self.max_disp_check = QtWidgets.QCheckBox("MaxDisp(m)")
        self.max_disp_spin = QtWidgets.QDoubleSpinBox()
        self.max_disp_spin.setRange(0, 1000)
        self.max_disp_spin.setValue(160.0)
        self.max_disp_spin.setDecimals(6)
        self.form_layout.addRow(self.max_disp_check, self.max_disp_spin)
        
        self.min_freq_check = QtWidgets.QCheckBox("MinFreq(Hz)")
        self.min_freq_spin = QtWidgets.QDoubleSpinBox()
        self.min_freq_spin.setRange(0, 10000)
        self.min_freq_spin.setValue(1000.00)
        self.min_freq_spin.setDecimals(2)
        self.form_layout.addRow(self.min_freq_check, self.min_freq_spin)
        
        self.max_temp_check = QtWidgets.QCheckBox("MaxTemp(K)")
        self.max_temp_spin = QtWidgets.QDoubleSpinBox()
        self.max_temp_spin.setRange(0, 5000)
        self.max_temp_spin.setValue(2000.00)
        self.max_temp_spin.setDecimals(2)
        self.form_layout.addRow(self.max_temp_check, self.max_temp_spin)
        
        self.min_feat_check = QtWidgets.QCheckBox("MinFeat(m)")
        self.min_feat_spin = QtWidgets.QDoubleSpinBox()
        self.min_feat_spin.setRange(0, 1)
        self.min_feat_spin.setValue(0.0)
        self.min_feat_spin.setDecimals(6)
        self.form_layout.addRow(self.min_feat_check, self.min_feat_spin)
        
        self.max_feat_check = QtWidgets.QCheckBox("MaxFeat(m)")
        self.max_feat_spin = QtWidgets.QDoubleSpinBox()
        self.max_feat_spin.setRange(0, 1)
        self.max_feat_spin.setValue(0.0)
        self.max_feat_spin.setDecimals(6)
        self.form_layout.addRow(self.max_feat_check, self.max_feat_spin)
        
        # Symmetry constraints
        self.x_symmetry_check = QtWidgets.QCheckBox("X-Symmetry")
        self.form_layout.addRow(self.x_symmetry_check)
        
        self.y_symmetry_check = QtWidgets.QCheckBox("Y-Symmetry")
        self.form_layout.addRow(self.y_symmetry_check)
        
        self.z_symmetry_check = QtWidgets.QCheckBox("Z-Symmetry")
        self.form_layout.addRow(self.z_symmetry_check)

        """Connect symmetry checkboxes to visualization updates"""
        self.x_symmetry_check.stateChanged.connect(self.update_symmetry_visualization)
        self.y_symmetry_check.stateChanged.connect(self.update_symmetry_visualization)
        self.z_symmetry_check.stateChanged.connect(self.update_symmetry_visualization)
        
        # Other constraints
        self.connected_topology_check = QtWidgets.QCheckBox("Connected Topology")
        self.form_layout.addRow(self.connected_topology_check)
        
        self.keep_fixed_faces_check = QtWidgets.QCheckBox("Keep Fixed Faces")
        self.form_layout.addRow(self.keep_fixed_faces_check)
        
        # Set scroll area widget
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Apply button
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_constraints)
        layout.addWidget(self.apply_button)
        
        # Set default values
        self.connected_topology_check.setChecked(True)
        
    def apply_constraints(self):
        """Gather and apply all constraint settings, keeping visualizations"""
        constraints = {
            'manufacturing': {
                'extrude': {
                    'enabled': self.extrude_check.isChecked(),
                    'direction': self.extrude_combo.currentText()
                },
                'am_build': {
                    'enabled': self.am_build_check.isChecked(),
                    'direction': self.am_build_combo.currentText()
                },
                'draw_direction': {
                    'enabled': self.draw_direction_check.isChecked(),
                    'direction': self.draw_direction_combo.currentText()
                },
                'cyclic_symmetry': {
                    'enabled': self.cyclic_sym_check.isChecked(),
                    'angle': self.cyclic_sym_combo.currentText()
                }
            },
            'patterns': {
                'x_grid': {
                    'enabled': self.x_grid_check.isChecked(),
                    'value': self.x_grid_spin.value()
                },
                'y_grid': {
                    'enabled': self.y_grid_check.isChecked(),
                    'value': self.y_grid_spin.value()
                },
                'z_grid': {
                    'enabled': self.z_grid_check.isChecked(),
                    'value': self.z_grid_spin.value()
                }
            },
            'performance': {
                'stress_safety': {
                    'enabled': self.stress_safety_check.isChecked(),
                    'value': self.stress_safety_spin.value()
                },
                'max_displacement': {
                    'enabled': self.max_disp_check.isChecked(),
                    'value': self.max_disp_spin.value()
                },
                'min_frequency': {
                    'enabled': self.min_freq_check.isChecked(),
                    'value': self.min_freq_spin.value()
                },
                'max_temperature': {
                    'enabled': self.max_temp_check.isChecked(),
                    'value': self.max_temp_spin.value()
                },
                'min_feature': {
                    'enabled': self.min_feat_check.isChecked(),
                    'value': self.min_feat_spin.value()
                },
                'max_feature': {
                    'enabled': self.max_feat_check.isChecked(),
                    'value': self.max_feat_spin.value()
                }
            },
            'symmetry': {
                'x_symmetry': self.x_symmetry_check.isChecked(),
                'y_symmetry': self.y_symmetry_check.isChecked(),
                'z_symmetry': self.z_symmetry_check.isChecked()
            },
            'other': {
                'connected_topology': self.connected_topology_check.isChecked(),
                'keep_fixed_faces': self.keep_fixed_faces_check.isChecked()
            }
        }

        # Store constraints in parent window
        if hasattr(self.parent(), 'topopt_constraints'):
            self.parent().topopt_constraints = constraints

        # Update visualizations based on the new settings
        self.update_extrude_visualization()
        self.update_draw_direction_visualization()
        self.update_cyclic_symmetry_visualization()
        self.update_grid_pattern_visualization() 
        self.update_symmetry_visualization()

        # Update icon in main window
        if hasattr(self.parent(), 'update_button_icon'):
            self.parent().update_button_icon("TopOpt Constraints", "check")

        # Update LivVar state
        self.parent().update_LivVar('topopt.constraints_defined', True)
            
        self.parent().message_text.append("TopOpt constraints applied")


    def visualize_extrude_direction(self):
        """Create a large arrow to visualize the extrude direction that persists"""
        # Check if parent window has geometry loaded
        if not hasattr(self.parent(), 'stl_geom') or self.parent().stl_geom is None:
            return

        # Initialize constraint actors if needed
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Get the model bounds to properly size and position the arrow
        if hasattr(self.parent().stl_geom, 'get_bounding_box'):
            bbox = self.parent().stl_geom.get_bounding_box()
        else:
            # Calculate bounding box manually if get_bounding_box not available
            vertices = np.array(self.parent().stl_geom.mesh.vectors).reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Calculate dimensions and center of the model
        model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        center = [(bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, (bbox[4] + bbox[5])/2]
        
        # Arrow dimensions - make it visually significant
        arrow_length = model_size * 1.26  # % of model size
        shaft_radius = model_size * 0.015
        tip_radius = model_size * 0.04
        tip_length = model_size * 0.09

        start_offset = model_size * 0.1  # 10% of model size for offset
        
        # Create arrow source
        arrow = vtk.vtkArrowSource()
        arrow.SetShaftRadius(shaft_radius / arrow_length)  # Normalize to arrow length
        arrow.SetTipRadius(tip_radius / arrow_length)
        arrow.SetTipLength(tip_length / arrow_length)
        
        # Create transform to position and orient the arrow
        transform = vtk.vtkTransform()
        
        # Position at one end of the model based on direction
        direction = self.extrude_combo.currentText()
        if direction == "XDir":
            # Position at minimum X, point to positive X
            start_pos = [bbox[0] - arrow_length * 0.2 + start_offset, center[1], center[2]]
            transform.Translate(start_pos)
            # No rotation needed - arrow points in X direction by default
        elif direction == "YDir":
            # Position at minimum Y, point to positive Y
            start_pos = [center[0], bbox[2] - arrow_length * 0.2 + start_offset, center[2]]
            transform.Translate(start_pos)
            transform.RotateZ(90)  # Rotate to point along Y axis
        elif direction == "ZDir":
            # Position at minimum Z, point to positive Z
            start_pos = [center[0], center[1], bbox[4] - arrow_length * 0.2 + start_offset]
            transform.Translate(start_pos)
            transform.RotateY(-90)  # Rotate to point along Z axis
        
        transform.Scale(arrow_length, arrow_length, arrow_length)
        
        # Create the mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(arrow.GetOutputPort())
        
        # Remove any existing extrude arrow before creating a new one
        if self.parent().topopt_constraint_actors['extrude']:
            self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['extrude'])
        
        # Create and add the new actor
        extrude_actor = vtk.vtkActor()
        extrude_actor.SetMapper(mapper)
        extrude_actor.SetUserTransform(transform)
        
        # Set color based on direction
        if direction == "XDir":
            extrude_actor.GetProperty().SetColor(1, 0, 0)  # Red for X
        elif direction == "YDir":
            extrude_actor.GetProperty().SetColor(0, 1, 0)  # Green for Y
        elif direction == "ZDir":
            extrude_actor.GetProperty().SetColor(0, 0, 1)  # Blue for Z
        
        # Store the actor in the parent's collection
        self.parent().topopt_constraint_actors['extrude'] = extrude_actor
        
        # Add to renderer
        self.parent().renderer.AddActor(extrude_actor)
        self.parent().vtkWidget.GetRenderWindow().Render()


        # Fix for the update_extrude_visualization method - remove references to constraints 
    def update_extrude_visualization(self):
        """Update the visualization when extrude settings change"""
        # If extrude is checked, show the arrow
        if self.extrude_check.isChecked():
            self.visualize_extrude_direction()
        else:
            # If not checked, remove the visualization
            if hasattr(self.parent(), 'topopt_constraint_actors') and self.parent().topopt_constraint_actors['extrude']:
                self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['extrude'])
                self.parent().topopt_constraint_actors['extrude'] = None
                self.parent().vtkWidget.GetRenderWindow().Render()

    def visualize_draw_direction(self):
        """Create double-sided arrows to visualize the draw direction constraint"""
        # Check if parent window has geometry loaded
        if not hasattr(self.parent(), 'stl_geom') or self.parent().stl_geom is None:
            return

        # Initialize constraint actors if needed
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Get the model bounds to properly size and position the arrows
        if hasattr(self.parent().stl_geom, 'get_bounding_box'):
            bbox = self.parent().stl_geom.get_bounding_box()
        else:
            # Calculate bounding box manually if get_bounding_box not available
            vertices = np.array(self.parent().stl_geom.mesh.vectors).reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Calculate dimensions and center of the model
        model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        center = [(bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, (bbox[4] + bbox[5])/2]
        
        # Arrow dimensions - make it visually significant
        arrow_length = model_size * 1.26  # 130% of model size
        shaft_radius = model_size * 0.015
        tip_radius = model_size * 0.04
        tip_length = model_size * 0.08
        
        # Remove any existing draw direction arrows before creating new ones
        if self.parent().topopt_constraint_actors['draw_direction']:
            if isinstance(self.parent().topopt_constraint_actors['draw_direction'], list):
                for actor in self.parent().topopt_constraint_actors['draw_direction']:
                    self.parent().renderer.RemoveActor(actor)
            else:
                self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['draw_direction'])
            self.parent().topopt_constraint_actors['draw_direction'] = None
        
        # Create a list to store the arrow actors
        arrow_actors = []
        
        # Create two arrows (double-sided) for the specified direction
        direction = self.draw_direction_combo.currentText()
        
        # First arrow (pointing in positive direction)
        arrow1 = vtk.vtkArrowSource()
        arrow1.SetShaftRadius(shaft_radius / arrow_length)
        arrow1.SetTipRadius(tip_radius / arrow_length)
        arrow1.SetTipLength(tip_length / arrow_length)
        
        transform1 = vtk.vtkTransform()
        transform1.Translate(center)
        
        if direction == "XDir":
            # Offset from center in negative X direction
            transform1.Translate(-arrow_length/2, 0, 0)
            # No rotation needed - arrow points in X direction by default
        elif direction == "YDir":
            # Offset from center in negative Y direction
            transform1.Translate(0, -arrow_length/2, 0)
            transform1.RotateZ(90)  # Rotate to point along Y axis
        elif direction == "ZDir":
            # Offset from center in negative Z direction
            transform1.Translate(0, 0, -arrow_length/2)
            transform1.RotateY(-90)  # Rotate to point along Z axis
        
        transform1.Scale(arrow_length, arrow_length, arrow_length)
        
        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(arrow1.GetOutputPort())
        
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.SetUserTransform(transform1)
        actor1.GetProperty().SetColor(0, 1, 0)  # Green for all draw direction arrows
        
        self.parent().renderer.AddActor(actor1)
        arrow_actors.append(actor1)
        
        # Second arrow (pointing in negative direction)
        arrow2 = vtk.vtkArrowSource()
        arrow2.SetShaftRadius(shaft_radius / arrow_length)
        arrow2.SetTipRadius(tip_radius / arrow_length)
        arrow2.SetTipLength(tip_length / arrow_length)
        
        transform2 = vtk.vtkTransform()
        transform2.Translate(center)
        
        if direction == "XDir":
            # Offset from center in positive X direction
            transform2.Translate(arrow_length/2, 0, 0)
            transform2.RotateZ(180)  # Rotate to point in negative X direction
        elif direction == "YDir":
            # Offset from center in positive Y direction
            transform2.Translate(0, arrow_length/2, 0)
            transform2.RotateZ(-90)  # Rotate to point in negative Y direction
        elif direction == "ZDir":
            # Offset from center in positive Z direction
            transform2.Translate(0, 0, arrow_length/2)
            transform2.RotateY(90)  # Rotate to point in negative Z direction
        
        transform2.Scale(arrow_length, arrow_length, arrow_length)
        
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(arrow2.GetOutputPort())
        
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        actor2.SetUserTransform(transform2)
        actor2.GetProperty().SetColor(0, 1, 0)  # Green for all draw direction arrows
        
        self.parent().renderer.AddActor(actor2)
        arrow_actors.append(actor2)
        
        # Store the actors in the parent's collection
        self.parent().topopt_constraint_actors['draw_direction'] = arrow_actors
        
        # Render the changes
        self.parent().vtkWidget.GetRenderWindow().Render()

    def update_draw_direction_visualization(self):
        """Update the visualization when draw direction settings change"""
        # If draw direction is checked, show the arrows
        if self.draw_direction_check.isChecked():
            self.visualize_draw_direction()
        else:
            # If not checked, remove the visualization
            if hasattr(self.parent(), 'topopt_constraint_actors') and self.parent().topopt_constraint_actors['draw_direction']:
                if isinstance(self.parent().topopt_constraint_actors['draw_direction'], list):
                    for actor in self.parent().topopt_constraint_actors['draw_direction']:
                        self.parent().renderer.RemoveActor(actor)
                else:
                    self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['draw_direction'])
                self.parent().topopt_constraint_actors['draw_direction'] = None
                self.parent().vtkWidget.GetRenderWindow().Render()

    def visualize_cyclic_symmetry(self):
        """Visualize the cyclic symmetry planes"""
        # Check if parent window has geometry loaded
        if not hasattr(self.parent(), 'stl_geom') or self.parent().stl_geom is None:
            return

        # Initialize constraint actors if needed
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Get the model bounds to properly size and position the planes
        if hasattr(self.parent().stl_geom, 'get_bounding_box'):
            bbox = self.parent().stl_geom.get_bounding_box()
        else:
            # Calculate bounding box manually if get_bounding_box not available
            vertices = np.array(self.parent().stl_geom.mesh.vectors).reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Extract the cyclic symmetry pattern from the dropdown
        angle_text = self.cyclic_sym_combo.currentText()
        import re
        planes_match = re.search(r'\((\d+)\)', angle_text)
        num_planes = int(planes_match.group(1)) if planes_match else 4
        
        # Remove any existing cyclic symmetry visualization
        if self.parent().topopt_constraint_actors['cyclic_symmetry']:
            if isinstance(self.parent().topopt_constraint_actors['cyclic_symmetry'], list):
                for actor in self.parent().topopt_constraint_actors['cyclic_symmetry']:
                    self.parent().renderer.RemoveActor(actor)
            else:
                self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['cyclic_symmetry'])
            self.parent().topopt_constraint_actors['cyclic_symmetry'] = None
        
        # Calculate dimensions and center of the model
        model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        center = [(bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, (bbox[4] + bbox[5])/2]
        
        # Calculate plane size based on model
        plane_size = model_size * 1.5
        
        # Store all actors for this constraint in a list
        cyclic_actors = []
        
        # Create exactly num_planes planes
        for i in range(num_planes):
            # Create a plane source
            plane_source = vtk.vtkPlaneSource()
            
            # For Z-axis symmetry, planes are vertical along Z axis
            # Set up the plane with proper dimensions
            half_size = plane_size / 2
            plane_source.SetOrigin(-half_size, 0, bbox[4] - plane_size/4)
            plane_source.SetPoint1(half_size, 0, bbox[4] - plane_size/4)
            plane_source.SetPoint2(-half_size, 0, bbox[5] + plane_size/4)
            plane_source.Update()
            
            # Create a transform for positioning and rotating the plane
            transform = vtk.vtkTransform()
            transform.Identity()
            transform.Translate(center[0], center[1], center[2])
            
            # Calculate angle for this plane - distribute planes evenly
            angle = i * (360.0 / num_planes)
            transform.RotateZ(angle)
            
            # Apply transform to the plane
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(plane_source.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            
            plane_actor = vtk.vtkActor()
            plane_actor.SetMapper(mapper)
            
            # Set appearance properties
            plane_actor.GetProperty().SetColor(0.8, 0.2, 0.8)  # Purple for cyclic symmetry
            plane_actor.GetProperty().SetOpacity(0.5)  # Semi-transparent
            plane_actor.GetProperty().SetLineWidth(2)
            plane_actor.GetProperty().SetRepresentationToWireframe()  # Only show outline
            
            # Add to renderer and store in actors list
            self.parent().renderer.AddActor(plane_actor)
            cyclic_actors.append(plane_actor)
        
        # Store the actors in the parent's collection
        self.parent().topopt_constraint_actors['cyclic_symmetry'] = cyclic_actors
        
        # Render the changes
        self.parent().vtkWidget.GetRenderWindow().Render()

    def update_cyclic_symmetry_visualization(self):
        """Update the visualization when cyclic symmetry settings change"""
        # If cyclic symmetry is checked, show the planes
        if self.cyclic_sym_check.isChecked():
            self.visualize_cyclic_symmetry()
        else:
            # If not checked, remove the visualization
            if hasattr(self.parent(), 'topopt_constraint_actors') and self.parent().topopt_constraint_actors['cyclic_symmetry']:
                # It could be a list of actors for multiple planes
                if isinstance(self.parent().topopt_constraint_actors['cyclic_symmetry'], list):
                    for actor in self.parent().topopt_constraint_actors['cyclic_symmetry']:
                        self.parent().renderer.RemoveActor(actor)
                else:
                    self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['cyclic_symmetry'])
                
                self.parent().topopt_constraint_actors['cyclic_symmetry'] = None
                self.parent().vtkWidget.GetRenderWindow().Render()

    def visualize_grid_patterns(self):
        """
        Visualize the grid patterns (X, Y, Z) within the bounding box of the model.
        Creates planes at regular intervals based on the number of grid divisions.
        """
        # Check if parent window has geometry loaded
        if not hasattr(self.parent(), 'stl_geom') or self.parent().stl_geom is None:
            return

        # Initialize constraint actors if needed
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Get the model bounds
        if hasattr(self.parent().stl_geom, 'get_bounding_box'):
            bbox = self.parent().stl_geom.get_bounding_box()
        else:
            # Calculate bounding box manually if get_bounding_box not available
            vertices = np.array(self.parent().stl_geom.mesh.vectors).reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Visualize the bounding box first
        self.visualize_bounding_box(bbox)
        
        # Create grid planes for each enabled direction
        grid_planes = {
            'x_grid': [],
            'y_grid': [],
            'z_grid': []
        }
        
        # Create X grid planes if enabled
        if self.x_grid_check.isChecked():
            num_x_divisions = self.x_grid_spin.value()
            grid_planes['x_grid'] = self.create_grid_planes(bbox, 'x', num_x_divisions)
        
        # Create Y grid planes if enabled
        if self.y_grid_check.isChecked():
            num_y_divisions = self.y_grid_spin.value()
            grid_planes['y_grid'] = self.create_grid_planes(bbox, 'y', num_y_divisions)
        
        # Create Z grid planes if enabled
        if self.z_grid_check.isChecked():
            num_z_divisions = self.z_grid_spin.value()
            grid_planes['z_grid'] = self.create_grid_planes(bbox, 'z', num_z_divisions)
        
        # Store the planes in the parent's constraint actors
        self.parent().topopt_constraint_actors['grid_patterns'] = grid_planes
        
        # Render the scene
        self.parent().vtkWidget.GetRenderWindow().Render()

    def create_grid_planes(self, bbox, axis, num_divisions):
        """
        Create a set of grid border lines along the specified axis with the given number of divisions.
        
        Parameters:
        - bbox: Tuple containing (xmin, xmax, ymin, ymax, zmin, zmax)
        - axis: 'x', 'y', or 'z'
        - num_divisions: Number of divisions (minimum 2)
        
        Returns:
        - List of vtkActor objects representing the grid borders
        """
        if num_divisions < 2:
            num_divisions = 2  # Ensure at least 2 divisions
        
        border_actors = []
        
        # Extract bounds for the given axis
        if axis == 'x':
            min_val, max_val = bbox[0], bbox[1]
            width = bbox[3] - bbox[2]
            height = bbox[5] - bbox[4]
            min_y, max_y = bbox[2], bbox[3]
            min_z, max_z = bbox[4], bbox[5]
            color = (1.0, 0.3, 0.3)  # Reddish for X grid lines
        elif axis == 'y':
            min_val, max_val = bbox[2], bbox[3]
            width = bbox[1] - bbox[0]
            height = bbox[5] - bbox[4]
            min_x, max_x = bbox[0], bbox[1]
            min_z, max_z = bbox[4], bbox[5]
            color = (0.3, 1.0, 0.3)  # Greenish for Y grid lines
        else:  # z
            min_val, max_val = bbox[4], bbox[5]
            width = bbox[1] - bbox[0]
            height = bbox[3] - bbox[2]
            min_x, max_x = bbox[0], bbox[1]
            min_y, max_y = bbox[2], bbox[3]
            color = (0.3, 0.3, 1.0)  # Bluish for Z grid lines
        
        # Calculate spacing between planes
        axis_length = max_val - min_val
        spacing = axis_length / (num_divisions - 1) if num_divisions > 1 else axis_length
        
        # Create grid border lines
        for i in range(num_divisions):
            position = min_val + i * spacing
            
            # Create points and lines for the grid boundary
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            # Set points and lines based on the axis
            if axis == 'x':
                # YZ plane border at position
                points.InsertNextPoint(position, min_y, min_z)  # 0
                points.InsertNextPoint(position, max_y, min_z)  # 1
                points.InsertNextPoint(position, max_y, max_z)  # 2
                points.InsertNextPoint(position, min_y, max_z)  # 3
                
            elif axis == 'y':
                # XZ plane border at position
                points.InsertNextPoint(min_x, position, min_z)  # 0
                points.InsertNextPoint(max_x, position, min_z)  # 1
                points.InsertNextPoint(max_x, position, max_z)  # 2
                points.InsertNextPoint(min_x, position, max_z)  # 3
                
            else:  # z
                # XY plane border at position
                points.InsertNextPoint(min_x, min_y, position)  # 0
                points.InsertNextPoint(max_x, min_y, position)  # 1
                points.InsertNextPoint(max_x, max_y, position)  # 2
                points.InsertNextPoint(min_x, max_y, position)  # 3
            
            # Create the four line segments (rectangle edges)
            for j in range(4):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, j)
                line.GetPointIds().SetId(1, (j+1) % 4)
                lines.InsertNextCell(line)
            
            # Create a polydata object
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)
            
            # Create the mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polyData)
            
            # Create and add the new actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set appearance properties
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2)  # Slightly thinner than symmetry lines
            
            # Add to renderer
            self.parent().renderer.AddActor(actor)
            border_actors.append(actor)
        
        return border_actors

    def visualize_bounding_box(self, bbox):
        """
        Create a wireframe visualization of the bounding box.
        
        Parameters:
        - bbox: Tuple containing (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        # Remove any existing bounding box visualization
        if 'bounding_box' in self.parent().topopt_constraint_actors:
            if self.parent().topopt_constraint_actors['bounding_box']:
                self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['bounding_box'])
        
        # Extract bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        
        # Create points for the 8 corners of the cube
        points = vtk.vtkPoints()
        points.InsertNextPoint(xmin, ymin, zmin)  # 0
        points.InsertNextPoint(xmax, ymin, zmin)  # 1
        points.InsertNextPoint(xmax, ymax, zmin)  # 2
        points.InsertNextPoint(xmin, ymax, zmin)  # 3
        points.InsertNextPoint(xmin, ymin, zmax)  # 4
        points.InsertNextPoint(xmax, ymin, zmax)  # 5
        points.InsertNextPoint(xmax, ymax, zmax)  # 6
        points.InsertNextPoint(xmin, ymax, zmax)  # 7
        
        # Create the 12 edges of the cube
        lines = vtk.vtkCellArray()
        
        # Bottom face (z=zmin)
        for i in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, (i+1) % 4)
            lines.InsertNextCell(line)
        
        # Top face (z=zmax)
        for i in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i+4)
            line.GetPointIds().SetId(1, ((i+1) % 4) + 4)
            lines.InsertNextCell(line)
        
        # Connecting edges between top and bottom faces
        for i in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i+4)
            lines.InsertNextCell(line)
        
        # Create a polydata object
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set appearance
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
        actor.GetProperty().SetLineWidth(1.5)  # Slightly thicker than default
        
        # Add to renderer
        self.parent().renderer.AddActor(actor)
        
        # Store the actor
        self.parent().topopt_constraint_actors['bounding_box'] = actor

    def update_grid_pattern_visualization(self):
        """
        Update the grid pattern visualization when any grid pattern settings change.
        Should be connected to the checkbox and spinbox signals.
        """

        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()

        # Remove any existing grid pattern visualization
        if 'grid_patterns' in self.parent().topopt_constraint_actors:
            grid_patterns = self.parent().topopt_constraint_actors['grid_patterns']
            
            if grid_patterns:
                for axis, actors in grid_patterns.items():
                    for actor in actors:
                        self.parent().renderer.RemoveActor(actor)
        
        # Create new visualization if any grid pattern is checked
        if (self.x_grid_check.isChecked() or 
            self.y_grid_check.isChecked() or 
            self.z_grid_check.isChecked()):
            self.visualize_grid_patterns()
        else:
            # If none are checked, also remove the bounding box
            if 'bounding_box' in self.parent().topopt_constraint_actors:
                if self.parent().topopt_constraint_actors['bounding_box']:
                    self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['bounding_box'])
                    self.parent().topopt_constraint_actors['bounding_box'] = None
            
            # Clear the grid patterns entry
            self.parent().topopt_constraint_actors['grid_patterns'] = {
                'x_grid': [],
                'y_grid': [],
                'z_grid': []
            }
        
        # Update the render window
        self.parent().vtkWidget.GetRenderWindow().Render()

    def visualize_symmetry_planes(self):
        """
        Visualize symmetry planes (X, Y, Z) with a slight scale offset beyond the bounding box.
        """
        # Check if parent window has geometry loaded
        if not hasattr(self.parent(), 'stl_geom') or self.parent().stl_geom is None:
            return

        # Initialize constraint actors if needed
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Get the model bounds
        if hasattr(self.parent().stl_geom, 'get_bounding_box'):
            bbox = self.parent().stl_geom.get_bounding_box()
        else:
            # Calculate bounding box manually if get_bounding_box not available
            vertices = np.array(self.parent().stl_geom.mesh.vectors).reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Scale factor to make planes slightly larger than bounding box
        scale_offset = 1.15  # 15% larger than bounding box
        
        # Create symmetry planes for each enabled direction
        if self.x_symmetry_check.isChecked():
            self.create_symmetry_plane(bbox, 'x', scale_offset)
        
        if self.y_symmetry_check.isChecked():
            self.create_symmetry_plane(bbox, 'y', scale_offset)
        
        if self.z_symmetry_check.isChecked():
            self.create_symmetry_plane(bbox, 'z', scale_offset)
        
        # Render the scene
        self.parent().vtkWidget.GetRenderWindow().Render()

    def create_symmetry_plane(self, bbox, axis, scale_offset=1.15):
        """
        Create a symmetry plane border (outline only) along the specified axis with a scale offset.
        
        Parameters:
        - bbox: Tuple containing (xmin, xmax, ymin, ymax, zmin, zmax)
        - axis: 'x', 'y', or 'z'
        - scale_offset: Factor to scale the border beyond the bounding box
        """
        # Extract bounds and center for the given axis
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        
        # Calculate dimensions
        width_x = (xmax - xmin) * scale_offset
        width_y = (ymax - ymin) * scale_offset
        width_z = (zmax - zmin) * scale_offset
        
        # Create line segments for the border instead of a full plane
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        # Set points and lines based on the axis
        if axis == 'x':
            # YZ plane border at center_x
            half_height = width_z / 2
            half_width = width_y / 2
            
            # Four corners of the rectangle
            points.InsertNextPoint(center_x, center_y - half_width, center_z - half_height)  # 0
            points.InsertNextPoint(center_x, center_y + half_width, center_z - half_height)  # 1
            points.InsertNextPoint(center_x, center_y + half_width, center_z + half_height)  # 2
            points.InsertNextPoint(center_x, center_y - half_width, center_z + half_height)  # 3
            
            color = (1.0, 0.0, 0.0)  # Bright red for X plane
            
        elif axis == 'y':
            # XZ plane border at center_y
            half_height = width_z / 2
            half_width = width_x / 2
            
            # Four corners of the rectangle
            points.InsertNextPoint(center_x - half_width, center_y, center_z - half_height)  # 0
            points.InsertNextPoint(center_x + half_width, center_y, center_z - half_height)  # 1
            points.InsertNextPoint(center_x + half_width, center_y, center_z + half_height)  # 2
            points.InsertNextPoint(center_x - half_width, center_y, center_z + half_height)  # 3
            
            color = (0.0, 1.0, 0.0)  # Bright green for Y plane
            
        else:  # z
            # XY plane border at center_z
            half_height = width_y / 2
            half_width = width_x / 2
            
            # Four corners of the rectangle
            points.InsertNextPoint(center_x - half_width, center_y - half_height, center_z)  # 0
            points.InsertNextPoint(center_x + half_width, center_y - half_height, center_z)  # 1
            points.InsertNextPoint(center_x + half_width, center_y + half_height, center_z)  # 2
            points.InsertNextPoint(center_x - half_width, center_y + half_height, center_z)  # 3
            
            color = (0.0, 0.0, 1.0)  # Bright blue for Z plane
        
        # Create the four line segments (rectangle edges)
        line1 = vtk.vtkLine()
        line1.GetPointIds().SetId(0, 0)
        line1.GetPointIds().SetId(1, 1)
        lines.InsertNextCell(line1)
        
        line2 = vtk.vtkLine()
        line2.GetPointIds().SetId(0, 1)
        line2.GetPointIds().SetId(1, 2)
        lines.InsertNextCell(line2)
        
        line3 = vtk.vtkLine()
        line3.GetPointIds().SetId(0, 2)
        line3.GetPointIds().SetId(1, 3)
        lines.InsertNextCell(line3)
        
        line4 = vtk.vtkLine()
        line4.GetPointIds().SetId(0, 3)
        line4.GetPointIds().SetId(1, 0)
        lines.InsertNextCell(line4)
        
        # Create a polydata object
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        
        # Create the mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)
        
        # Remove any existing symmetry plane for this axis
        if self.parent().topopt_constraint_actors['symmetry'][f'{axis}_symmetry']:
            self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['symmetry'][f'{axis}_symmetry'])
        
        # Create and add the new actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set appearance properties
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(3)  # Thicker line for better visibility
        
        # Add to renderer
        self.parent().renderer.AddActor(actor)
        
        # Store the actor in the parent's collection
        self.parent().topopt_constraint_actors['symmetry'][f'{axis}_symmetry'] = actor

    def update_symmetry_visualization(self):
        """
        Update the symmetry plane visualizations when checkboxes change.
        """
        # Check if parent has topopt_constraint_actors initialized
        if not hasattr(self.parent(), 'topopt_constraint_actors'):
            self.parent().initialize_constraint_actors()
        
        # Visualize symmetry planes if they're checked
        self.visualize_symmetry_planes()
        
        # Remove planes that are unchecked
        if not self.x_symmetry_check.isChecked() and self.parent().topopt_constraint_actors['symmetry']['x_symmetry']:
            self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['symmetry']['x_symmetry'])
            self.parent().topopt_constraint_actors['symmetry']['x_symmetry'] = None
        
        if not self.y_symmetry_check.isChecked() and self.parent().topopt_constraint_actors['symmetry']['y_symmetry']:
            self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['symmetry']['y_symmetry'])
            self.parent().topopt_constraint_actors['symmetry']['y_symmetry'] = None
        
        if not self.z_symmetry_check.isChecked() and self.parent().topopt_constraint_actors['symmetry']['z_symmetry']:
            self.parent().renderer.RemoveActor(self.parent().topopt_constraint_actors['symmetry']['z_symmetry'])
            self.parent().topopt_constraint_actors['symmetry']['z_symmetry'] = None
        
        # Update the render window
        self.parent().vtkWidget.GetRenderWindow().Render()

    # def apply_non_visual_constraints(self):
    #     """
    #     Apply non-visual constraints and store their values in the parent window.
    #     These include performance constraints (stress, displacement, etc.) and 
    #     other constraints like connected topology and fixed faces.
    #     """
    #     # Get performance constraints
    #     performance_constraints = {
    #         'stress_safety': {
    #             'enabled': self.stress_safety_check.isChecked(),
    #             'value': self.stress_safety_spin.value()
    #         },
    #         'max_displacement': {
    #             'enabled': self.max_disp_check.isChecked(),
    #             'value': self.max_disp_spin.value()
    #         },
    #         'min_frequency': {
    #             'enabled': self.min_freq_check.isChecked(),
    #             'value': self.min_freq_spin.value()
    #         },
    #         'max_temperature': {
    #             'enabled': self.max_temp_check.isChecked(),
    #             'value': self.max_temp_spin.value()
    #         },
    #         'min_feature': {
    #             'enabled': self.min_feat_check.isChecked(),
    #             'value': self.min_feat_spin.value()
    #         },
    #         'max_feature': {
    #             'enabled': self.max_feat_check.isChecked(),
    #             'value': self.max_feat_spin.value()
    #         }
    #     }
        
    #     # Get other constraints
    #     other_constraints = {
    #         'connected_topology': self.connected_topology_check.isChecked(),
    #         'keep_fixed_faces': self.keep_fixed_faces_check.isChecked()
    #     }
        
    #     # Create a log message for all applied non-visual constraints
    #     log_message = "Applied non-visual constraints:\n"
        
    #     # Add performance constraints to log
    #     for constraint_name, constraint_data in performance_constraints.items():
    #         if constraint_data['enabled']:
    #             display_name = self.get_display_name(constraint_name)
    #             log_message += f"- {display_name}: {constraint_data['value']}\n"
        
    #     # Add other constraints to log
    #     for constraint_name, enabled in other_constraints.items():
    #         if enabled:
    #             display_name = self.get_display_name(constraint_name)
    #             log_message += f"- {display_name}\n"
        
    #     # Log the applied constraints
    #     if hasattr(self.parent(), 'log_message'):
    #         self.parent().log_message(log_message)
    #     else:
    #         self.parent().message_text.append(log_message)
        
    #     # Return the constraints in case needed elsewhere
    #     return {
    #         'performance': performance_constraints,
    #         'other': other_constraints
    #     }

    # # Helper method to convert constraint names to user-friendly display names
    # def get_display_name(self, constraint_name):
    #     """Convert internal constraint names to user-friendly display names."""
    #     name_map = {
    #         'stress_safety': 'Stress Safety Factor',
    #         'max_displacement': 'Maximum Displacement',
    #         'min_frequency': 'Minimum Frequency',
    #         'max_temperature': 'Maximum Temperature',
    #         'min_feature': 'Minimum Feature Size',
    #         'max_feature': 'Maximum Feature Size',
    #         'connected_topology': 'Connected Topology',
    #         'keep_fixed_faces': 'Keep Fixed Faces'
    #     }
    #     return name_map.get(constraint_name, constraint_name.replace('_', ' ').title())

#---------------------------------------------------------------------------------
class OptimizeTopologyWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize Topology")
        self.resize(300, 400)
        self.parent = parent
        
        # Initialize optimization state
        self.optimization_running = False
        self.optimization_thread = None
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Method selection
        method_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Method")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["DENSITY-OC", "DENSITY-MMA", "PARETO", "Levelset"])
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
        
        # Action buttons
        self.optimize_button = QtWidgets.QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.optimize_button)
        
        self.stop_button = QtWidgets.QPushButton("STOP OPTIMIZATION!")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)  # Initially disabled
        layout.addWidget(self.stop_button)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Results display
        self.results_label = QtWidgets.QLabel("Results will appear here")
        layout.addWidget(self.results_label)
        
        # Add stretch to push everything up
        layout.addStretch()
    
    def log_message(self, message):
        """Log a message to the parent's message text widget"""
        if hasattr(self.parent, 'message_text'):
            self.parent.message_text.append(message)

    def apply_topopt_constraints(self, to_params, constraints):
        """
        Apply the constraints from the TopOptConstraintsWindow to TOParams
        
        Parameters:
        -----------
        to_params : TOParams
            Topology optimization parameters to update
        constraints : dict
            Constraints dictionary from TopOptConstraintsWindow
        """
        # Apply manufacturing constraints
        manufacturing = constraints.get('manufacturing', {})
        
        # Extrude constraints
        extrude = manufacturing.get('extrude', {})
        if extrude.get('enabled', False):
            direction = extrude.get('direction', 'XDir')
            if direction == 'XDir':
                to_params.ExtrudeX = True
                to_params.ExtrudeY = False
                to_params.ExtrudeZ = False
            elif direction == 'YDir':
                to_params.ExtrudeX = False
                to_params.ExtrudeY = True
                to_params.ExtrudeZ = False
            elif direction == 'ZDir':
                to_params.ExtrudeX = False
                to_params.ExtrudeY = False
                to_params.ExtrudeZ = True
        
        # AM Build constraint
        am_build = manufacturing.get('am_build', {})
        if am_build.get('enabled', False):
            to_params.AMBuildConstraint = True
            # Set build direction
            direction = am_build.get('direction', '+ZDir')
            if direction == "+XDir":
                to_params.AMBuildDir = [1, 0, 0]
            elif direction == "-XDir":
                to_params.AMBuildDir = [-1, 0, 0]
            elif direction == "+YDir":
                to_params.AMBuildDir = [0, 1, 0]
            elif direction == "-YDir":
                to_params.AMBuildDir = [0, -1, 0]
            elif direction == "+ZDir":
                to_params.AMBuildDir = [0, 0, 1]
            elif direction == "-ZDir":
                to_params.AMBuildDir = [0, 0, -1]
        else:
            to_params.AMBuildConstraint = False
        
        # Draw direction constraint
        draw_dir = manufacturing.get('draw_direction', {})
        if draw_dir.get('enabled', False):
            to_params.DrawDirConstraint = True
            direction = draw_dir.get('direction', 'ZDir')
            if direction == 'XDir':
                to_params.DrawDirection = [1, 0, 0]
            elif direction == 'YDir':
                to_params.DrawDirection = [0, 1, 0]
            elif direction == 'ZDir':
                to_params.DrawDirection = [0, 0, 1]
        else:
            to_params.DrawDirConstraint = False
        
        # Cyclic symmetry
        cyclic_symmetry = manufacturing.get('cyclic_symmetry', {})
        if cyclic_symmetry.get('enabled', False):
            angle_text = cyclic_symmetry.get('angle', '(6) 60 deg')
            
            import re
            # Extract the number of planes from the parentheses
            planes_match = re.search(r'\((\d+)\)', angle_text)
            if planes_match:
                try:
                    # Number of planes
                    num_planes = int(planes_match.group(1))
                    
                    # Check if it's the problematic 4 planes (90° symmetry)
                    if num_planes == 4:
                        self.log_message(f"Warning: 4-plane (90°) symmetry is known to cause filter issues. Using 6-plane (60°) symmetry instead.")
                        to_params.ZAxisAngularSymmetry = 6
                    else:
                        to_params.ZAxisAngularSymmetry = num_planes
                        self.log_message(f"Applied Z-axis angular symmetry with {num_planes} planes")
                except (ValueError, TypeError):
                    self.log_message(f"Warning: Could not parse number of planes from '{angle_text}'. Disabling constraint.")
                    to_params.ZAxisAngularSymmetry = 0
            else:
                # No valid number of planes found in text
                self.log_message(f"Warning: No valid number of planes found in '{angle_text}'. Disabling cyclic symmetry constraint.")
                to_params.ZAxisAngularSymmetry = 0
        else:
            # Constraint not enabled
            to_params.ZAxisAngularSymmetry = 0
        
        # Grid patterns
        patterns = constraints.get('patterns', {})
        
        x_grid = patterns.get('x_grid', {})
        if x_grid.get('enabled', False):
            to_params.XGridPattern = True
            to_params.XGridCount = x_grid.get('value', 2)
        else:
            to_params.XGridPattern = False
            
        y_grid = patterns.get('y_grid', {})
        if y_grid.get('enabled', False):
            to_params.YGridPattern = True
            to_params.YGridCount = y_grid.get('value', 2)
        else:
            to_params.YGridPattern = False
            
        z_grid = patterns.get('z_grid', {})
        if z_grid.get('enabled', False):
            to_params.ZGridPattern = True
            to_params.ZGridCount = z_grid.get('value', 2)
        else:
            to_params.ZGridPattern = False
        
        # Symmetry constraints
        symmetry = constraints.get('symmetry', {})
        to_params.XSymmetry = symmetry.get('x_symmetry', False)
        to_params.YSymmetry = symmetry.get('y_symmetry', False)
        to_params.ZSymmetry = symmetry.get('z_symmetry', False)
        
        # Performance constraints
        performance = constraints.get('performance', {})
        
        stress_safety = performance.get('stress_safety', {})
        if stress_safety.get('enabled', False):
            to_params.StressConstraint = True
            to_params.StressSafetyFactor = stress_safety.get('value', 1.0)
        else:
            to_params.StressConstraint = False
            
        max_disp = performance.get('max_displacement', {})
        if max_disp.get('enabled', False):
            to_params.DisplacementConstraint = True
            to_params.MaxDisplacement = max_disp.get('value', 0.01)
        else:
            to_params.DisplacementConstraint = False
            
        min_freq = performance.get('min_frequency', {})
        if min_freq.get('enabled', False):
            to_params.FrequencyConstraint = True
            to_params.MinFrequency = min_freq.get('value', 100.0)
        else:
            to_params.FrequencyConstraint = False
            
        max_temp = performance.get('max_temperature', {})
        if max_temp.get('enabled', False):
            to_params.TemperatureConstraint = True
            to_params.MaxTemperature = max_temp.get('value', 373.0)  # 373K = 100C
        else:
            to_params.TemperatureConstraint = False
            
        min_feature = performance.get('min_feature', {})
        if min_feature.get('enabled', False):
            to_params.MinFeatureConstraint = True
            to_params.MinFeatureSize = min_feature.get('value', 0.0)
        else:
            to_params.MinFeatureConstraint = False
            
        max_feature = performance.get('max_feature', {})
        if max_feature.get('enabled', False):
            to_params.MaxFeatureConstraint = True
            to_params.MaxFeatureSize = max_feature.get('value', 1.0)
        else:
            to_params.MaxFeatureConstraint = False
        
        # Other constraints
        other = constraints.get('other', {})
        to_params.RemoveHangingElems = other.get('connected_topology', True)
        to_params.KeepFixedElems = other.get('keep_fixed_faces', False)
        
        return to_params
    
    @staticmethod
    def ProcessDataforTopOpt(existing_mesh, fixed_nodes, load_data, youngs_modulus, poissons_ratio, to_params):
        """
        Process data specifically for topology optimization
        
        Parameters:
        -----------
        existing_mesh : Mesher object
            Pre-generated mesh from GUI
        fixed_nodes : dict
            Dictionary of fixed nodes {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
        load_data : dict
            Dictionary containing load_nodes_groups and load_forces
        youngs_modulus : float
            Young's modulus from material selection
        poissons_ratio : float
            Poisson's ratio from material selection
        to_params : TOParams
            Topology optimization parameters
        
        Returns:
        --------
        mesh : Mesher object
            Processed mesh 
        mat_prop : StructuralMaterial
            Material properties
        bc : BC
            Boundary conditions
        """
        import mat_lib
        import bound_cond
        import numpy as np
        
        mesh = existing_mesh
        
        # Process fixed nodes
        fixed_dofs = []
        for node in fixed_nodes['xyz']:
            fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
            mesh.node_indices[node, 3] = 1
            
        for node in fixed_nodes['x']:
            fixed_dofs.append(3*node)
            mesh.node_indices[node, 3] = 2
            
        for node in fixed_nodes['y']:
            fixed_dofs.append(3*node + 1)
            mesh.node_indices[node, 3] = 3
            
        for node in fixed_nodes['z']:
            fixed_dofs.append(3*node + 2)
            mesh.node_indices[node, 3] = 4
            
        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
        
        # Process loads
        force = np.zeros(3*mesh.num_nodes)
        
        # Process direct forces if they exist
        if 'load_nodes_groups' in load_data and 'load_forces' in load_data:
            for nodes, force_vector in zip(load_data['load_nodes_groups'], load_data['load_forces']):
                if nodes:
                    force_per_node = np.array(force_vector) / len(nodes)
                    for node in nodes:
                        force[3*node:3*node + 3] += force_per_node
                        mesh.node_indices[node, 3] = 5

        # Process torque forces if they exist
        if 'torque_nodes_groups' in load_data and 'torque_values' in load_data and 'torque_axis_points' in load_data:
            for i, nodes in enumerate(load_data['torque_nodes_groups']):
                if nodes and i < len(load_data['torque_values']):
                    torque_vector = np.array(load_data['torque_values'][i])
                    axis_point = np.array(load_data['torque_axis_points'][i])
                    torque_magnitude = np.linalg.norm(torque_vector)
                    
                    # Skip if magnitude is too small
                    if torque_magnitude < 1e-10:
                        continue
                    
                    # Process each node in this torque group
                    if torque_magnitude > 0:
                        torque_dir = torque_vector / torque_magnitude
                        
                        for node in nodes:
                            # Get node position
                            if 0 <= node < len(mesh.node_xyz):
                                node_pos = mesh.node_xyz[node]
                                
                                # Vector from axis point to node
                                r_vector = node_pos - axis_point
                                
                                # Project r_vector onto plane perpendicular to torque axis
                                dot_product = np.dot(r_vector, torque_dir)
                                r_proj = r_vector - dot_product * torque_dir
                                r_norm = np.linalg.norm(r_proj)
                                
                                if r_norm > 1e-10:  # Only apply force if node is away from axis
                                    # Calculate tangential direction
                                    tangent_dir = np.cross(torque_dir, r_proj)
                                    tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
                                    
                                    # Calculate force magnitude
                                    force_magnitude = torque_magnitude / (r_norm * len(nodes))
                                    
                                    # Apply force in tangential direction
                                    force_vector = force_magnitude * tangent_dir
                                    force[3*node:3*node + 3] += force_vector
                                    
                                    # Mark as torque node for visualization
                                    mesh.node_indices[node, 3] = 6
        
        # Find elements that should be kept (fixed elements if specified in to_params)
        if to_params.KeepFixedElems:
            fixed_elems = []
            for elem_idx in range(mesh.num_elems):
                nodes = mesh.elemArray[elem_idx]
                if any(node in fixed_nodes['xyz'] or 
                       node in fixed_nodes['x'] or 
                       node in fixed_nodes['y'] or 
                       node in fixed_nodes['z'] for node in nodes):
                    fixed_elems.append(elem_idx)
            
            if fixed_elems:
                to_params.ElemsToKeep = np.array(fixed_elems)
        
        # Create boundary conditions and material properties
        bc = bound_cond.BC(force=force,
                        fixed_dofs=fixed_dofs,
                        dirichlet_values=dirichlet_values)
        
        mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                                            poissons_ratio=poissons_ratio)
        
        return mesh, mat_prop, bc
    
    def start_optimization(self):
        """Start the topology optimization process"""
        # Check prerequisites
        if (not hasattr(self.parent, 'mesh') or self.parent.mesh is None) and \
           (not hasattr(self.parent, 'analysis_mesher') or self.parent.analysis_mesher is None):
            QtWidgets.QMessageBox.warning(
                self, "Missing Data", "No mesh available. Create or import a mesh first."
            )
            return False

        # If mesh is available in analysis_mesher but not in mesh, create the reference
        if hasattr(self.parent, 'analysis_mesher') and self.parent.analysis_mesher is not None:
            if not hasattr(self.parent, 'mesh') or self.parent.mesh is None:
                self.parent.mesh = self.parent.analysis_mesher
        
        # Check for material data
        if not hasattr(self.parent, 'material_data'):
            QtWidgets.QMessageBox.warning(
                self, "Missing Data", "No material properties defined."
            )
            return False
        
        # Create boundary conditions if they don't exist yet
        if not hasattr(self.parent, 'bc') or self.parent.bc is None:
            try:
                self.log_message("Creating boundary conditions from mesh data...")
                
                # Process fixed nodes
                fixed_dofs = []
                if hasattr(self.parent, 'fixed_nodes'):
                    for node in self.parent.fixed_nodes.get('xyz', set()):
                        fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
                    
                    for node in self.parent.fixed_nodes.get('x', set()):
                        fixed_dofs.append(3*node)
                    
                    for node in self.parent.fixed_nodes.get('y', set()):
                        fixed_dofs.append(3*node + 1)
                    
                    for node in self.parent.fixed_nodes.get('z', set()):
                        fixed_dofs.append(3*node + 2)
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Missing Data", "No boundary conditions defined. Apply boundary conditions first."
                    )
                    return False
                
                # Process load forces
                force = np.zeros(3 * self.parent.mesh.num_nodes)
                if hasattr(self.parent, 'load_nodes_groups') and hasattr(self.parent, 'load_forces') and \
                   len(self.parent.load_nodes_groups) == len(self.parent.load_forces) and \
                   len(self.parent.load_nodes_groups) > 0:
                    for nodes, force_vector in zip(self.parent.load_nodes_groups, self.parent.load_forces):
                        if nodes:
                            force_per_node = np.array(force_vector) / len(nodes)
                            for node in nodes:
                                force[3*node:3*node + 3] += force_per_node
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Missing Data", "No loads defined. Apply loads first."
                    )
                    return False
                
                # Process torque forces if available
                if hasattr(self.parent, 'torque_nodes_groups') and \
                   hasattr(self.parent, 'torque_values') and \
                   hasattr(self.parent, 'torque_axis_points'):
                    for i, nodes in enumerate(self.parent.torque_nodes_groups):
                        if nodes and i < len(self.parent.torque_values):
                            torque_vector = np.array(self.parent.torque_values[i])
                            axis_point = np.array(self.parent.torque_axis_points[i])
                            torque_magnitude = np.linalg.norm(torque_vector)
                            
                            # Skip if magnitude is too small
                            if torque_magnitude < 1e-10:
                                continue
                            
                            # Process each node in this torque group
                            if torque_magnitude > 0:
                                torque_dir = torque_vector / torque_magnitude
                                
                                for node in nodes:
                                    # Get node position
                                    if 0 <= node < len(self.parent.mesh.node_xyz):
                                        node_pos = self.parent.mesh.node_xyz[node]
                                        
                                        # Vector from axis point to node
                                        r_vector = node_pos - axis_point
                                        
                                        # Project r_vector onto plane perpendicular to torque axis
                                        dot_product = np.dot(r_vector, torque_dir)
                                        r_proj = r_vector - dot_product * torque_dir
                                        r_norm = np.linalg.norm(r_proj)
                                        
                                        if r_norm > 1e-10:  # Only apply force if node is away from axis
                                            # Calculate tangential direction
                                            tangent_dir = np.cross(torque_dir, r_proj)
                                            tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
                                            
                                            # Calculate force magnitude
                                            force_magnitude = torque_magnitude / (r_norm * len(nodes))
                                            
                                            # Apply force in tangential direction
                                            force_vector = force_magnitude * tangent_dir
                                            force[3*node:3*node + 3] += force_vector
                
                # Create BC object
                from bound_cond import BC
                fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
                dirichlet_values = np.zeros_like(fixed_dofs, dtype=np.float64)
                
                self.parent.bc = BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
                self.log_message(f"Created BC with {len(fixed_dofs)} fixed DOFs")
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Failed to create boundary conditions: {str(e)}"
                )
                return False
        
        # Get topology optimization parameters
        method = self.method_combo.currentText()
        volume_fraction = self.vol_spinbox.value()
        save_intermediate = self.save_intermediate.isChecked()
        
        # Update UI state
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Log start
        self.log_message(f"Starting topology optimization with method: {method}")
        self.log_message(f"Target volume fraction: {volume_fraction}")
        
        # Update state if available
        if hasattr(self.parent, 'update_button_icon'):
            self.parent.update_button_icon("Structural TopOpt", "process")
        
        if hasattr(self.parent, 'update_LivVar'):
            self.parent.update_LivVar('topopt.structural_performing', True)
        
        # Start optimization in a separate thread
        self.optimization_thread = threading.Thread(
            target=self.run_optimization,
            args=(method, volume_fraction, save_intermediate)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        return True
    
    def run_optimization(self, method, volume_fraction, save_intermediate):
        """Run the topology optimization process in a separate thread"""
        try:
            # Get data from parent
            if hasattr(self.parent, 'analysis_mesher'):
                mesh = self.parent.analysis_mesher
            elif hasattr(self.parent, 'mesh'):
                mesh = self.parent.mesh
            else:
                raise ValueError("No mesh available for optimization")
            
            # Get material properties
            if hasattr(self.parent, 'material_data'):
                # Create material properties from material data
                youngs_modulus = self.parent.material_data['young_modulus']
                poissons_ratio = self.parent.material_data['poisson_ratio']
            else:
                raise ValueError("No material properties defined")
            
            # Get structural loads data
            fixed_nodes = self.parent.fixed_nodes if hasattr(self.parent, 'fixed_nodes') else {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
            load_data = {
                'load_nodes_groups': self.parent.load_nodes_groups if hasattr(self.parent, 'load_nodes_groups') else [],
                'load_forces': self.parent.load_forces if hasattr(self.parent, 'load_forces') else []
            }
            
            # Get body forces if available
            if hasattr(self.parent, 'torque_nodes_groups') and self.parent.torque_nodes_groups:
                load_data['torque_nodes_groups'] = self.parent.torque_nodes_groups 
                load_data['torque_values'] = self.parent.torque_values
                load_data['torque_axis_points'] = self.parent.torque_axis_points
                
        
            
            # Create TO parameters
            to_params = TOParams()
            to_params.DesiredVolFraction = volume_fraction
            
            # Apply constraints if available
            if hasattr(self.parent, 'topopt_constraints') and self.parent.topopt_constraints:
                try:
                    to_params = self.apply_topopt_constraints(to_params, self.parent.topopt_constraints)
                except Exception as constraint_error:
                    self.log_message(f"Warning: Failed to apply some constraints: {str(constraint_error)}")
                    # Reset potentially problematic constraints
                    to_params.ZAxisAngularSymmetry = 0
                    self.log_message("Disabled Z-axis angular symmetry due to error")
            
            # Process data for topology optimization
            mesh, mat_prop, bc = self.ProcessDataforTopOpt(
                existing_mesh=mesh,
                fixed_nodes=fixed_nodes,
                load_data=load_data,
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                to_params=to_params
            )
            
            # Initialize solver
            solver = lin_solv.Solvers.PARDISO
            
            # Create FE solver
            fe_solver = StructFEA(
                mesh=mesh,
                mat_prop=mat_prop,
                bc=bc,
                solver=solver,
                rtol=1e-8
            )
            
            # Run optimization based on method
            start_time = time.time()
            
            if method == "DENSITY-MMA":
                u, history, success, error_msg, n_feas = topopt_mma(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    plot_progress=False
                )
                
            elif method == "DENSITY-OC":
                u, history, success, error_msg, n_feas = topopt_optimality_criteria(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    plot_progress=False
                )
                
            elif method == "PARETO":
                u, history, success, error_msg, n_feas = topopt_pareto(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    plot_progress=False
                )
                
            elif method == "Levelset":
                u, history, success, error_msg, n_feas = topopt_levelset(
                    fe_solver=fe_solver,
                    to_params=to_params,
                    plot_progress=False
                )
            
            time_taken = time.time() - start_time
            
            # Store results in parent
            self.parent.fe_solver = fe_solver
            self.parent.optimization_results = {
                'u': u,
                'history': history,
                'success': success,
                'error_msg': error_msg,
                'n_feas': n_feas,
                'time_taken': time_taken,
                'method': method,
                'volume_fraction': volume_fraction
            }
            
            # Signal completion on the main thread
            QtCore.QMetaObject.invokeMethod(
                self, "optimization_completed", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, success),
                QtCore.Q_ARG(str, error_msg if not success else "")
            )
            
        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message(error_msg)
            
            # Signal failure on the main thread
            QtCore.QMetaObject.invokeMethod(
                self, "optimization_completed", 
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False),
                QtCore.Q_ARG(str, str(e))
            )
    
    @QtCore.pyqtSlot(bool, str)
    def optimization_completed(self, success, error_msg):
        """Handle optimization completion (called on main thread)"""
        self.optimization_running = False
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        
        if success:
            # Get results
            results = self.parent.optimization_results
            history = results['history']
            final_volume = history['volume'][-1]
            final_compliance = history['compliance'][-1]
            time_taken = results['time_taken']
            
            # Log results
            self.log_message(f"Optimization completed successfully in {time_taken:.1f} seconds")
            self.log_message(f"Final volume fraction: {final_volume:.3f}")
            self.log_message(f"Final compliance: {final_compliance:.4g}")
            
            # Update results label
            self.results_label.setText(f"Volume: {final_volume:.3f}, Compliance: {final_compliance:.4g}")
            
            # Update state if available
            if hasattr(self.parent, 'update_button_icon'):
                self.parent.update_button_icon("Structural TopOpt", "check")
            
            if hasattr(self.parent, 'update_LivVar'):
                self.parent.update_LivVar('topopt.structural_performed', True)
                self.parent.update_LivVar('topopt.structural_performing', False)
            
            # Visualize results
            # Visualize results using the method within this class
            self.visualize_optimized_topology()
            
        else:
            self.log_message(f"Optimization failed: {error_msg}")
            self.results_label.setText(f"Error: {error_msg}")
            
            # Update state if available
            if hasattr(self.parent, 'update_button_icon'):
                self.parent.update_button_icon("Structural TopOpt", "error")
            
            if hasattr(self.parent, 'update_LivVar'):
                self.parent.update_LivVar('topopt.structural_performing', False)
            
            # Show error message
            QtWidgets.QMessageBox.critical(
                self, "Optimization Failed", f"Optimization failed: {error_msg}"
            )
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_running:
            self.log_message("Requesting optimization to stop...")
            # Set a flag to stop optimization in the next iteration
            if hasattr(self.parent, 'stop_optimization'):
                self.parent.stop_optimization = True
            self.stop_button.setEnabled(False)

    def visualize_optimized_topology(self):
        """Visualize the optimized topology in the 3D viewer after optimization completes"""
        try:
            # Check if we have optimization results
            if not hasattr(self.parent, 'optimization_results') or self.parent.optimization_results is None:
                self.parent.message_text.append("Error: No optimization results available to visualize")
                return
            
            # Get optimization results
            results = self.parent.optimization_results
            history = results.get('history', {})
            method = results.get('method', 'Unknown')
            # Use default value of 0.5 if volume_fraction is not available
            volume_fraction = results.get('volume_fraction', 0.5)
            
            # Get the density field (optimization result 'u' contains the density values)
            if not hasattr(self.parent, 'fe_solver') or self.parent.fe_solver is None:
                self.parent.message_text.append("Error: No FE solver with optimized mesh available")
                return
            
            # Get element densities from the optimization results
            # The optimization result 'u' contains the density values directly
            mesh = self.parent.fe_solver.mesh
            if 'u' in results:
                # Use the optimization result directly
                densities = results['u']
            else:
                # Try to get densities from the mesh if they were stored there
                if hasattr(mesh, 'densities'):
                    densities = np.array(mesh.densities)
                else:
                    self.parent.message_text.append("Error: No density data found in optimization results")
                    return
            
            threshold = 0.5  # Default threshold for visualization
            
            # Log progress
            self.parent.message_text.append("\nVisualizing optimized topology...")
            self.parent.message_text.append(f"Method: {method}")
            self.parent.message_text.append(f"Volume fraction: {volume_fraction:.3f}")
            if history:
                if 'volume' in history and history['volume']:
                    self.parent.message_text.append(f"Final volume: {history['volume'][-1]:.3f}")
                if 'compliance' in history and history['compliance']:
                    self.parent.message_text.append(f"Final compliance: {history['compliance'][-1]:.4e}")
            
            # Remove any existing optimization visualization
            if hasattr(self.parent, 'optimized_mesh_actor') and self.parent.optimized_mesh_actor is not None:
                self.parent.renderer.RemoveActor(self.parent.optimized_mesh_actor)
                self.parent.optimized_mesh_actor = None
            
            # Remove any scalar bar from previous visualization
            if hasattr(self.parent, 'scalar_bar') and self.parent.scalar_bar is not None:
                self.parent.renderer.RemoveActor(self.parent.scalar_bar)
                self.parent.scalar_bar = None
            
            # Hide mesh actor if exists
            if hasattr(self.parent, 'mesh_actor') and self.parent.mesh_actor is not None:
                self.parent.mesh_actor.SetVisibility(False)
            
            # Hide stl actor if exists
            if hasattr(self.parent, 'stl_actor') and self.parent.stl_actor is not None:
                self.parent.stl_actor.SetVisibility(False)
            
            # Create points for visualization
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()
            
            # Add points
            for i in range(mesh.num_nodes):
                points.InsertNextPoint(mesh.node_xyz[i])
            
            # Add hex elements
            for elem in mesh.elemArray:
                hex_cell = vtk.vtkHexahedron()
                for i in range(8):
                    hex_cell.GetPointIds().SetId(i, elem[i])
                cells.InsertNextCell(hex_cell)
            
            # Create mesh structure
            vtk_mesh = vtk.vtkUnstructuredGrid()
            vtk_mesh.SetPoints(points)
            vtk_mesh.SetCells(vtk.VTK_HEXAHEDRON, cells)
            
            # Add density as scalars - map element-based values to cells
            scalars = vtk.vtkFloatArray()
            scalars.SetNumberOfComponents(1)
            scalars.SetName("Density")
            
            for density in densities:
                scalars.InsertNextValue(density)
            
            vtk_mesh.GetCellData().SetScalars(scalars)
            
            # Create mapper with color mapping
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(vtk_mesh)
            mapper.SetScalarRange(0, 1) # Density range is typically 0-1
            
            # Create custom color lookup table
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # Blue (low density) to red (high density)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetNumberOfTableValues(256)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            # Create the optimized mesh actor
            self.parent.optimized_mesh_actor = vtk.vtkActor()
            self.parent.optimized_mesh_actor.SetMapper(mapper)
            self.parent.optimized_mesh_actor.GetProperty().EdgeVisibilityOn()
            self.parent.optimized_mesh_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.parent.optimized_mesh_actor.GetProperty().SetLineWidth(1)
            
            # Create a scalar bar
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(mapper.GetLookupTable())
            scalar_bar.SetTitle("Element Density")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetPosition(0.85, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)

            # Improve font appearance
            scalar_bar.UnconstrainedFontSizeOn()
            
            # Create title text property
            title_text_prop = vtk.vtkTextProperty()
            title_text_prop.SetFontFamilyToArial()
            title_text_prop.SetFontSize(22)
            title_text_prop.SetBold(True)
            title_text_prop.SetColor(0, 0, 0)

            # Create label text property
            label_text_prop = vtk.vtkTextProperty()
            label_text_prop.SetFontFamilyToArial()
            label_text_prop.SetFontSize(18)
            label_text_prop.SetBold(False)
            label_text_prop.SetColor(0, 0, 0)

            # Apply the text properties
            scalar_bar.SetTitleTextProperty(title_text_prop)
            scalar_bar.SetLabelTextProperty(label_text_prop)
            
            # Store and add actor
            self.parent.scalar_bar = scalar_bar
            
            # Add title for the optimization result
            title = f"Optimized topology (Vol.Frac: {volume_fraction:.3f})"
            if not hasattr(self.parent, 'title_actor') or self.parent.title_actor is None:
                title_actor = vtk.vtkTextActor()
                title_actor.SetPosition(10, 10)
                title_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
                title_actor.GetTextProperty().SetFontSize(16)
                title_actor.GetTextProperty().SetBold(True)
                self.parent.title_actor = title_actor
                self.parent.renderer.AddActor2D(title_actor)
            
            self.parent.title_actor.SetInput(title)
            
            # Add actors to renderer
            self.parent.renderer.AddActor(self.parent.optimized_mesh_actor)
            self.parent.renderer.AddActor(self.parent.scalar_bar)
            
            # Reset camera and render
            self.parent.renderer.ResetCamera()
            self.parent.vtkWidget.GetRenderWindow().Render()
            
        except Exception as e:
            self.parent.message_text.append(f"Error visualizing optimized topology: {str(e)}")
            import traceback
            traceback.print_exc()

    def visualize_isosurface(self, mesh, densities, threshold=0.5):
        """Create a smooth isosurface visualization of the optimized topology"""
        try:
            # Check if mesh has structured grid dimensions
            is_structured = hasattr(mesh, 'nx') and hasattr(mesh, 'ny') and hasattr(mesh, 'nz')
            
            if is_structured:
                # Structured mesh approach
                nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
                
                # Create image data
                image_data = vtk.vtkImageData()
                image_data.SetDimensions(nx+1, ny+1, nz+1)
                image_data.SetOrigin(mesh.x_min, mesh.y_min, mesh.z_min)
                image_data.SetSpacing(
                    (mesh.x_max - mesh.x_min) / nx,
                    (mesh.y_max - mesh.y_min) / ny,
                    (mesh.z_max - mesh.z_min) / nz
                )
                
                # Assign density values to cell data
                cell_data = vtk.vtkFloatArray()
                cell_data.SetName("Density")
                cell_data.SetNumberOfComponents(1)
                cell_data.SetNumberOfTuples(nx * ny * nz)
                
                for i in range(len(densities)):
                    cell_data.SetValue(i, densities[i])
                    
                image_data.GetCellData().AddArray(cell_data)
                image_data.GetCellData().SetActiveScalars("Density")
                
                # Convert cell data to point data for smoother isosurface
                cell_to_point = vtk.vtkCellDataToPointData()
                cell_to_point.SetInputData(image_data)
                cell_to_point.PassCellDataOn()
                cell_to_point.Update()
                
                # Create isosurface using contour filter
                contour = vtk.vtkContourFilter()
                contour.SetInputConnection(cell_to_point.GetOutputPort())
                contour.SetValue(0, threshold)  # Isovalue = threshold
                contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Density")
                contour.Update()
                
            else:
                # Unstructured grid approach
                # Create a vtkUnstructuredGrid from the mesh
                points = vtk.vtkPoints()
                for node_coords in mesh.node_xyz:
                    points.InsertNextPoint(node_coords)
                    
                grid = vtk.vtkUnstructuredGrid()
                grid.SetPoints(points)
                
                # Add hexahedral cells with density values
                density_array = vtk.vtkFloatArray()
                density_array.SetName("Density")
                
                for elem_idx, elem in enumerate(mesh.elemArray):
                    if elem_idx < len(densities):  # Safety check
                        hex_elem = vtk.vtkHexahedron()
                        for i, node_id in enumerate(elem):
                            if i < 8:  # Ensure we only use 8 points for a hexahedron
                                hex_elem.GetPointIds().SetId(i, node_id)
                        
                        cell_id = grid.InsertNextCell(hex_elem.GetCellType(), hex_elem.GetPointIds())
                        density_array.InsertNextValue(densities[elem_idx])
                
                grid.GetCellData().AddArray(density_array)
                grid.GetCellData().SetActiveScalars("Density")
                
                # Convert cell data to point data
                cell_to_point = vtk.vtkCellDataToPointData()
                cell_to_point.SetInputData(grid)
                cell_to_point.PassCellDataOn()
                cell_to_point.Update()
                
                # Create isosurface using contour filter
                contour = vtk.vtkContourFilter()
                contour.SetInputData(cell_to_point.GetOutput())
                contour.SetValue(0, threshold)
                contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Density")
                contour.Update()
            
            # Clean up mesh and generate normals for better visualization
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(contour.GetOutputPort())
            clean.Update()
            
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(clean.GetOutputPort())
            normals.SetFeatureAngle(60)
            normals.SplittingOff()
            normals.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            
            # Create actor with material properties
            self.optimized_mesh_actor = vtk.vtkActor()
            self.optimized_mesh_actor.SetMapper(mapper)
            self.optimized_mesh_actor.GetProperty().SetColor(0.2, 0.7, 0.9)  # Blue color
            self.optimized_mesh_actor.GetProperty().SetAmbient(0.2)
            self.optimized_mesh_actor.GetProperty().SetDiffuse(0.8)
            self.optimized_mesh_actor.GetProperty().SetSpecular(0.5)
            self.optimized_mesh_actor.GetProperty().SetSpecularPower(20)
            
            # Add to renderer
            self.renderer.AddActor(self.optimized_mesh_actor)
            
        except Exception as e:
            self.message_text.append(f"Error creating isosurface: {str(e)}")
            traceback.print_exc()

    def visualize_elements(self, mesh, densities, threshold=0.5):
        """Create an element-based visualization of the optimized topology"""
        try:
            # Create points for the mesh nodes
            points = vtk.vtkPoints()
            for node_coords in mesh.node_xyz:
                points.InsertNextPoint(node_coords)
            
            # Create unstructured grid
            grid = vtk.vtkUnstructuredGrid()
            grid.SetPoints(points)
            
            # Add cells (elements) with density above threshold
            for elem_idx, elem in enumerate(mesh.elemArray):
                if elem_idx < len(densities) and densities[elem_idx] > threshold:
                    hex_elem = vtk.vtkHexahedron()
                    for i, node_id in enumerate(elem):
                        if i < 8:  # Hexahedron needs 8 points
                            hex_elem.GetPointIds().SetId(i, node_id)
                    
                    grid.InsertNextCell(hex_elem.GetCellType(), hex_elem.GetPointIds())
            
            # Create mapper and actor
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(grid)
            
            self.optimized_mesh_actor = vtk.vtkActor()
            self.optimized_mesh_actor.SetMapper(mapper)
            self.optimized_mesh_actor.GetProperty().SetColor(0.2, 0.7, 0.9)
            self.optimized_mesh_actor.GetProperty().SetAmbient(0.2)
            self.optimized_mesh_actor.GetProperty().SetDiffuse(0.8)
            self.optimized_mesh_actor.GetProperty().SetOpacity(1.0)
            self.optimized_mesh_actor.GetProperty().EdgeVisibilityOn()
            self.optimized_mesh_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.optimized_mesh_actor.GetProperty().SetLineWidth(1.0)
            
            # Add to renderer
            self.renderer.AddActor(self.optimized_mesh_actor)
            
        except Exception as e:
            self.message_text.append(f"Error creating element visualization: {str(e)}")
            traceback.print_exc()
    
# ---------------------------------------------------------------------------------

class ThermalLoadsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowModality(Qt.NonModal)  # Allow face selection while dialog is open
        self.setWindowTitle("Thermal Loads")
        self.parent = parent
        self.thermal_loads = {
            "fixed_temps": [],      # List of (node_ids, temperature)
            "heat_sources": [],     # List of (node_ids, heat_value)
            "total_heat_sources": [],  # List of (node_ids, total_heat_value)
            "convection": [],       # List of (node_ids, h_coeff, ambient_temp)
            "radiation": [],        # List of (node_ids, emissivity, ambient_temp)
            "internal_heat": []     # List of (node_ids, heat_generation)
        }

        if hasattr(self.parent, 'thermal_loads'):
            # Ensure all required keys exist in the parent's dictionary
            for key in self.thermal_loads:
                if key not in self.parent.thermal_loads:
                    self.parent.thermal_loads[key] = []
        else:
            # Create the dictionary in the parent if it doesn't exist
            self.parent.thermal_loads = self.thermal_loads
        
        # Make sure parent has actor lists (don't create local ones)
        if not hasattr(self.parent, 'heat_source_actors'):
            self.parent.heat_source_actors = []
        if not hasattr(self.parent, 'fixed_temp_actors'):
            self.parent.fixed_temp_actors = []
        if not hasattr(self.parent, 'total_heat_actors'):
            self.parent.total_heat_actors = []  # New actor list for total heat
        if not hasattr(self.parent, 'convection_actors'):
            self.parent.convection_actors = []
        if not hasattr(self.parent, 'radiation_actors'):
            self.parent.radiation_actors = []
        if not hasattr(self.parent, 'internal_heat_actors'):
            self.parent.internal_heat_actors = []
        
        self.setup_ui()
        self.load_existing_thermal_loads()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Get temperature unit symbol
        temp_unit = self.parent.settings.get_temperature_unit_symbol()
        length_unit = self.parent.settings.get_length_unit_string()
        
        # Selection dropdown
        selection_layout = QtWidgets.QHBoxLayout()
        selection_label = QtWidgets.QLabel("Selection")
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(["Coarse Cylinder"])
        selection_layout.addWidget(selection_label)
        selection_layout.addWidget(self.selection_combo)
        layout.addLayout(selection_layout)
        
        # Thermal Type dropdown
        thermal_type_layout = QtWidgets.QHBoxLayout()
        thermal_type_label = QtWidgets.QLabel("Thermal Type")
        self.thermal_type = QtWidgets.QComboBox()
        self.thermal_type.addItems(["Temperature", "Heat Flux", "Total Heat", "Convection", "Radiation", "Internal Heat"])
        thermal_type_layout.addWidget(thermal_type_label)
        thermal_type_layout.addWidget(self.thermal_type)
        layout.addLayout(thermal_type_layout)
        
        # Temperature input group
        self.temp_group = QtWidgets.QGroupBox()
        temp_layout = QtWidgets.QVBoxLayout(self.temp_group)
        
        # Temperature value
        temp_value_layout = QtWidgets.QHBoxLayout()
        temp_value_label = QtWidgets.QLabel(f"Temperature ({temp_unit})")
        self.temp_value_spin = QtWidgets.QDoubleSpinBox()
        
        # Set range based on temperature unit
        if self.parent.settings.temperature_unit == "Kelvin":
            self.temp_value_spin.setRange(0, 10000)
            self.temp_value_spin.setValue(300)
        elif self.parent.settings.temperature_unit == "Celsius":
            self.temp_value_spin.setRange(-273.15, 10000)
            self.temp_value_spin.setValue(25)
        else:  # Fahrenheit
            self.temp_value_spin.setRange(-459.67, 18000)
            self.temp_value_spin.setValue(77)
            
        self.temp_value_spin.setDecimals(1)
        temp_value_layout.addWidget(temp_value_label)
        temp_value_layout.addWidget(self.temp_value_spin)
        temp_layout.addLayout(temp_value_layout)
        
        layout.addWidget(self.temp_group)
        
        # Heat Flux input group
        self.heat_flux_group = QtWidgets.QGroupBox()
        heat_flux_layout = QtWidgets.QVBoxLayout(self.heat_flux_group)
        
       # Heat Flux value with unit based on current unit system
        heat_flux_value_layout = QtWidgets.QHBoxLayout()
        heat_flux_value_label = QtWidgets.QLabel(f"Heat Flux (W/{length_unit}²)")
        self.heat_flux_value_spin = QtWidgets.QDoubleSpinBox()
        self.heat_flux_value_spin.setRange(-1e6, 1e6)
        
        # Adjust default value based on unit system
        if self.parent.settings.unit_system == "MKS":
            self.heat_flux_value_spin.setValue(1000)  # W/m²
        elif self.parent.settings.unit_system == "mmKS":
            self.heat_flux_value_spin.setValue(0.001)  # W/mm²
        else:  # IPS
            self.heat_flux_value_spin.setValue(0.645)  # W/in²
            
        self.heat_flux_value_spin.setDecimals(6)
        heat_flux_value_layout.addWidget(heat_flux_value_label)
        heat_flux_value_layout.addWidget(self.heat_flux_value_spin)
        heat_flux_layout.addLayout(heat_flux_value_layout)
        
        layout.addWidget(self.heat_flux_group)
        
        # Internal Heat Generation input group
        self.internal_heat_group = QtWidgets.QGroupBox()
        internal_heat_layout = QtWidgets.QVBoxLayout(self.internal_heat_group)
        
        # Internal Heat Generation value
        internal_heat_value_layout = QtWidgets.QHBoxLayout()
        internal_heat_value_label = QtWidgets.QLabel(f"Heat Generation (W/{length_unit}³)")
        self.internal_heat_value_spin = QtWidgets.QDoubleSpinBox()
        self.internal_heat_value_spin.setRange(0, 1e8)
        
        # Adjust default value based on unit system
        if self.parent.settings.unit_system == "MKS":
            self.internal_heat_value_spin.setValue(10000)  # W/m³
        elif self.parent.settings.unit_system == "mmKS":
            self.internal_heat_value_spin.setValue(1e-5)   # W/mm³
        else:  # IPS
            self.internal_heat_value_spin.setValue(0.016)  # W/in³
            
        self.internal_heat_value_spin.setDecimals(6)
        internal_heat_value_layout.addWidget(internal_heat_value_label)
        internal_heat_value_layout.addWidget(self.internal_heat_value_spin)
        internal_heat_layout.addLayout(internal_heat_value_layout)
        
        layout.addWidget(self.internal_heat_group)

        # Total Heat UI 
        self.total_heat_group = QtWidgets.QGroupBox()
        total_heat_layout = QtWidgets.QVBoxLayout(self.total_heat_group)
        
        # Total heat value
        total_heat_value_layout = QtWidgets.QHBoxLayout()
        total_heat_value_label = QtWidgets.QLabel("Total Heat (W)")
        self.total_heat_value_spin = QtWidgets.QDoubleSpinBox()
        self.total_heat_value_spin.setRange(-1e6, 1e6)
        self.total_heat_value_spin.setDecimals(1)
        self.total_heat_value_spin.setValue(100)
        total_heat_value_layout.addWidget(total_heat_value_label)
        total_heat_value_layout.addWidget(self.total_heat_value_spin)
        total_heat_layout.addLayout(total_heat_value_layout)
        
        layout.addWidget(self.total_heat_group)
        
        # Convection input group
        self.convection_group = QtWidgets.QGroupBox()
        convection_layout = QtWidgets.QVBoxLayout(self.convection_group)
        
        # Convection coefficient
        conv_coeff_layout = QtWidgets.QHBoxLayout()
        conv_coeff_label = QtWidgets.QLabel(f"Convection Coeff. (W/{length_unit}²·{temp_unit})")
        self.conv_coeff_spin = QtWidgets.QDoubleSpinBox()
        self.conv_coeff_spin.setRange(0, 10000)
        self.conv_coeff_spin.setValue(25)
        self.conv_coeff_spin.setDecimals(3)
        conv_coeff_layout.addWidget(conv_coeff_label)
        conv_coeff_layout.addWidget(self.conv_coeff_spin)
        convection_layout.addLayout(conv_coeff_layout)
        
        # Ambient temperature
        ambient_temp_layout = QtWidgets.QHBoxLayout()
        ambient_temp_label = QtWidgets.QLabel(f"Ambient Temp. ({temp_unit})")
        self.ambient_temp_spin = QtWidgets.QDoubleSpinBox()
        
        # Set range based on temperature unit
        if self.parent.settings.temperature_unit == "Kelvin":
            self.ambient_temp_spin.setRange(0, 10000)
            self.ambient_temp_spin.setValue(300)
        elif self.parent.settings.temperature_unit == "Celsius":
            self.ambient_temp_spin.setRange(-273.15, 10000)
            self.ambient_temp_spin.setValue(25)
        else:  # Fahrenheit
            self.ambient_temp_spin.setRange(-459.67, 18000)
            self.ambient_temp_spin.setValue(77)
            
        self.ambient_temp_spin.setDecimals(1)
        ambient_temp_layout.addWidget(ambient_temp_label)
        ambient_temp_layout.addWidget(self.ambient_temp_spin)
        convection_layout.addLayout(ambient_temp_layout)
        
        layout.addWidget(self.convection_group)
        
        # Radiation input group
        self.radiation_group = QtWidgets.QGroupBox()
        radiation_layout = QtWidgets.QVBoxLayout(self.radiation_group)
        
        # Emissivity
        emissivity_layout = QtWidgets.QHBoxLayout()
        emissivity_label = QtWidgets.QLabel("Emissivity (0-1)")
        self.emissivity_spin = QtWidgets.QDoubleSpinBox()
        self.emissivity_spin.setRange(0, 1)
        self.emissivity_spin.setValue(0.8)
        self.emissivity_spin.setDecimals(3)
        self.emissivity_spin.setSingleStep(0.05)
        emissivity_layout.addWidget(emissivity_label)
        emissivity_layout.addWidget(self.emissivity_spin)
        radiation_layout.addLayout(emissivity_layout)
        
        # Radiation ambient temperature
        rad_ambient_temp_layout = QtWidgets.QHBoxLayout()
        rad_ambient_temp_label = QtWidgets.QLabel(f"Ambient Temp. ({temp_unit})")
        self.rad_ambient_temp_spin = QtWidgets.QDoubleSpinBox()
        
        # Set range based on temperature unit
        if self.parent.settings.temperature_unit == "Kelvin":
            self.rad_ambient_temp_spin.setRange(0, 10000)
            self.rad_ambient_temp_spin.setValue(300)
        elif self.parent.settings.temperature_unit == "Celsius":
            self.rad_ambient_temp_spin.setRange(-273.15, 10000)
            self.rad_ambient_temp_spin.setValue(25)
        else:  # Fahrenheit
            self.rad_ambient_temp_spin.setRange(-459.67, 18000)
            self.rad_ambient_temp_spin.setValue(77)
            
        self.rad_ambient_temp_spin.setDecimals(1)
        rad_ambient_temp_layout.addWidget(rad_ambient_temp_label)
        rad_ambient_temp_layout.addWidget(self.rad_ambient_temp_spin)
        radiation_layout.addLayout(rad_ambient_temp_layout)
        
        layout.addWidget(self.radiation_group)
        
        # Initially hide all groups
        self.heat_flux_group.setVisible(False)
        self.convection_group.setVisible(False)
        self.radiation_group.setVisible(False)
        self.internal_heat_group.setVisible(False)
        self.total_heat_group.setVisible(False)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_thermal_load)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        
        # Connect thermal type change
        self.thermal_type.currentTextChanged.connect(self.on_thermal_type_changed)
    
    def on_thermal_type_changed(self, thermal_type):
        # Show/hide input groups based on thermal type
        self.temp_group.setVisible(thermal_type == "Temperature")
        self.heat_flux_group.setVisible(thermal_type == "Heat Flux")
        self.total_heat_group.setVisible(thermal_type == "Total Heat")
        self.convection_group.setVisible(thermal_type == "Convection")
        self.radiation_group.setVisible(thermal_type == "Radiation")
        self.internal_heat_group.setVisible(thermal_type == "Internal Heat")
        self.adjustSize()
    
    def load_existing_thermal_loads(self):
        """Load any existing thermal loads from the parent"""
        if hasattr(self.parent, 'thermal_loads'):
            self.thermal_loads = self.parent.thermal_loads
    
    def apply_thermal_load(self):
        """Apply the selected thermal load type to selected faces"""
        thermal_type = self.thermal_type.currentText()
        
        if thermal_type == "Temperature":
            self.apply_temperature()
        elif thermal_type == "Heat Flux":
            self.apply_heat_flux()
        elif thermal_type == "Total Heat":
            self.apply_total_heat()
        elif thermal_type == "Convection":
            self.apply_convection()
        elif thermal_type == "Radiation":
            self.apply_radiation()
        elif thermal_type == "Internal Heat":
            self.apply_internal_heat()
    
    def apply_temperature(self):
        """Apply fixed temperature to selected nodes"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
                
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No surface selected")
            return
        
        # Get nodes for the selected triangles
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        
        # Get temperature from UI in the unit shown to user
        temperature_ui = self.temp_value_spin.value()
        
        # Convert to Kelvin for internal storage (assuming your solver works in Kelvin)
        if self.parent.settings.temperature_unit == "Celsius":
            temperature_kelvin = temperature_ui + 273.15
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            temperature_kelvin = (temperature_ui - 32) * 5/9 + 273.15
        else:  # Already Kelvin
            temperature_kelvin = temperature_ui
        
        # Store in thermal loads (in Kelvin)
        self.thermal_loads["fixed_temps"].append((nodes, temperature_kelvin))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_fixed_temp(nodes, temperature_kelvin)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Show confirmation message with the user-facing temperature unit
        if self.parent.settings.temperature_unit == "Celsius":
            display_temp = temperature_ui
            unit = "°C"
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            display_temp = temperature_ui
            unit = "°F"
        else:
            display_temp = temperature_ui
            unit = "K"
        
        self.parent.message_text.append(f"Applied temperature of {display_temp}{unit} to {len(nodes)} triangles")
    
    def apply_heat_flux(self):
        """Apply heat flux to selected nodes"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No surface selected")
            return
        
        # Get nodes for the selected triangles
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        heat_flux = self.heat_flux_value_spin.value()
        
        # Add to thermal loads
        self.thermal_loads["heat_sources"].append((nodes, heat_flux))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_heat_source(nodes, heat_flux)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_LivVar("thermal_loads.heat_sources", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Show confirmation message
        self.parent.message_text.append(f"Applied heat flux of {heat_flux}W/m² to {len(nodes)} triangles")
    
    def apply_total_heat(self):
        """Apply total heat to selected nodes"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No surface selected")
            return
        
        # Get nodes for the selected triangles
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        total_heat = self.total_heat_value_spin.value()
        
        # Calculate total area of the selected triangles
        total_area = 0
        for idx in selected_triangles:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                # Get triangle vertices
                triangle = self.parent.stl_geom.get_triangle_data(idx)
                total_area += triangle['area']
        
        # Calculate heat flux by dividing total heat by area
        if total_area <= 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Total area of selected triangles is zero or negative")
            return
            
        heat_flux = total_heat / total_area
        
        # Store in total_heat_sources instead of heat_sources
        self.thermal_loads["total_heat_sources"].append((nodes, total_heat))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_heat_source(nodes, heat_flux, is_total_heat=True, total_heat_value=total_heat)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_LivVar("thermal_loads.heat_sources", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Show confirmation message
        self.parent.message_text.append(
            f"Applied total heat of {total_heat} W to {len(nodes)} triangles"
        )


    def apply_convection(self):
        """Apply convection to selected nodes"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No surface selected")
            return
        
        # Get nodes for the selected triangles
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        h_coeff = self.conv_coeff_spin.value()
        ambient_temp = self.ambient_temp_spin.value()
        
        # Add to thermal loads
        self.thermal_loads["convection"].append((nodes, h_coeff, ambient_temp))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_convection(nodes, h_coeff, ambient_temp)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_LivVar("thermal_loads.convection_applied", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Show confirmation message
        self.parent.message_text.append(f"Applied convection with h={h_coeff}W/m²K, T={ambient_temp}°C to {len(nodes)} triangles")

    def apply_radiation(self):
        """Apply radiation to selected nodes"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No surface selected")
            return
        
        # Get nodes for the selected triangles
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        emissivity = self.emissivity_spin.value()
        
        # Get radiation ambient temperature with unit conversion
        if self.parent.settings.temperature_unit == "Celsius":
            ambient_temp = self.rad_ambient_temp_spin.value() + 273.15  # Convert to Kelvin
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            ambient_temp = (self.rad_ambient_temp_spin.value() - 32) * 5/9 + 273.15  # Convert to Kelvin
        else:
            ambient_temp = self.rad_ambient_temp_spin.value()  # Already in Kelvin
        
        # Add to thermal loads
        self.thermal_loads["radiation"].append((nodes, emissivity, ambient_temp))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_radiation(nodes, emissivity, ambient_temp)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_LivVar("thermal_loads.radiation_applied", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Display temperature in user-specified units
        if self.parent.settings.temperature_unit == "Celsius":
            display_temp = self.rad_ambient_temp_spin.value()
            unit = "°C"
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            display_temp = self.rad_ambient_temp_spin.value()
            unit = "°F"
        else:
            display_temp = ambient_temp
            unit = "K"
        
        # Show confirmation message
        self.parent.message_text.append(f"Applied radiation with emissivity={emissivity}, T={display_temp}{unit} to {len(nodes)} triangles")

    def apply_internal_heat(self):
        """Apply internal heat generation to selected elements"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        selected_triangles = [i for i, highlight in enumerate(self.parent.stl_geom.tri_highlight) if highlight]
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "Error", "No volume selected")
            return
        
        # Get nodes for the selected triangles (these will be used to select volume elements)
        nodes = selected_triangles  # This will be processed to actual nodes during meshing
        heat_generation = self.internal_heat_value_spin.value()
        
        # Add to thermal loads
        self.thermal_loads["internal_heat"].append((nodes, heat_generation))
        
        # Update the parent's thermal loads
        self.parent.thermal_loads = self.thermal_loads
        
        # Create visualization
        self.visualize_internal_heat(nodes, heat_generation)
        
        # Clear selection after applying
        self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
        self.parent.update_highlights()
        
        # Update parent's LivVar and UI
        self.parent.update_LivVar("thermal_loads.applied", True)
        self.parent.update_LivVar("thermal_loads.internal_heat_applied", True)
        self.parent.update_button_icon("Thermal Loads", "check")
        
        # Show confirmation message
        self.parent.message_text.append(f"Applied internal heat generation of {heat_generation}W/m³ to {len(nodes)} triangles")

    def visualize_radiation(self, nodes, emissivity, ambient_temp):
        """Visualize radiation on the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
                
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0, 0.5)  # Purple for radiation
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.radiation_actors.append(actor)
        
        # Convert from internal Kelvin to display unit for ambient temperature
        if self.parent.settings.temperature_unit == "Celsius":
            display_temp = ambient_temp - 273.15
            unit = "°C"
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            display_temp = (ambient_temp - 273.15) * 9/5 + 32
            unit = "°F"
        else:
            display_temp = ambient_temp
            unit = "K"
        
        # Add text label with radiation properties
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"ε = {emissivity}, T = {display_temp:.1f}{unit}")
        text_actor.SetPosition(10, 70)
        text_actor.GetTextProperty().SetColor(0.5, 0, 0.5)  # Purple text
        text_actor.GetTextProperty().SetFontSize(14)
        self.parent.renderer.AddActor2D(text_actor)
        self.radiation_actors.append(text_actor)
        
        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()

    def visualize_internal_heat(self, nodes, heat_generation):
        """Visualize internal heat generation in the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
                
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.5, 0)  # Orange for internal heat
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.internal_heat_actors.append(actor)
        
        # Add text label with heat generation
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Q = {heat_generation}W/m³")
        text_actor.SetPosition(10, 90)
        text_actor.GetTextProperty().SetColor(1, 0.5, 0)  # Orange text
        text_actor.GetTextProperty().SetFontSize(14)
        self.parent.renderer.AddActor2D(text_actor)
        self.internal_heat_actors.append(text_actor)
        
        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()

    def visualize_fixed_temp(self, nodes, temperature):
        """Visualize fixed temperature on the model with arrows"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
                
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # Blue for fixed temperature
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.parent.fixed_temp_actors.append(actor)
        
        # Convert from internal Kelvin to display unit
        if self.parent.settings.temperature_unit == "Celsius":
            display_temp = temperature - 273.15
            unit = "°C"
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            display_temp = (temperature - 273.15) * 9/5 + 32
            unit = "°F"
        else:  # Kelvin
            display_temp = temperature
            unit = "K"
        
        # Get bounding box for scaling
        if hasattr(self.parent.stl_geom, 'get_bounding_box'):
            bbox = self.parent.stl_geom.get_bounding_box()
        else:
            # Calculate bounding box from mesh
            vertices = self.parent.stl_geom.mesh.vectors.reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Calculate model size for arrow scaling
        geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        scale_factor = 0.08 * geom_size  # Scale arrows based on model size
        
        # Determine how many arrows to show
        MAX_MARKERS = 5
        THRESHOLD = 25
        
        if len(nodes) > THRESHOLD:
            step = len(nodes) // MAX_MARKERS
            display_indices = range(0, len(nodes), step)[:MAX_MARKERS]
            display_nodes = [nodes[i] for i in display_indices]
        else:
            display_nodes = nodes
        
        # Flag to indicate if we've added text display for one arrow
        text_label_added = False
        
        # Add arrows directly instead of in a separate function to ensure they're created
        for idx in display_nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                # Get triangle data
                triangle = None
                if hasattr(self.parent.stl_geom, 'get_triangle_data'):
                    triangle = self.parent.stl_geom.get_triangle_data(idx)
                else:
                    #Will remove this once functionality added for Bounding Box to test if it works no need to calculate evrytime even if its in else condition
                    # Manually calculate triangle data
                    vertices = self.parent.stl_geom.mesh.vectors[idx]
                    # Calculate center
                    center = [
                        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
                        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3,
                        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3
                    ]
                    
                    # Calculate normal using cross product
                    v1 = [vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], vertices[1][2] - vertices[0][2]]
                    v2 = [vertices[2][0] - vertices[0][0], vertices[2][1] - vertices[0][1], vertices[2][2] - vertices[0][2]]
                    
                    # Cross product
                    normal = [
                        v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]
                    ]
                    
                    # Normalize
                    length = sum(n*n for n in normal) ** 0.5
                    if length > 0:
                        normal = [n/length for n in normal]
                    else:
                        normal = [0, 0, 1]  # Default if calculation fails
                        
                    triangle = {'center': center, 'normal': normal, 'index': idx}
                
                if triangle:
                    # Create arrow for visualization
                    arrow = vtk.vtkArrowSource()
                    arrow.SetTipLength(0.3)
                    arrow.SetTipRadius(0.1)
                    arrow.SetShaftRadius(0.03)
                    
                    # For temperature, we want arrows pointing inward (same as heat flux)
                    normal = [-n for n in triangle['normal']]
                    
                    # Calculate arrow start position - move back along the inverted normal
                    # This positions the arrow behind the surface with its tip at the surface
                    arrow_length = scale_factor
                    
                    # Calculate start position (move back from center so tip is at the center)
                    start_pos = [
                        triangle['center'][0] - normal[0] * arrow_length,
                        triangle['center'][1] - normal[1] * arrow_length,
                        triangle['center'][2] - normal[2] * arrow_length
                    ]
                    
                    # Create transform
                    transform = vtk.vtkTransform()
                    
                    # Position transform at the starting position
                    transform.Translate(start_pos)
                    
                    # Handle special cases first for better numerical stability
                    if abs(normal[0]) > 0.999:  # Almost parallel to X axis
                        if normal[0] < 0:
                            transform.RotateY(180)
                    elif abs(normal[1]) > 0.999:  # Almost parallel to Y axis
                        # Rotate 90 deg around Z
                        if normal[1] > 0:
                            transform.RotateZ(90)
                        else:
                            transform.RotateZ(-90)
                    elif abs(normal[2]) > 0.999:  # Almost parallel to Z axis
                        # Rotate around Y to point along Z
                        if normal[2] > 0:
                            transform.RotateY(-90)
                        else:
                            transform.RotateY(90)
                    else:
                        # General case - first rotate in XY plane (Z rotation)
                        angle_z = math.degrees(math.atan2(normal[1], normal[0]))
                        transform.RotateZ(angle_z)
                        
                        # Then rotate to correct elevation (Y rotation)
                        # Project normal onto XY plane after Z rotation
                        length_xy = math.sqrt(normal[0]**2 + normal[1]**2)
                        angle_y = math.degrees(math.atan2(normal[2], length_xy))
                        transform.RotateY(angle_y)
                    
                    # Apply scaling
                    transform.Scale(scale_factor, scale_factor, scale_factor)
                    
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(arrow.GetOutputPort())
                    
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(mapper)
                    arrow_actor.SetUserTransform(transform)
                    
                    # Make arrows distinct - use different blue for better visibility
                    arrow_actor.GetProperty().SetColor(0.2, 0.2, 1.0)  # Bright blue for temperature arrows
                    # Ensure the arrow is not transparent
                    arrow_actor.GetProperty().SetOpacity(1.0)
                    
                    # Add to renderer and store
                    self.parent.renderer.AddActor(arrow_actor)
                    self.parent.fixed_temp_actors.append(arrow_actor) #(look at this afterwards)
                    
                    # Add text display next to one of the arrows (first arrow only)
                    if not text_label_added:
                        # Calculate position for text - offset from the arrow but closer
                        text_offset = 0.08 * geom_size  # Reduced offset to bring text closer
                        
                        # Place text near the arrow, offset in a good direction for visibility
                        text_pos = [
                            triangle['center'][0] + text_offset * (-0.5 if normal[0] < 0 else 0.5),
                            triangle['center'][1] + text_offset * (-0.5 if normal[1] < 0 else 0.5),
                            triangle['center'][2] + text_offset * (0.2 if normal[2] < 0 else -0.2)
                        ]
                        
                        # Create a vtkCaptionActor2D for 3D text
                        caption_actor = vtk.vtkCaptionActor2D()
                        
                        # Set text content
                        caption_text = f"T = {display_temp:.1f}{unit}"
                        
                        caption_actor.SetCaption(caption_text)
                        caption_actor.SetAttachmentPoint(text_pos)
                        
                        # Customize text appearance
                        caption_actor.BorderOff()
                        caption_actor.LeaderOff() 
                        
                        # Set text properties
                        text_prop = caption_actor.GetCaptionTextProperty()
                        text_prop.SetColor(0.2, 0.2, 1.0) 
                        text_prop.SetFontSize(5)

                        # And add these text scaling properties
                        caption_actor.SetWidth(0.15)
                        caption_actor.SetHeight(0.05)
                        text_prop.SetBold(True)
                        text_prop.SetShadow(True)
                        
                        # ... and similar for caption_actor ...
                        self.parent.renderer.AddActor(caption_actor)
                        self.parent.fixed_temp_actors.append(caption_actor)  # Use parent's list
                        
                        text_label_added = True
        
        # Force rendering update
        self.parent.vtkWidget.GetRenderWindow().Render()
    
    def visualize_heat_source(self, nodes, heat_value, is_total_heat=False, total_heat_value=None):
        """Visualize heat source on the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
        
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)

        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Use slightly different color for total heat vs. heat flux
        if is_total_heat:
            actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange for total heat
        else:
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red for heat flux
        
        actor.GetProperty().SetOpacity(0.7)

        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        if is_total_heat:
            if not hasattr(self.parent, 'total_heat_actors'):
                self.parent.total_heat_actors = []
            self.parent.total_heat_actors.append(actor)  # Use separate actor list for total heat
        else:
            self.parent.heat_source_actors.append(actor)

        # Add heat direction visualization - pass the values
        self.visualize_heat_arrows(nodes, is_total_heat, heat_value, total_heat_value)

        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()

    def visualize_heat_arrows(self, nodes, is_total_heat=False, heat_value=None, total_heat_value=None):
        """Visualize arrows starting behind the surface with tips at the triangle surface"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
    
        # Determine how many arrows to show
        MAX_MARKERS = 5
        THRESHOLD = 25
    
        if len(nodes) > THRESHOLD:
            step = len(nodes) // MAX_MARKERS
            display_indices = range(0, len(nodes), step)[:MAX_MARKERS]
            display_nodes = [nodes[i] for i in display_indices]
        else:
            display_nodes = nodes
    
        # Get bounding box for scaling
        if hasattr(self.parent.stl_geom, 'get_bounding_box'):
            bbox = self.parent.stl_geom.get_bounding_box()
        else:
            # Calculate bounding box from mesh
            vertices = self.parent.stl_geom.mesh.vectors.reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
    
        # Calculate model size for arrow scaling
        geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        scale_factor = 0.08 * geom_size  # Scale arrows based on model size
        
        # Flag to indicate if we've added text display for one arrow
        text_label_added = False
    
        for idx in display_nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                # Get triangle data
                triangle = None
                if hasattr(self.parent.stl_geom, 'get_triangle_data'):
                    triangle = self.parent.stl_geom.get_triangle_data(idx)
                else:
                    # Manually calculate if get_triangle_data doesn't exist
                    vertices = self.parent.stl_geom.mesh.vectors[idx]
                    # Calculate center
                    center = [
                        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
                        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3,
                        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3
                    ]
                
                    # Calculate normal using cross product
                    v1 = [vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], vertices[1][2] - vertices[0][2]]
                    v2 = [vertices[2][0] - vertices[0][0], vertices[2][1] - vertices[0][1], vertices[2][2] - vertices[0][2]]
                
                    # Cross product
                    normal = [
                        v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]
                    ]
                
                    # Normalize
                    length = sum(n*n for n in normal) ** 0.5
                    if length > 0:
                        normal = [n/length for n in normal]
                    else:
                        normal = [0, 0, 1]  # Default if calculation fails
                    
                    triangle = {'center': center, 'normal': normal, 'index': idx}
            
                if triangle:
                    # Create arrow for visualization
                    arrow = vtk.vtkArrowSource()
                    arrow.SetTipLength(0.3)
                    arrow.SetTipRadius(0.1)
                    arrow.SetShaftRadius(0.03)
                
                    # Get the normal - invert it since we want arrows pointing inward
                    normal = [-n for n in triangle['normal']]
                
                    # Calculate arrow start position - move back along the inverted normal
                    # This positions the arrow behind the surface with its tip at the surface
                    arrow_length = scale_factor
                
                    # Calculate start position (move back from center so tip is at the center)
                    start_pos = [
                        triangle['center'][0] - normal[0] * arrow_length,
                        triangle['center'][1] - normal[1] * arrow_length,
                        triangle['center'][2] - normal[2] * arrow_length
                    ]
                
                    # Create transform
                    transform = vtk.vtkTransform()
                
                    # Position transform at the starting position
                    transform.Translate(start_pos)
                
                    # Handle special cases first for better numerical stability
                    if abs(normal[0]) > 0.999:  # Almost parallel to X axis
                        if normal[0] < 0:
                            transform.RotateY(180)
                    elif abs(normal[1]) > 0.999:  # Almost parallel to Y axis
                        # Rotate 90 deg around Z then maybe 180 around X
                        if normal[1] > 0:
                            transform.RotateZ(90)
                        else:
                            transform.RotateZ(-90)
                    elif abs(normal[2]) > 0.999:  # Almost parallel to Z axis
                        # Rotate around Y to point along Z
                        if normal[2] > 0:
                            transform.RotateY(-90)
                        else:
                            transform.RotateY(90)
                    else:
                        # General case - first rotate in XY plane (Z rotation)
                        angle_z = math.degrees(math.atan2(normal[1], normal[0]))
                        transform.RotateZ(angle_z)
                    
                        # Then rotate to correct elevation (Y rotation)
                        # Project normal onto XY plane after Z rotation
                        length_xy = math.sqrt(normal[0]**2 + normal[1]**2)
                        angle_y = math.degrees(math.atan2(normal[2], length_xy))
                        transform.RotateY(angle_y)
                
                    # Apply scaling
                    transform.Scale(scale_factor, scale_factor, scale_factor)
                
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(arrow.GetOutputPort())
                
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.SetUserTransform(transform)
                
                    # Set color based on heat type
                    if is_total_heat:
                        actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange for total heat
                    else:
                        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red for heat flux
                
                   # Add to renderer and store in parent's lists
                    self.parent.renderer.AddActor(actor)
                    if is_total_heat:
                        if not hasattr(self.parent, 'total_heat_actors'):
                            self.parent.total_heat_actors = []
                        self.parent.total_heat_actors.append(actor)
                    else:
                        self.parent.heat_source_actors.append(actor)
                    
                    # Add text display next to one of the arrows (first arrow only)
                    if not text_label_added and heat_value is not None:
                        # Calculate position for text - offset from the arrow but closer
                        text_offset = 0.08 * geom_size  # Reduced offset to bring text closer
                        
                        # Place text near the arrow, offset in a good direction for visibility
                        text_pos = [
                            triangle['center'][0] + text_offset * (-0.5 if normal[0] < 0 else 0.5),
                            triangle['center'][1] + text_offset * (-0.5 if normal[1] < 0 else 0.5),
                            triangle['center'][2] + text_offset * (0.2 if normal[2] < 0 else -0.2)
                        ]
                        
                        # Create a vtkCaptionActor2D for 3D text
                        caption_actor = vtk.vtkCaptionActor2D()
                        
                        # Set text content
                        if is_total_heat:
                            caption_text = f"{total_heat_value} W"
                        else:
                            caption_text = f"{heat_value} W/m²"
                        
                        caption_actor.SetCaption(caption_text)
                        caption_actor.SetAttachmentPoint(text_pos)
                        
                        # Customize text appearance
                        caption_actor.BorderOff()
                        caption_actor.LeaderOff() 
                        
                        # Set text properties
                        text_prop = caption_actor.GetCaptionTextProperty()
                        if is_total_heat:
                            text_prop.SetColor(1.0, 0.5, 0.0)  # Orange text for total heat
                        else:
                            text_prop.SetColor(1.0, 0.0, 0.0)  # Red text for heat flux
                        text_prop.SetFontSize(5)
                        # And add these text scaling properties
                        caption_actor.SetWidth(0.15)
                        caption_actor.SetHeight(0.05)
                        text_prop.SetBold(True)
                        text_prop.SetShadow(True)
                        
                        #for caption_actor ...
                        self.parent.renderer.AddActor(caption_actor)
                        if is_total_heat:
                            self.parent.total_heat_actors.append(caption_actor)
                        else:
                            self.parent.heat_source_actors.append(caption_actor)  
                        
                        text_label_added = True
                        
    
    def visualize_convection(self, nodes, h_coeff, ambient_temp):
        """Visualize convection on the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
            
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 1, 0)  # Green for convection
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.convection_actors.append(actor)
        
        # Add text label with convection properties
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"h = {h_coeff}W/m²K, T = {ambient_temp}°C")
        text_actor.SetPosition(10, 50)
        text_actor.GetTextProperty().SetColor(0, 1, 0)  # Green text
        text_actor.GetTextProperty().SetFontSize(14)
        self.parent.renderer.AddActor2D(text_actor)
        self.parent.convection_actors.append(text_actor)
        
        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()

#---------------------------------------------------------------------------------
class ThermalTopOptWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize Topology")
        self.resize(300, 250)
        self.parent = parent
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Volume Fraction
        vol_frac_layout = QtWidgets.QHBoxLayout()
        vol_frac_label = QtWidgets.QLabel("Desired Vol.Frac.")
        self.vol_frac_spin = QtWidgets.QDoubleSpinBox()
        self.vol_frac_spin.setRange(0.1, 1.0)
        self.vol_frac_spin.setValue(0.50)
        self.vol_frac_spin.setDecimals(2)
        self.vol_frac_spin.setSingleStep(0.05)
        vol_frac_layout.addWidget(vol_frac_label)
        vol_frac_layout.addWidget(self.vol_frac_spin)
        layout.addLayout(vol_frac_layout)
        
        # Volume Step
        vol_step_layout = QtWidgets.QHBoxLayout()
        vol_step_label = QtWidgets.QLabel("Volume Step")
        self.vol_step_spin = QtWidgets.QDoubleSpinBox()
        self.vol_step_spin.setRange(0.001, 0.1)
        self.vol_step_spin.setValue(0.02500)
        self.vol_step_spin.setDecimals(5)
        self.vol_step_spin.setSingleStep(0.005)
        vol_step_layout.addWidget(vol_step_label)
        vol_step_layout.addWidget(self.vol_step_spin)
        layout.addLayout(vol_step_layout)
        
        # Save Intermediate
        self.save_intermediate = QtWidgets.QCheckBox("Save Intermediate?")
        layout.addWidget(self.save_intermediate)
        
        # Add some vertical spacing
        layout.addSpacing(20)
        
        # Action buttons
        self.optimize_button = QtWidgets.QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.start_optimization)
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
        """)
        layout.addWidget(self.optimize_button)
        
        self.stop_button = QtWidgets.QPushButton("STOP OPTIMIZATION!")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #dcdcdc;
                padding: 5px;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #a0a0a0;
            }
        """)
        layout.addWidget(self.stop_button)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        # Initialize optimization state
        self.optimization_running = False
        
    def start_optimization(self):
        """Start the thermal topology optimization process"""
        if not self.check_prerequisites():
            return
            
        optimization_params = {
            'volume_fraction': self.vol_frac_spin.value(),
            'volume_step': self.vol_step_spin.value(),
            'save_intermediate': self.save_intermediate.isChecked()
        }
        
        # Update UI state
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Store optimization parameters in parent window
        if hasattr(self.parent, 'thermal_optimization_params'):
            self.parent.thermal_optimization_params = optimization_params
            
        self.parent.message_text.append("Starting thermal topology optimization...")
        
        # Update sidebar button icon
        if hasattr(self.parent, 'update_button_icon'):
            self.parent.update_button_icon("Thermal TopOpt", "check")
        
         # Update LivVar state
        self.parent.update_LivVar('topopt.thermal_performed', True)

        # Here you would typically start the actual thermal optimization process
        self.parent.message_text.append(f"Thermal optimization parameters set:")
        self.parent.message_text.append(f"Volume Fraction: {optimization_params['volume_fraction']}")
        self.parent.message_text.append(f"Volume Step: {optimization_params['volume_step']}")
        
    def stop_optimization(self):
        """Stop the ongoing optimization process"""
        if self.optimization_running:
            self.optimization_running = False
            self.optimize_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.parent.message_text.append("Thermal optimization stopped by user")
            
    def check_prerequisites(self):
        """Check if all required conditions are met for thermal optimization"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return False
            
        if not hasattr(self.parent, 'material_data') or self.parent.material_data is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No material assigned")
            return False
            
        if not hasattr(self.parent, 'topopt_constraints') or self.parent.topopt_constraints is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No optimization constraints defined")
            return False
            
        return True

#---------------------------------------------------------------------------------
# class TopologyOptimizationVisualization:
#     """Class for visualizing topology optimization results in the PyTO GUI."""
    
#     def __init__(self, parent=None):
#         """Initialize visualization class with parent window reference."""
#         self.parent = parent
#         self.history = None
#         self.mesh = None
#         self.optimization_complete = False
#         self.threshold = 0.5  # Default density threshold for visualization
        
#     def display_results(self, fe_solver, history, elapsed_time):
#         """Display optimization results in the main window.
        
#         Args:
#             fe_solver: The FEA solver with optimized mesh
#             history: Dictionary containing optimization history
#             elapsed_time: Total optimization time in seconds
#         """
#         self.history = history
#         self.mesh = fe_solver.mesh
#         self.optimization_complete = True
        
#         # Update status in UI
#         self.parent.message_text.append(f"Topology optimization complete in {elapsed_time:.2f} seconds")
#         self.parent.message_text.append(f"Final volume fraction: {history['volume'][-1]:.4f}")
#         self.parent.message_text.append(f"Final compliance: {history['compliance'][-1]:.4f}")
        
#         # Update visualization in main window viewport
#         self.display_optimized_mesh()
        
#         # Create a history plot if available
#         if self.parent.enable_plot_window and len(history['volume']) > 1:
#             self.plot_optimization_history()
            
#     def display_optimized_mesh(self):
#         """Display the optimized mesh in the main window viewport."""
#         if not self.optimization_complete or self.mesh is None:
#             return
        
#         # Clear previous visualization actors
#         if hasattr(self.parent, 'optimized_mesh_actor') and self.parent.optimized_mesh_actor:
#             self.parent.renderer.RemoveActor(self.parent.optimized_mesh_actor)
            
#         # Convert the density field to a VTK mesh for visualization
#         import vtk
#         import numpy as np
        
#         # Get element densities
#         densities = np.array(self.mesh.densities)
        
#         # Create geometry - use isosurface or direct density visualization
#         if hasattr(self.parent, 'smooth_surface') and self.parent.smooth_surface:
#             # Create isosurface visualization (smoother)
#             self.create_isosurface_visualization(densities)
#         else:
#             # Create direct element visualization (blocky)
#             self.create_element_visualization(densities)
            
#         # Update render window
#         self.parent.vtkWidget.GetRenderWindow().Render()
        
#     def create_isosurface_visualization(self, densities):
#         """Create a smooth isosurface visualization of the optimized topology.
        
#         Args:
#             densities: Array of element densities
#         """
#         import vtk
#         import numpy as np
        
#         # Create structured points for the density field
#         nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        
#         # Create image data
#         image_data = vtk.vtkImageData()
#         image_data.SetDimensions(nx+1, ny+1, nz+1)
#         image_data.SetOrigin(self.mesh.x_min, self.mesh.y_min, self.mesh.z_min)
#         image_data.SetSpacing(
#             (self.mesh.x_max - self.mesh.x_min) / nx,
#             (self.mesh.y_max - self.mesh.y_min) / ny,
#             (self.mesh.z_max - self.mesh.z_min) / nz
#         )
        
#         # Assign density values to cell data
#         cell_data = vtk.vtkFloatArray()
#         cell_data.SetName("Density")
#         cell_data.SetNumberOfComponents(1)
#         cell_data.SetNumberOfTuples(nx * ny * nz)
        
#         for i in range(len(densities)):
#             cell_data.SetValue(i, densities[i])
            
#         image_data.GetCellData().AddArray(cell_data)
#         image_data.GetCellData().SetActiveScalars("Density")
        
#         # Convert cell data to point data for smoother isosurface
#         cell_to_point = vtk.vtkCellDataToPointData()
#         cell_to_point.SetInputData(image_data)
#         cell_to_point.PassCellDataOn()
#         cell_to_point.Update()
        
#         # Create isosurface using contour filter
#         contour = vtk.vtkContourFilter()
#         contour.SetInputConnection(cell_to_point.GetOutputPort())
#         contour.SetValue(0, self.threshold)  # Isovalue = threshold
#         contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Density")
#         contour.Update()
        
#         # Clean up mesh and generate normals
#         clean = vtk.vtkCleanPolyData()
#         clean.SetInputConnection(contour.GetOutputPort())
        
#         normals = vtk.vtkPolyDataNormals()
#         normals.SetInputConnection(clean.GetOutputPort())
#         normals.SetFeatureAngle(60)
#         normals.SplittingOff()
        
#         # Create mapper and actor
#         mapper = vtk.vtkPolyDataMapper()
#         mapper.SetInputConnection(normals.GetOutputPort())
        
#         self.parent.optimized_mesh_actor = vtk.vtkActor()
#         self.parent.optimized_mesh_actor.SetMapper(mapper)
#         self.parent.optimized_mesh_actor.GetProperty().SetColor(0.2, 0.7, 1.0)
        
#         # Add to renderer
#         self.parent.renderer.AddActor(self.parent.optimized_mesh_actor)
        
#     def create_element_visualization(self, densities):
#         """Create a direct element-based visualization of the optimized topology.
        
#         Args:
#             densities: Array of element densities
#         """
#         import vtk
#         import numpy as np
        
#         # Create unstructured grid from mesh elements
#         points = vtk.vtkPoints()
#         for i in range(len(self.mesh.nodeCoords)):
#             x, y, z = self.mesh.nodeCoords[i]
#             points.InsertNextPoint(x, y, z)
            
#         # Create grid
#         grid = vtk.vtkUnstructuredGrid()
#         grid.SetPoints(points)
        
#         # Add cells (elements) with density above threshold
#         density_array = vtk.vtkFloatArray()
#         density_array.SetName("Density")
#         density_array.SetNumberOfComponents(1)
        
#         for i in range(len(densities)):
#             if densities[i] > self.threshold:
#                 # Get element nodes
#                 elem_nodes = self.mesh.elemArray[i]
                
#                 # Create hexahedron (for hex elements)
#                 hex_elem = vtk.vtkHexahedron()
#                 for j in range(len(elem_nodes)):
#                     hex_elem.GetPointIds().SetId(j, elem_nodes[j])
                    
#                 # Add to grid
#                 grid.InsertNextCell(hex_elem.GetCellType(), hex_elem.GetPointIds())
#                 density_array.InsertNextValue(densities[i])
                
#         # Add density data
#         grid.GetCellData().AddArray(density_array)
#         grid.GetCellData().SetActiveScalars("Density")
        
#         # Create mapper and actor
#         mapper = vtk.vtkDataSetMapper()
#         mapper.SetInputData(grid)
#         mapper.ScalarVisibilityOff()
        
#         self.parent.optimized_mesh_actor = vtk.vtkActor()
#         self.parent.optimized_mesh_actor.SetMapper(mapper)
#         self.parent.optimized_mesh_actor.GetProperty().SetColor(0.2, 0.7, 1.0)
        
#         # Add to renderer
#         self.parent.renderer.AddActor(self.parent.optimized_mesh_actor)
    
#     def plot_optimization_history(self):
#         """Create a plot of the optimization history."""
#         import matplotlib.pyplot as plt
#         from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
#         from PyQt5.QtWidgets import QDialog, QVBoxLayout
        
#         # Create figure
#         fig = plt.figure(figsize=(8, 6))
        
#         # Plot compliance history
#         plt.subplot(2, 1, 1)
#         plt.plot(self.history['compliance'], 'bo-')
#         plt.ylabel('Compliance')
#         plt.title('Topology Optimization History')
#         plt.grid(True)
        
#         # Plot volume history
#         plt.subplot(2, 1, 2)
#         plt.plot(self.history['volume'], 'ro-')
#         plt.xlabel('Iteration')
#         plt.ylabel('Volume Fraction')
#         plt.grid(True)
        
#         # Create dialog
#         dialog = QDialog(self.parent)
#         dialog.setWindowTitle("Optimization History")
#         dialog.resize(800, 600)
        
#         # Create canvas for matplotlib figure
#         canvas = FigureCanvasQTAgg(fig)
        
#         # Add to layout
#         layout = QVBoxLayout(dialog)
#         layout.addWidget(canvas)
        
#         # Show dialog
#         dialog.show()
    
#     def update_optimization_progress(self, iteration, total_iterations, volume, compliance):
#         """Update the visualization during optimization (for interactive feedback).
        
#         Args:
#             iteration: Current iteration number
#             total_iterations: Total expected iterations
#             volume: Current volume fraction
#             compliance: Current compliance value
#         """
#         # Update progress bar if available
#         if hasattr(self.parent, 'progress_bar'):
#             progress = int(100 * iteration / total_iterations)
#             self.parent.progress_bar.setValue(progress)
        
#         # Update status message
#         self.parent.message_text.append(f"Iteration {iteration}: vol={volume:.4f}, compliance={compliance:.4f}")
        
#         # Update visualization once every few iterations (not every iteration to avoid slowdowns)
#         if iteration % 5 == 0 and hasattr(self.parent, 'mesh'):
#             self.display_optimized_mesh()
    
#     def export_optimized_model(self, filename):
#         """Export the optimized model to an STL file.
        
#         Args:
#             filename: Path to save the STL file
#         """
#         if not self.optimization_complete or self.mesh is None:
#             self.parent.message_text.append("No optimized model available to export")
#             return False
            
#         import vtk
        
#         # Get the current visualization actor
#         if hasattr(self.parent, 'optimized_mesh_actor') and self.parent.optimized_mesh_actor:
#             # Create STL writer
#             writer = vtk.vtkSTLWriter()
#             writer.SetFileName(filename)
            
#             # Get the mapper's input
#             mapper = self.parent.optimized_mesh_actor.GetMapper()
            
#             # If it's a dataset mapper, we need to get its input data
#             if isinstance(mapper, vtk.vtkDataSetMapper):
#                 # Convert unstructured grid to polydata
#                 geom_filter = vtk.vtkGeometryFilter()
#                 geom_filter.SetInputData(mapper.GetInput())
#                 geom_filter.Update()
#                 writer.SetInputData(geom_filter.GetOutput())
#             else:
#                 # It's already a polydata mapper
#                 writer.SetInputData(mapper.GetInput())
                
#             writer.Write()
#             self.parent.message_text.append(f"Optimized model exported to {filename}")
#             return True
        
#         self.parent.message_text.append("No visualization available to export")
#         return False

#---------------------------------------------------------------------------------
class ProjectData:
    def __init__(self):
        self.version = "2025.01"
        self.stl_file_path = None
        self.settings = None
        self.material_data = None
        self.structuralBC = None
        self.thermalBC = None
        self.topopt_constraints = None
        self.optimization_params = None
        self.analysis_settings = {
            "n_elements": 10000,  # Default value
            "solver_type": "PARDISO"  # Default solver
        }
    
    def to_dict(self):
        return {
            'version': self.version,
            'stl_file_path': self.stl_file_path,
            'settings': {
                'unit_system': self.settings.unit_system,
                'temperature_unit': self.settings.temperature_unit,
                'angle_unit': self.settings.angle_unit
            } if self.settings else None,
            'material_data': self.material_data,
            'structuralBC': self.structuralBC,
            'thermalBC': self.thermalBC,
            'topopt_constraints': self.topopt_constraints,
            'optimization_params': self.optimization_params,
            'analysis_settings': self.analysis_settings
        }
    
    @classmethod
    def from_dict(cls, data):
        project = cls()
        project.version = data.get('version', "2025.01")
        project.stl_file_path = data.get('stl_file_path')
        
        if data.get('settings'):
            settings = Settings()
            settings.update_settings(
                data['settings']['unit_system'],
                data['settings']['temperature_unit'],
                data['settings']['angle_unit']
            )
            project.settings = settings
        
        project.structuralBC = data.get('structuralBC', [])
        
        # Make a copy of the thermal BC to avoid modifying the original data
        thermalBC = data.get('thermalBC', {}).copy() if data.get('thermalBC') else {}
        
        # Ensure all expected keys exist
        if 'total_heat_sources' not in thermalBC:
            thermalBC['total_heat_sources'] = []
        
        # If thermal_counts exists, ensure it has the total_heat_sources field
        if 'thermal_counts' in thermalBC and 'total_heat_sources' not in thermalBC['thermal_counts']:
            thermalBC['thermal_counts']['total_heat_sources'] = 0
        
        project.thermalBC = thermalBC
        project.material_data = data.get('material_data')
        
        # Load topopt constraints and optimization parameters
        project.topopt_constraints = data.get('topopt_constraints')
        project.optimization_params = data.get('optimization_params')
        
        # Load analysis settings with defaults if not present
        if data.get('analysis_settings'):
            project.analysis_settings = data.get('analysis_settings')
        else:
            project.analysis_settings = {
                "n_elements": 10000,
                "solver_type": "PARDISO"
            }
        
        return project
#---------------------------------------------------------------------------------
class ProjectsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Projects")
        self.resize(200, 150)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        
        save_button = QtWidgets.QPushButton("Save Project")
        save_button.clicked.connect(self.save_project)  
        layout.addWidget(save_button)
        
        load_button = QtWidgets.QPushButton("Load Project")
        load_button.clicked.connect(self.load_project) 
        layout.addWidget(load_button)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def save_project(self):
        if not self.parent.stl_geom: 
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry loaded to save")
            return
                
        project = ProjectData()
        project.stl_file_path = os.path.basename(self.parent.stl_geom.file_path)
        project.settings = self.parent.settings
        project.material_data = self.parent.material_data

        # Set analysis settings from parent if they exist
        if hasattr(self.parent, 'analysis_settings'):
            project.analysis_settings = self.parent.analysis_settings
        
        # Create a structuralBC dictionary with organized data
        structuralBC = {}
        
        # Add load forces at the top
        if hasattr(self.parent, 'load_forces'):
            load_forces_list = []
            for force in self.parent.load_forces:
                if hasattr(force, 'tolist'):  
                    force = force.tolist()
                else:
                    force = [float(f) for f in force] 
                load_forces_list.append(force)
            structuralBC['load_forces'] = load_forces_list
        
        # Add counts
        fixed_count = 0
        if hasattr(self.parent, 'fixed_faces'):
            fixed_count += len(self.parent.fixed_faces)
        if hasattr(self.parent, 'fixed_faces_x'):
            fixed_count += len(self.parent.fixed_faces_x)
        if hasattr(self.parent, 'fixed_faces_y'):
            fixed_count += len(self.parent.fixed_faces_y)
        if hasattr(self.parent, 'fixed_faces_z'):
            fixed_count += len(self.parent.fixed_faces_z)
            
        structuralBC['constraint_counts'] = {
            'fixed_triangles': fixed_count
        }
        
        load_count = 0
        if hasattr(self.parent, 'load_faces_groups'):
            for group in self.parent.load_faces_groups:
                load_count += len(group)
            structuralBC['constraint_counts']['loaded_triangles'] = load_count
        
        # Store triangle indices for fixed constraints
        if hasattr(self.parent, 'fixed_faces'):
            structuralBC['fixed_faces_indices'] = [face['index'] for face in self.parent.fixed_faces]
        if hasattr(self.parent, 'fixed_faces_x'):
            structuralBC['fixed_faces_x_indices'] = [face['index'] for face in self.parent.fixed_faces_x]
        if hasattr(self.parent, 'fixed_faces_y'):
            structuralBC['fixed_faces_y_indices'] = [face['index'] for face in self.parent.fixed_faces_y]
        if hasattr(self.parent, 'fixed_faces_z'):
            structuralBC['fixed_faces_z_indices'] = [face['index'] for face in self.parent.fixed_faces_z]
        
        # Store triangle indices for load groups
        if hasattr(self.parent, 'load_faces_groups'):
            load_faces_indices = []
            for group in self.parent.load_faces_groups:
                group_indices = [face['index'] for face in group]
                load_faces_indices.append(group_indices)
            structuralBC['load_faces_indices'] = load_faces_indices
        
        # Store the boundary conditions
        project.structuralBC = structuralBC

        # Save thermal boundary conditions
        thermalBC = {}
        
        if hasattr(self.parent, 'thermal_loads'):
            # Store fixed temperatures
            if self.parent.thermal_loads.get('fixed_temps'):
                fixed_temps_data = []
                for triangles, temperature in self.parent.thermal_loads['fixed_temps']:
                    fixed_temps_data.append({
                        'triangles': triangles,
                        'temperature': temperature
                    })
                thermalBC['fixed_temps'] = fixed_temps_data
                
            # Store heat sources
            if self.parent.thermal_loads.get('heat_sources'):
                heat_sources_data = []
                for triangles, heat_flux in self.parent.thermal_loads['heat_sources']:
                    heat_sources_data.append({
                        'triangles': triangles,
                        'heat_flux': heat_flux
                    })
                thermalBC['heat_sources'] = heat_sources_data
            
        # Store total heat sources separately
        if self.parent.thermal_loads.get('total_heat_sources'):
            total_heat_data = []
            for triangles, total_heat in self.parent.thermal_loads['total_heat_sources']:
                total_heat_data.append({
                    'triangles': triangles,
                    'total_heat': total_heat
                })
            thermalBC['total_heat_sources'] = total_heat_data
                
            # Store convection data
            if self.parent.thermal_loads.get('convection'):
                convection_data = []
                for triangles, h_coeff, ambient_temp in self.parent.thermal_loads['convection']:
                    convection_data.append({
                        'triangles': triangles,
                        'h_coeff': h_coeff,
                        'ambient_temp': ambient_temp
                    })
                thermalBC['convection'] = convection_data
                
            # Store thermal loads counts
            thermalBC['thermal_counts'] = {
            'fixed_temps': sum(len(data[0]) for data in self.parent.thermal_loads.get('fixed_temps', [])),
            'heat_sources': sum(len(data[0]) for data in self.parent.thermal_loads.get('heat_sources', [])),
            'total_heat_sources': sum(len(data[0]) for data in self.parent.thermal_loads.get('total_heat_sources', [])),
            'convection': sum(len(data[0]) for data in self.parent.thermal_loads.get('convection', []))
        }
        
        project.thermalBC = thermalBC
        project.topopt_constraints = self.parent.topopt_constraints
        project.optimization_params = self.parent.optimization_params

        # Save to file
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "Project Files (*.pyto);;All Files (*)",
            options=options
        )
        
        if file_path:
            if not file_path.endswith('.pyto'):
                file_path += '.pyto'
                    
            try:
                with open(file_path, 'w') as f:
                    json.dump(project.to_dict(), f, indent=4)
                self.parent.message_text.append(f"Project saved successfully to {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def load_project(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project Files (*.pyto);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                project = ProjectData.from_dict(data)

                # Load analysis settings if present
                if hasattr(project, 'analysis_settings'):
                    self.parent.analysis_settings = project.analysis_settings
                    self.parent.message_text.append(f"Loaded analysis settings: {project.analysis_settings.get('n_elements')} elements, solver: {project.analysis_settings.get('solver_type')}")
                
                # Clear current state
                self.parent.clear_selections()
                
                # Prioritize STL loading by trying different path options
                stl_loaded = False
                
                if project.stl_file_path:
                    stl_filename = os.path.basename(project.stl_file_path)
                    current_dir = os.getcwd()
                    project_dir = os.path.dirname(file_path)
                    
                    # Try these paths in order
                    stl_load_attempts = [
                        os.path.join(".", stl_filename),
                        stl_filename,
                        os.path.join(project_dir, stl_filename),
                        os.path.join(os.path.dirname(current_dir), stl_filename),
                        os.path.join(current_dir, "models", stl_filename),
                        os.path.join(current_dir, "stl", stl_filename),
                        os.path.join(current_dir, "data", stl_filename),
                        project.stl_file_path
                    ]

                    for stl_path in stl_load_attempts:
                        try:
                            if os.path.isfile(stl_path):
                                self.parent.load_stl_file(stl_path)
                                stl_loaded = True
                                break
                        except Exception:
                            continue
                    
                    if not stl_loaded:
                        msg_result = QtWidgets.QMessageBox.question(
                            self, 
                            "STL File Not Found", 
                            f"Could not find the STL file: {stl_filename}\nWould you like to locate it manually?",
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                        )
                        
                        if msg_result == QtWidgets.QMessageBox.Yes:
                            stl_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                                self, 
                                "Select STL File", 
                                "", 
                                "STL Files (*.stl);;All Files (*)",
                                options=options
                            )
                            if stl_path:
                                try:
                                    self.parent.load_stl_file(stl_path)
                                    stl_loaded = True
                                except Exception as e:
                                    QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load STL: {str(e)}")
                
                # Load settings
                if project.settings:
                    self.parent.settings = project.settings

                # Load TopOpt constraints if present
                if project.topopt_constraints:
                    self.parent.topopt_constraints = project.topopt_constraints
                    self.restore_topopt_constraints(project.topopt_constraints)
                    self.parent.message_text.append(f"Loaded topology optimization constraints")

                # Also ensure the optimization parameters are loaded if present
                if project.optimization_params:
                    self.parent.optimization_params = project.optimization_params
                    self.parent.message_text.append(f"Loaded optimization parameters")
                    
                # Restore structural boundary conditions if STL is loaded
                if self.parent.stl_geom and project.structuralBC:
                    self.restore_structural_bc(project.structuralBC)

                # After loading STL and structural BCs, load thermal BCs
                if self.parent.stl_geom and hasattr(project, 'thermalBC') and project.thermalBC:
                    self.restore_thermal_bc(project.thermalBC)
                
                # Add material settings restoration
                if project.material_data:
                    self.parent.material_data = project.material_data
                    # Update geometry color based on material
                    if hasattr(self.parent, 'stl_actor'):
                        if project.material_data['name'] == "Custom":
                            self.parent.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                        elif project.material_data['name'] == "AlloySteel":
                            self.parent.stl_actor.GetProperty().SetColor(0.7, 0.7, 0.8)
                        elif project.material_data['name'] == "Aluminum":
                            self.parent.stl_actor.GetProperty().SetColor(0.9, 0.9, 0.9)
                        elif project.material_data['name'] == "Titanium":
                            self.parent.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.7)
                        elif project.material_data['name'] == "StainlessSteel":
                            self.parent.stl_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
                        self.parent.vtkWidget.GetRenderWindow().Render()
                    
                    # Update Material button icon to check
                    self.parent.update_button_icon("Material", "check")
                    self.parent.update_LivVar('material_defined', True)
                
                # Update any other UI states based on what was loaded
                if stl_loaded:
                    self.parent.update_button_icon("Geometry", "check")
                    self.parent.update_LivVar('geometry_loaded', True)
                
                self.parent.message_text.append(f"Project loaded successfully from {file_path}")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def restore_force(self, load_data):
        """Create a force arrow visualization"""
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
        
        self.parent.renderer.AddActor(actor)
        self.parent.force_actors.append(actor)

    def restore_constraint(self, constraint_data):
        """Restore a constraint with its black color for the saved faces"""
        if 'faces' not in constraint_data:
            return
            
        # Create points and cells for the constrained faces
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use the saved face data to recreate the geometry
        for face in constraint_data['faces']:
            vertices = face['vertices']
            point_ids = []
            for v in vertices:
                point_ids.append(points.InsertNextPoint(v))
            tri = vtk.vtkTriangle()
            for i in range(3):
                tri.GetPointIds().SetId(i, point_ids[i])
            cells.InsertNextCell(tri)

        # Create polydata for the constrained faces
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        constraint_actor = vtk.vtkActor()
        constraint_actor.SetMapper(mapper)
        constraint_actor.GetProperty().SetColor(*constraint_data['color'])  # Set black color
        
        # Add to renderer and store in constraint_actors
        self.parent.renderer.AddActor(constraint_actor)
        self.parent.constraint_actors.append(constraint_actor)
        
        # Update display
        self.parent.vtkWidget.GetRenderWindow().Render()

    def restore_structural_bc(self, bc_data):
        """Reconstruct and visualize boundary conditions from saved triangle indices"""
        if not self.parent.stl_geom:
            return
            
        # Output summary info first
        if 'constraint_counts' in bc_data:
            counts = bc_data['constraint_counts']
            if 'fixed_triangles' in counts:
                self.parent.message_text.append(f"Loading {counts['fixed_triangles']} fixed triangles")
            if 'loaded_triangles' in counts:
                self.parent.message_text.append(f"Loading {counts['loaded_triangles']} loaded triangles")
        
        # Process load forces first
        if 'load_forces' in bc_data:
            self.parent.load_forces = bc_data['load_forces']
            
        # Restore fixed constraints
        if 'fixed_faces_indices' in bc_data:
            # Convert indices to face data
            self.parent.fixed_faces = []
            for idx in bc_data['fixed_faces_indices']:
                face_data = self.parent.stl_geom.get_triangle_data(idx)
                if face_data:
                    self.parent.fixed_faces.append(face_data)
            
            # Create constraint for visualization
            constraint_data = {
                'faces': self.parent.fixed_faces,
                'color': (0, 0, 0)  # Black color
            }
            self.restore_constraint(constraint_data)
            
        if 'fixed_faces_x_indices' in bc_data:
            self.parent.fixed_faces_x = []
            for idx in bc_data['fixed_faces_x_indices']:
                face_data = self.parent.stl_geom.get_triangle_data(idx)
                if face_data:
                    self.parent.fixed_faces_x.append(face_data)
            
            constraint_data = {
                'faces': self.parent.fixed_faces_x,
                'color': (0.7, 0.3, 0.3)  # Light red for X constraints
            }
            self.restore_constraint(constraint_data)
            
        if 'fixed_faces_y_indices' in bc_data:
            self.parent.fixed_faces_y = []
            for idx in bc_data['fixed_faces_y_indices']:
                face_data = self.parent.stl_geom.get_triangle_data(idx)
                if face_data:
                    self.parent.fixed_faces_y.append(face_data)
            
            constraint_data = {
                'faces': self.parent.fixed_faces_y,
                'color': (0.3, 0.7, 0.3)  # Light green for Y constraints
            }
            self.restore_constraint(constraint_data)
            
        if 'fixed_faces_z_indices' in bc_data:
            self.parent.fixed_faces_z = []
            for idx in bc_data['fixed_faces_z_indices']:
                face_data = self.parent.stl_geom.get_triangle_data(idx)
                if face_data:
                    self.parent.fixed_faces_z.append(face_data)
            
            constraint_data = {
                'faces': self.parent.fixed_faces_z,
                'color': (0.3, 0.3, 0.7)  # Light blue for Z constraints
            }
            self.restore_constraint(constraint_data)
        
        # Restore load groups and forces
        if 'load_faces_indices' in bc_data:
            self.parent.load_faces_groups = []
            
            for i, group_indices in enumerate(bc_data['load_faces_indices']):
                faces_group = []
                for idx in group_indices:
                    face_data = self.parent.stl_geom.get_triangle_data(idx)
                    if face_data:
                        faces_group.append(face_data)
                
                self.parent.load_faces_groups.append(faces_group)
                
                # Visualize force if we have matching force data
                if 'load_forces' in bc_data and i < len(bc_data['load_forces']):
                    force = bc_data['load_forces'][i]
                    force_direction = np.array(force)
                    magnitude = np.linalg.norm(force_direction)
                    
                    if magnitude > 0:
                        # Normalize direction
                        direction = force_direction / magnitude
                        
                        # Calculate rotation angles once
                        dx, dy, dz = direction
                        
                        if abs(dx) > 0 or abs(dy) > 0:
                            angle_z = math.degrees(math.atan2(dy, dx))
                        else:
                            angle_z = 0
                            
                        angle_y = -math.degrees(math.asin(dz))
                        
                        # Get geometry size for scaling
                        if hasattr(self.parent.stl_geom, 'mesh') and hasattr(self.parent.stl_geom.mesh, 'points'):
                            points = np.array(self.parent.stl_geom.mesh.points)
                            bbox = [points[:,0].min(), points[:,0].max(),
                                    points[:,1].min(), points[:,1].max(),
                                    points[:,2].min(), points[:,2].max()]
                            geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
                            scale_factor = 0.10 * geom_size
                        else:
                            scale_factor = 0.1
                        
                        # Display arrows based on group size
                        MAX_ARROWS = 5  # Maximum number of arrows to display per group
                        THRESHOLD = 25   # Threshold for reducing number of arrows
                        
                        if len(faces_group) > THRESHOLD:
                            step = len(faces_group) // MAX_ARROWS
                            display_faces = faces_group[::step][:MAX_ARROWS]  # Select evenly spaced faces
                        else:
                            display_faces = faces_group
                        
                        # Create an arrow for each display face
                        for face in display_faces:
                            load_data = {
                                'position': face['center'],
                                'rotation_z': angle_z,
                                'rotation_y': angle_y,
                                'rotation_x': 0,
                                'scale': [scale_factor, scale_factor, scale_factor],
                                'color': (1, 0, 0)  # Red for force
                            }
                            self.restore_force(load_data)
        
        # Update UI state
        if hasattr(self.parent, 'fixed_faces') or hasattr(self.parent, 'fixed_faces_x') or \
        hasattr(self.parent, 'fixed_faces_y') or hasattr(self.parent, 'fixed_faces_z'):
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
            self.parent.update_LivVar('structural_loads.applied', True)
            self.parent.update_button_icon("Structural Loads", "check")
        
        if hasattr(self.parent, 'load_forces') and self.parent.load_forces:
            self.parent.update_LivVar('structural_loads.forces_applied', True)
            self.parent.update_LivVar('structural_loads.applied', True)
            self.parent.update_button_icon("Structural Loads", "check")

    def restore_thermal_bc(self, thermal_bc):
        """Reconstruct and visualize thermal boundary conditions from saved data"""
        if not self.parent.stl_geom:
            return
        
        # Ensure thermal_bc has all required keys
        if 'total_heat_sources' not in thermal_bc:
            thermal_bc['total_heat_sources'] = []
        
        if 'thermal_counts' in thermal_bc and 'total_heat_sources' not in thermal_bc['thermal_counts']:
            thermal_bc['thermal_counts']['total_heat_sources'] = 0
            
        # Output summary info first
        if 'thermal_counts' in thermal_bc:
            counts = thermal_bc['thermal_counts']
            if 'fixed_temps' in counts and counts['fixed_temps'] > 0:
                self.parent.message_text.append(f"Loading {counts['fixed_temps']} fixed temperature triangles")
            if 'heat_sources' in counts and counts['heat_sources'] > 0:
                self.parent.message_text.append(f"Loading {counts['heat_sources']} heat source triangles")
            if 'total_heat_sources' in counts and counts['total_heat_sources'] > 0:
                self.parent.message_text.append(f"Loading {counts['total_heat_sources']} total heat triangles")
            if 'convection' in counts and counts['convection'] > 0:
                self.parent.message_text.append(f"Loading {counts['convection']} convection triangles")
        
        # Initialize thermal loads data structure if not exists
        if not hasattr(self.parent, 'thermal_loads'):
            self.parent.thermal_loads = {
                "fixed_temps": [],
                "heat_sources": [],
                "total_heat_sources": [],
                "convection": [],
                "radiation": [],
                "internal_heat": []
            }
        
        # Initialize actor lists if they don't exist
        if not hasattr(self.parent, 'fixed_temp_actors'):
            self.parent.fixed_temp_actors = []
        if not hasattr(self.parent, 'heat_source_actors'):
            self.parent.heat_source_actors = []
        if not hasattr(self.parent, 'total_heat_actors'):
            self.parent.total_heat_actors = []
        if not hasattr(self.parent, 'convection_actors'):
            self.parent.convection_actors = []

        # Initialize self.fixed_temp_actors for use within this class's methods
        if not hasattr(self, 'fixed_temp_actors'):
            self.fixed_temp_actors = []
        
        # Restore fixed temperatures
        if 'fixed_temps' in thermal_bc:
            for data in thermal_bc['fixed_temps']:
                triangles = data['triangles']
                temperature = data['temperature']
                
                # Add to parent's thermal loads
                self.parent.thermal_loads['fixed_temps'].append((triangles, temperature))
                
                # Visualize
                self.visualize_fixed_temp(triangles, temperature)
        
        # Restore heat sources
        if 'heat_sources' in thermal_bc:
            for data in thermal_bc['heat_sources']:
                triangles = data['triangles']
                heat_flux = data['heat_flux']
                
                # Add to parent's thermal loads
                self.parent.thermal_loads['heat_sources'].append((triangles, heat_flux))
                
                # Visualize
                self.visualize_heat_source(triangles, heat_flux, is_total_heat=False)
        
        # Restore heat sources
        if 'heat_sources' in thermal_bc:
            for data in thermal_bc['heat_sources']:
                triangles = data['triangles']
                heat_flux = data['heat_flux']
                
                # Add to parent's thermal loads
                self.parent.thermal_loads['heat_sources'].append((triangles, heat_flux))
                
                # Visualize
                self.visualize_heat_source(triangles, heat_flux, is_total_heat=False)
        
        # For total heat sources, create a temporary ThermalLoadsWindow and use its apply_total_heat method
        try:
            if 'total_heat_sources' in thermal_bc:
                # Create a temporary ThermalLoadsWindow to access its methods
                temp_window = ThermalLoadsWindow(self.parent)
                
                for data in thermal_bc['total_heat_sources']:
                    triangles = data['triangles']
                    total_heat = data['total_heat']
                    
                    # Set up the temporary window with the data
                    # Select the triangles
                    self.parent.stl_geom.tri_highlight = [False] * self.parent.stl_geom.stl_n_triangles
                    for idx in triangles:
                        if idx < len(self.parent.stl_geom.tri_highlight):
                            self.parent.stl_geom.tri_highlight[idx] = True
                    
                    # Set the total heat value
                    temp_window.total_heat_value_spin.setValue(total_heat)
                    
                    # Call the apply_total_heat method
                    temp_window.apply_total_heat()
                
                # Clean up
                temp_window.close()
        except Exception as e:
            self.parent.message_text.append(f"Warning: Error loading total heat sources: {str(e)}")
        
        # Restore convection
        if 'convection' in thermal_bc:
            for data in thermal_bc['convection']:
                triangles = data['triangles']
                h_coeff = data['h_coeff']
                ambient_temp = data['ambient_temp']
                
                # Add to parent's thermal loads
                self.parent.thermal_loads['convection'].append((triangles, h_coeff, ambient_temp))
                
                # Visualize
                self.visualize_convection(triangles, h_coeff, ambient_temp)
        
        # Update UI state if any thermal loads were loaded
        if (thermal_bc.get('fixed_temps') or 
            thermal_bc.get('heat_sources') or 
            thermal_bc.get('total_heat_sources') or
            thermal_bc.get('convection')):
            self.parent.update_LivVar('thermal_loads.applied', True)
            
            if thermal_bc.get('heat_sources') or thermal_bc.get('total_heat_sources'):
                self.parent.update_LivVar('thermal_loads.heat_sources', True)
                
            if thermal_bc.get('convection'):
                self.parent.update_LivVar('thermal_loads.convection_applied', True)
                
            self.parent.update_button_icon("Thermal Loads", "check")

    def restore_topopt_constraints(self, constraints):
        """Restore topology optimization constraints and visualization from saved data"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
            
        # Initialize constraint actors if needed
        if not hasattr(self.parent, 'topopt_constraint_actors'):
            self.parent.initialize_constraint_actors()
            
        # Create a temporary TopOptConstraintsWindow to use its methods
        try:
            temp_window = TopOptConstraintsWindow(self.parent)
            
            # Set the UI elements based on the saved constraints
            if 'manufacturing' in constraints:
                manufacturing = constraints['manufacturing']
                
                # Set extrude constraints
                if 'extrude' in manufacturing:
                    extrude_data = manufacturing['extrude']
                    temp_window.extrude_check.setChecked(extrude_data.get('enabled', False))
                    if 'direction' in extrude_data:
                        index = temp_window.extrude_combo.findText(extrude_data['direction'])
                        if index >= 0:
                            temp_window.extrude_combo.setCurrentIndex(index)
                
                # Set AM build constraints
                if 'am_build' in manufacturing:
                    am_build_data = manufacturing['am_build']
                    temp_window.am_build_check.setChecked(am_build_data.get('enabled', False))
                    if 'direction' in am_build_data:
                        index = temp_window.am_build_combo.findText(am_build_data['direction'])
                        if index >= 0:
                            temp_window.am_build_combo.setCurrentIndex(index)
                
                # Set draw direction constraints
                if 'draw_direction' in manufacturing:
                    draw_dir_data = manufacturing['draw_direction']
                    temp_window.draw_direction_check.setChecked(draw_dir_data.get('enabled', False))
                    if 'direction' in draw_dir_data:
                        index = temp_window.draw_direction_combo.findText(draw_dir_data['direction'])
                        if index >= 0:
                            temp_window.draw_direction_combo.setCurrentIndex(index)
                
                # Set cyclic symmetry constraints
                if 'cyclic_symmetry' in manufacturing:
                    cyclic_sym_data = manufacturing['cyclic_symmetry']
                    temp_window.cyclic_sym_check.setChecked(cyclic_sym_data.get('enabled', False))
                    if 'angle' in cyclic_sym_data:
                        index = temp_window.cyclic_sym_combo.findText(cyclic_sym_data['angle'])
                        if index >= 0:
                            temp_window.cyclic_sym_combo.setCurrentIndex(index)
            
            # Set pattern constraints
            if 'patterns' in constraints:
                patterns = constraints['patterns']
                
                # Set X grid pattern
                if 'x_grid' in patterns:
                    x_grid_data = patterns['x_grid']
                    temp_window.x_grid_check.setChecked(x_grid_data.get('enabled', False))
                    if 'value' in x_grid_data:
                        temp_window.x_grid_spin.setValue(x_grid_data['value'])
                
                # Set Y grid pattern
                if 'y_grid' in patterns:
                    y_grid_data = patterns['y_grid']
                    temp_window.y_grid_check.setChecked(y_grid_data.get('enabled', False))
                    if 'value' in y_grid_data:
                        temp_window.y_grid_spin.setValue(y_grid_data['value'])
                
                # Set Z grid pattern
                if 'z_grid' in patterns:
                    z_grid_data = patterns['z_grid']
                    temp_window.z_grid_check.setChecked(z_grid_data.get('enabled', False))
                    if 'value' in z_grid_data:
                        temp_window.z_grid_spin.setValue(z_grid_data['value'])
            
            # Set performance constraints
            if 'performance' in constraints:
                performance = constraints['performance']
                
                # Set stress safety
                if 'stress_safety' in performance:
                    stress_data = performance['stress_safety']
                    temp_window.stress_safety_check.setChecked(stress_data.get('enabled', False))
                    if 'value' in stress_data:
                        temp_window.stress_safety_spin.setValue(stress_data['value'])
                
                # Set max displacement
                if 'max_displacement' in performance:
                    disp_data = performance['max_displacement']
                    temp_window.max_disp_check.setChecked(disp_data.get('enabled', False))
                    if 'value' in disp_data:
                        temp_window.max_disp_spin.setValue(disp_data['value'])
                
                # Set min frequency
                if 'min_frequency' in performance:
                    freq_data = performance['min_frequency']
                    temp_window.min_freq_check.setChecked(freq_data.get('enabled', False))
                    if 'value' in freq_data:
                        temp_window.min_freq_spin.setValue(freq_data['value'])
                
                # Set max temperature
                if 'max_temperature' in performance:
                    temp_data = performance['max_temperature']
                    temp_window.max_temp_check.setChecked(temp_data.get('enabled', False))
                    if 'value' in temp_data:
                        temp_window.max_temp_spin.setValue(temp_data['value'])
                
                # Set min feature
                if 'min_feature' in performance:
                    min_feat_data = performance['min_feature']
                    temp_window.min_feat_check.setChecked(min_feat_data.get('enabled', False))
                    if 'value' in min_feat_data:
                        temp_window.min_feat_spin.setValue(min_feat_data['value'])
                
                # Set max feature
                if 'max_feature' in performance:
                    max_feat_data = performance['max_feature']
                    temp_window.max_feat_check.setChecked(max_feat_data.get('enabled', False))
                    if 'value' in max_feat_data:
                        temp_window.max_feat_spin.setValue(max_feat_data['value'])
            
            # Set symmetry constraints
            if 'symmetry' in constraints:
                symmetry = constraints['symmetry']
                
                # Set X symmetry
                if 'x_symmetry' in symmetry:
                    temp_window.x_symmetry_check.setChecked(symmetry['x_symmetry'])
                
                # Set Y symmetry
                if 'y_symmetry' in symmetry:
                    temp_window.y_symmetry_check.setChecked(symmetry['y_symmetry'])
                
                # Set Z symmetry
                if 'z_symmetry' in symmetry:
                    temp_window.z_symmetry_check.setChecked(symmetry['z_symmetry'])
            
            # Set other constraints
            if 'other' in constraints:
                other = constraints['other']
                
                # Set connected topology
                if 'connected_topology' in other:
                    temp_window.connected_topology_check.setChecked(other['connected_topology'])
                
                # Set keep fixed faces
                if 'keep_fixed_faces' in other:
                    temp_window.keep_fixed_faces_check.setChecked(other['keep_fixed_faces'])
            
            # Apply the constraints to update visualizations
            temp_window.apply_constraints()
            
            # Clean up
            temp_window.close()
            
            # Update UI indicators
            self.parent.update_button_icon("TopOpt Constraints", "check")
            self.parent.update_LivVar('topopt.constraints_defined', True)
            
        except Exception as e:
            self.parent.message_text.append(f"Warning: Error loading topology optimization constraints: {str(e)}")

    def visualize_fixed_temp(self, nodes, temperature):
        """Visualize fixed temperature on the model with arrows"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
                
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # Blue for fixed temperature
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.parent.fixed_temp_actors.append(actor)
        
        # Convert from internal Kelvin to display unit
        if self.parent.settings.temperature_unit == "Celsius":
            display_temp = temperature - 273.15
            unit = "°C"
        elif self.parent.settings.temperature_unit == "Fahrenheit":
            display_temp = (temperature - 273.15) * 9/5 + 32
            unit = "°F"
        else:  # Kelvin
            display_temp = temperature
            unit = "K"
        
        # Get bounding box for scaling
        if hasattr(self.parent.stl_geom, 'get_bounding_box'):
            bbox = self.parent.stl_geom.get_bounding_box()
        else:
            # Calculate bounding box from mesh
            vertices = self.parent.stl_geom.mesh.vectors.reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Calculate model size for arrow scaling
        geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        scale_factor = 0.08 * geom_size  # Scale arrows based on model size
        
        # Determine how many arrows to show
        MAX_MARKERS = 5
        THRESHOLD = 25
        
        if len(nodes) > THRESHOLD:
            step = len(nodes) // MAX_MARKERS
            display_indices = range(0, len(nodes), step)[:MAX_MARKERS]
            display_nodes = [nodes[i] for i in display_indices]
        else:
            display_nodes = nodes
        
        # Flag to indicate if we've added text display for one arrow
        text_label_added = False
        
        # Add arrows directly instead of in a separate function to ensure they're created
        for idx in display_nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                # Get triangle data
                triangle = None
                if hasattr(self.parent.stl_geom, 'get_triangle_data'):
                    triangle = self.parent.stl_geom.get_triangle_data(idx)
                else:
                    #Will remove this once functionality added for Bounding Box to test if it works no need to calculate evrytime even if its in else condition
                    # Manually calculate triangle data
                    vertices = self.parent.stl_geom.mesh.vectors[idx]
                    # Calculate center
                    center = [
                        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
                        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3,
                        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3
                    ]
                    
                    # Calculate normal using cross product
                    v1 = [vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], vertices[1][2] - vertices[0][2]]
                    v2 = [vertices[2][0] - vertices[0][0], vertices[2][1] - vertices[0][1], vertices[2][2] - vertices[0][2]]
                    
                    # Cross product
                    normal = [
                        v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]
                    ]
                    
                    # Normalize
                    length = sum(n*n for n in normal) ** 0.5
                    if length > 0:
                        normal = [n/length for n in normal]
                    else:
                        normal = [0, 0, 1]  # Default if calculation fails
                        
                    triangle = {'center': center, 'normal': normal, 'index': idx}
                
                if triangle:
                    # Create arrow for visualization
                    arrow = vtk.vtkArrowSource()
                    arrow.SetTipLength(0.3)
                    arrow.SetTipRadius(0.1)
                    arrow.SetShaftRadius(0.03)
                    
                    # For temperature, we want arrows pointing inward (same as heat flux)
                    normal = [-n for n in triangle['normal']]
                    
                    # Calculate arrow start position - move back along the inverted normal
                    # This positions the arrow behind the surface with its tip at the surface
                    arrow_length = scale_factor
                    
                    # Calculate start position (move back from center so tip is at the center)
                    start_pos = [
                        triangle['center'][0] - normal[0] * arrow_length,
                        triangle['center'][1] - normal[1] * arrow_length,
                        triangle['center'][2] - normal[2] * arrow_length
                    ]
                    
                    # Create transform
                    transform = vtk.vtkTransform()
                    
                    # Position transform at the starting position
                    transform.Translate(start_pos)
                    
                    # Handle special cases first for better numerical stability
                    if abs(normal[0]) > 0.999:  # Almost parallel to X axis
                        if normal[0] < 0:
                            transform.RotateY(180)
                    elif abs(normal[1]) > 0.999:  # Almost parallel to Y axis
                        # Rotate 90 deg around Z
                        if normal[1] > 0:
                            transform.RotateZ(90)
                        else:
                            transform.RotateZ(-90)
                    elif abs(normal[2]) > 0.999:  # Almost parallel to Z axis
                        # Rotate around Y to point along Z
                        if normal[2] > 0:
                            transform.RotateY(-90)
                        else:
                            transform.RotateY(90)
                    else:
                        # General case - first rotate in XY plane (Z rotation)
                        angle_z = math.degrees(math.atan2(normal[1], normal[0]))
                        transform.RotateZ(angle_z)
                        
                        # Then rotate to correct elevation (Y rotation)
                        # Project normal onto XY plane after Z rotation
                        length_xy = math.sqrt(normal[0]**2 + normal[1]**2)
                        angle_y = math.degrees(math.atan2(normal[2], length_xy))
                        transform.RotateY(angle_y)
                    
                    # Apply scaling
                    transform.Scale(scale_factor, scale_factor, scale_factor)
                    
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(arrow.GetOutputPort())
                    
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(mapper)
                    arrow_actor.SetUserTransform(transform)
                    
                    # Make arrows distinct - use different blue for better visibility
                    arrow_actor.GetProperty().SetColor(0.2, 0.2, 1.0)  # Bright blue for temperature arrows
                    # Ensure the arrow is not transparent
                    arrow_actor.GetProperty().SetOpacity(1.0)
                    
                    # Add to renderer and store
                    self.parent.renderer.AddActor(arrow_actor)
                    self.parent.fixed_temp_actors.append(arrow_actor)
                    
                    # Add text display next to one of the arrows (first arrow only)
                    if not text_label_added:
                        # Calculate position for text - offset from the arrow but closer
                        text_offset = 0.08 * geom_size  # Reduced offset to bring text closer
                        
                        # Place text near the arrow, offset in a good direction for visibility
                        text_pos = [
                            triangle['center'][0] + text_offset * (-0.5 if normal[0] < 0 else 0.5),
                            triangle['center'][1] + text_offset * (-0.5 if normal[1] < 0 else 0.5),
                            triangle['center'][2] + text_offset * (0.2 if normal[2] < 0 else -0.2)
                        ]
                        
                        # Create a vtkCaptionActor2D for 3D text
                        caption_actor = vtk.vtkCaptionActor2D()
                        
                        # Set text content
                        caption_text = f"T = {display_temp:.1f}{unit}"
                        
                        caption_actor.SetCaption(caption_text)
                        caption_actor.SetAttachmentPoint(text_pos)
                        
                        # Customize text appearance
                        caption_actor.BorderOff()
                        caption_actor.LeaderOff() 
                        
                        # Set text properties
                        text_prop = caption_actor.GetCaptionTextProperty()
                        text_prop.SetColor(0.2, 0.2, 1.0) 
                        text_prop.SetFontSize(5)

                        # And add these text scaling properties
                        caption_actor.SetWidth(0.15)
                        caption_actor.SetHeight(0.05)
                        text_prop.SetBold(True)
                        text_prop.SetShadow(True)
                        
                        # ... and similar for caption_actor ...
                        self.parent.renderer.AddActor(caption_actor)
                        self.parent.fixed_temp_actors.append(caption_actor)  # Use parent's list
                        
                        text_label_added = True
        
        # Force rendering update
        self.parent.vtkWidget.GetRenderWindow().Render()

    def visualize_heat_source(self, nodes, heat_flux, is_total_heat=False, total_heat_value=None):
        """Visualize heat source on the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
            
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Use slightly different color for total heat vs. heat flux
        if is_total_heat:
            actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange for total heat
        else:
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red for heat flux
        
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        if is_total_heat:
            if not hasattr(self.parent, 'total_heat_actors'):
                self.parent.total_heat_actors = []
            self.parent.total_heat_actors.append(actor)
        else:
            self.parent.heat_source_actors.append(actor)
        
        # Get bounding box for scaling arrows
        if hasattr(self.parent.stl_geom, 'get_bounding_box'):
            bbox = self.parent.stl_geom.get_bounding_box()
        else:
            # Calculate bounding box from mesh
            vertices = self.parent.stl_geom.mesh.vectors.reshape(-1, 3)
            xmin, ymin, zmin = np.min(vertices, axis=0)
            xmax, ymax, zmax = np.max(vertices, axis=0)
            bbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Calculate model size for arrow scaling
        geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        scale_factor = 0.08 * geom_size  # Scale arrows based on model size
        
        # Add heat arrows - determine how many to display
        MAX_MARKERS = 5
        THRESHOLD = 25
        
        if len(nodes) > THRESHOLD:
            step = len(nodes) // MAX_MARKERS
            display_indices = range(0, len(nodes), step)[:MAX_MARKERS]
            display_nodes = [nodes[i] for i in display_indices]
        else:
            display_nodes = nodes
        
        # Flag for text label
        text_label_added = False
        
        # Create arrows for each display node
        for idx in display_nodes:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                # Get triangle data
                triangle = None
                if hasattr(self.parent.stl_geom, 'get_triangle_data'):
                    triangle = self.parent.stl_geom.get_triangle_data(idx)
                else:
                    # Manual calculation if not available
                    vertices = self.parent.stl_geom.mesh.vectors[idx]
                    center = [
                        (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
                        (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3,
                        (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3
                    ]
                    
                    v1 = [vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1], vertices[1][2] - vertices[0][2]]
                    v2 = [vertices[2][0] - vertices[0][0], vertices[2][1] - vertices[0][1], vertices[2][2] - vertices[0][2]]
                    
                    normal = [
                        v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]
                    ]
                    
                    length = sum(n*n for n in normal) ** 0.5
                    if length > 0:
                        normal = [n/length for n in normal]
                    else:
                        normal = [0, 0, 1]
                        
                    triangle = {'center': center, 'normal': normal, 'index': idx}
                
                if triangle:
                    # Create arrow
                    arrow = vtk.vtkArrowSource()
                    arrow.SetTipLength(0.3)
                    arrow.SetTipRadius(0.1)
                    arrow.SetShaftRadius(0.03)
                    
                    # Get inverted normal
                    normal = [-n for n in triangle['normal']]
                    
                    # Calculate start position
                    arrow_length = scale_factor
                    start_pos = [
                        triangle['center'][0] - normal[0] * arrow_length,
                        triangle['center'][1] - normal[1] * arrow_length,
                        triangle['center'][2] - normal[2] * arrow_length
                    ]
                    
                    # Create transform
                    transform = vtk.vtkTransform()
                    transform.Translate(start_pos)
                    
                    # Calculate rotation angles
                    if abs(normal[0]) > 0.999:
                        if normal[0] < 0:
                            transform.RotateY(180)
                    elif abs(normal[1]) > 0.999:
                        if normal[1] > 0:
                            transform.RotateZ(90)
                        else:
                            transform.RotateZ(-90)
                    elif abs(normal[2]) > 0.999:
                        if normal[2] > 0:
                            transform.RotateY(-90)
                        else:
                            transform.RotateY(90)
                    else:
                        angle_z = math.degrees(math.atan2(normal[1], normal[0]))
                        transform.RotateZ(angle_z)
                        
                        length_xy = math.sqrt(normal[0]**2 + normal[1]**2)
                        angle_y = math.degrees(math.atan2(normal[2], length_xy))
                        transform.RotateY(angle_y)
                    
                    # Apply scaling
                    transform.Scale(scale_factor, scale_factor, scale_factor)
                    
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(arrow.GetOutputPort())
                    
                    arrow_actor = vtk.vtkActor()
                    arrow_actor.SetMapper(mapper)
                    arrow_actor.SetUserTransform(transform)
                    
                    # Set color based on type
                    if is_total_heat:
                        arrow_actor.GetProperty().SetColor(1.0, 0.5, 0.0)  # Orange for total heat
                    else:
                        arrow_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red for heat flux
                    
                    # Add to renderer
                    self.parent.renderer.AddActor(arrow_actor)
                    
                    # Add to correct actor list
                    if is_total_heat:
                        self.parent.total_heat_actors.append(arrow_actor)
                    else:
                        self.parent.heat_source_actors.append(arrow_actor)
                    
                    # Add text label
                    if not text_label_added:
                        text_offset = 0.08 * geom_size
                        text_pos = [
                            triangle['center'][0] + text_offset * (-0.5 if normal[0] < 0 else 0.5),
                            triangle['center'][1] + text_offset * (-0.5 if normal[1] < 0 else 0.5),
                            triangle['center'][2] + text_offset * (0.2 if normal[2] < 0 else -0.2)
                        ]
                        
                        caption_actor = vtk.vtkCaptionActor2D()
                        
                        if is_total_heat and total_heat_value is not None:
                            caption_text = f"{total_heat_value} W"
                        else:
                            caption_text = f"{heat_flux} W/m²"
                        
                        caption_actor.SetCaption(caption_text)
                        caption_actor.SetAttachmentPoint(text_pos)
                        
                        caption_actor.BorderOff()
                        caption_actor.LeaderOff()
                        
                        text_prop = caption_actor.GetCaptionTextProperty()
                        if is_total_heat:
                            text_prop.SetColor(1.0, 0.5, 0.0)
                        else:
                            text_prop.SetColor(1.0, 0.0, 0.0)
                        
                        text_prop.SetFontSize(5)
                        caption_actor.SetWidth(0.15)
                        caption_actor.SetHeight(0.05)
                        text_prop.SetBold(True)
                        text_prop.SetShadow(True)
                        
                        self.parent.renderer.AddActor(caption_actor)
                        
                        if is_total_heat:
                            self.parent.total_heat_actors.append(caption_actor)
                        else:
                            self.parent.heat_source_actors.append(caption_actor)
                        
                        text_label_added = True
        
        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()

    def visualize_convection(self, triangles, h_coeff, ambient_temp):
        """Visualize convection on the model"""
        if not hasattr(self.parent, 'stl_geom') or self.parent.stl_geom is None:
            return
            
        # Create points and cells for visualization
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        
        # Use selected triangles for visualization
        for idx in triangles:
            if idx < len(self.parent.stl_geom.mesh.vectors):
                vertices = self.parent.stl_geom.mesh.vectors[idx]
                point_ids = []
                for v in vertices:
                    point_ids.append(points.InsertNextPoint(v))
                tri = vtk.vtkTriangle()
                for i in range(3):
                    tri.GetPointIds().SetId(i, point_ids[i])
                cells.InsertNextCell(tri)
        
        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0.8, 0)  # Green for convection
        actor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.parent.renderer.AddActor(actor)
        self.parent.convection_actors.append(actor)
        
        # Add text label with convection parameters
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"h = {h_coeff} W/m²K, T∞ = {ambient_temp}°C")
        text_actor.SetPosition(10, 50)
        text_actor.GetTextProperty().SetColor(0, 0.8, 0)  # Green text
        text_actor.GetTextProperty().SetFontSize(14)
        self.parent.renderer.AddActor2D(text_actor)
        self.parent.convection_actors.append(text_actor)
        
        # Update rendering
        self.parent.vtkWidget.GetRenderWindow().Render()
            
#----------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("pyTO")
    
    screen = QtWidgets.QApplication.primaryScreen()
    geometry = screen.availableGeometry()
    # Get screen dimensions
    screen_width = geometry.width()
    screen_height = geometry.height()
    # Calculate window size (80% of screen size)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Set minimum size to prevent window from becoming too small
    window.setMinimumSize(800, 600)
    
    # Set initial window size
    window.resize(window_width, window_height)
    
    # Center the window on screen
    frame_geometry = window.frameGeometry()
    screen_center = QtWidgets.QApplication.primaryScreen().availableGeometry().center()
    frame_geometry.moveCenter(screen_center)
    window.move(frame_geometry.topLeft())
    
    window.show()
    sys.exit(app.exec_())

