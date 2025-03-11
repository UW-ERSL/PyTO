import sys
import vtk
import math
import numpy as np
import os
import json
import time

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QFrame
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt
from queue import Queue
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QFrame
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt  # Updated import: added Qt
sys.path.append('./src') #assuming the pyTO src files is in the parent directory
from STLGeom import STLGeom
from mesher import Mesher
import bound_cond
import mat_lib


'''
pyTOGUI To do:
1. Main: Disable x Buttons
2. Update Button status
3. Analysis: Update mesh elements based on mesh quality

'''
class ProjectData:
    def __init__(self):
        self.version = "2025.01"
        self.stl_file_path = None
        self.settings = None
        self.material_data = None  # Add this line
        

    def to_dict(self):
        return {
            'version': self.version,
            'stl_file_path': self.stl_file_path,
            'settings': {
                'unit_system': self.settings.unit_system,
                'temperature_unit': self.settings.temperature_unit,
                'angle_unit': self.settings.angle_unit
            } if self.settings else None,
            'material_data': self.material_data,  # Add this line
            'structuralBC': self.structuralBC,
            'topopt_constraints': self.topopt_constraints,
            'optimization_params': self.optimization_params,
            
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
        project.material_data = data.get('material_data')  # Add this line
        return project
    
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
                self.main_window.stl_filepath = file_path[:]
                self.main_window.load_stl_file(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load geometry: {str(e)}")
        self.close()

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
        self.setFixedSize(400, 400)  # Fixed size window
        
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
            self.parent.message_text.append(f"Material applied: {material_data['name']}")
            
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
            self.close()
            
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", str(e))


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
        self.load_type.addItems(["Force", "Fixed XYZ", "Fixed X", "Fixed Y", "Fixed Z"])
        load_type_layout.addWidget(load_type_label)
        load_type_layout.addWidget(self.load_type)
        layout.addLayout(load_type_layout)
        
        # Force components
        self.force_group = QtWidgets.QGroupBox()
        force_layout = QtWidgets.QVBoxLayout(self.force_group)
        
        # X Force
        x_force_layout = QtWidgets.QHBoxLayout()
        x_force_label = QtWidgets.QLabel("X-Force (N)")
        self.x_force_spin = QtWidgets.QDoubleSpinBox()
        self.x_force_spin.setRange(-1e6, 1e6)
        self.x_force_spin.setDecimals(1)
        x_force_layout.addWidget(x_force_label)
        x_force_layout.addWidget(self.x_force_spin)
        force_layout.addLayout(x_force_layout)
        
        # Y Force
        y_force_layout = QtWidgets.QHBoxLayout()
        y_force_label = QtWidgets.QLabel("Y-Force (N)")
        self.y_force_spin = QtWidgets.QDoubleSpinBox()
        self.y_force_spin.setRange(-1e6, 1e6)
        self.y_force_spin.setDecimals(1)
        y_force_layout.addWidget(y_force_label)
        y_force_layout.addWidget(self.y_force_spin)
        force_layout.addLayout(y_force_layout)
        
        # Z Force
        z_force_layout = QtWidgets.QHBoxLayout()
        z_force_label = QtWidgets.QLabel("Z-Force (N)")
        self.z_force_spin = QtWidgets.QDoubleSpinBox()
        self.z_force_spin.setRange(-1e6, 1e6)
        self.z_force_spin.setDecimals(1)
        z_force_layout.addWidget(z_force_label)
        z_force_layout.addWidget(self.z_force_spin)
        force_layout.addLayout(z_force_layout)
        
        layout.addWidget(self.force_group)
        
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
        self.force_group.setVisible(show_force)
        self.adjustSize()
        
    def apply_load(self):
        load_type = self.load_type.currentText()
        
        if load_type == "Force":
            self.apply_force()
        elif load_type == "Fixed XYZ":
            self.apply_fixed_constraint()
        elif load_type == "Fixed X":
            self.apply_fixed_constraint_x()
        elif load_type == "Fixed Y":
            self.apply_fixed_constraint_y()
        elif load_type == "Fixed Z":
            self.apply_fixed_constraint_z()
            
    def apply_force(self):
        if not self.parent.stl_geom:
            return
                
        try:
            # Get force components from spinboxes
            force_x = self.x_force_spin.value()
            force_y = self.y_force_spin.value()
            force_z = self.z_force_spin.value()
            
            # Calculate magnitude
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
            
            # Visualization code
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
                
                # Calculate direction angles
                dx, dy, dz = force_x/magnitude, force_y/magnitude, force_z/magnitude
                
                # Calculate rotations
                if abs(dx) > 0 or abs(dy) > 0:
                    angle_z = math.degrees(math.atan2(dy, dx))
                else:
                    angle_z = 0
                    
                angle_y = -math.degrees(math.asin(dz))
                
                transform.RotateZ(angle_z)
                transform.RotateY(angle_y)
                
                # Dynamic scaling based on geometry bounding size
                points = np.array(self.parent.stl_geom.mesh.points)
                bbox = [points[:,0].min(), points[:,0].max(),
                        points[:,1].min(), points[:,1].max(),
                        points[:,2].min(), points[:,2].max()]
                geom_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
                scale_factor = 0.10 * geom_size
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
                
            # Release forced faces from selection
            for triangle in selected_faces:
                idx = triangle['index']
                self.parent.stl_geom.tri_highlight[idx] = False
            self.parent.update_highlights()
            
            self.parent.vtkWidget.GetRenderWindow().Render()
            self.parent.message_text.append(f"Applied force of {magnitude:.2f}N to {len(selected_faces)} triangles")
            self.parent.update_button_icon("Structural Loads", "check")
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
    
class AnalysisWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Analysis")
        self.resize(300, 300)
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
        self.elements_spin.setValue(10000)
        elements_layout.addWidget(elements_label)
        elements_layout.addWidget(self.elements_spin)
        layout.addLayout(elements_layout)
        
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
        layout.addLayout(deflation_groups_layout)
        
        # # Number of Modes
        # modes_layout = QtWidgets.QHBoxLayout()
        # modes_label = QtWidgets.QLabel("#Modes")
        # self.modes_spin = QtWidgets.QSpinBox()
        # self.modes_spin.setRange(1, 100)
        # self.modes_spin.setValue(1)
        # modes_layout.addWidget(modes_label)
        # modes_layout.addWidget(self.modes_spin)
        # layout.addLayout(modes_layout)
        
        # Include Thermal Effect
        self.thermal_check = QtWidgets.QCheckBox("Include Thermal Effect")
        layout.addWidget(self.thermal_check)
        
        # Zero-strain Temperature
        temp_layout = QtWidgets.QHBoxLayout()
        temp_label = QtWidgets.QLabel("Zero-strain T(K):")
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0, 1000)
        self.temp_spin.setValue(300.00)
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
        # self.toggle_results_button = QtWidgets.QPushButton("Toggle Results View")
        # self.toggle_results_button.clicked.connect(self.toggle_results_view)
        # layout.addWidget(self.toggle_results_button)

        # def toggle_results_view(self):
        #     """Toggle between original and deformed mesh"""
        #     if hasattr(self.parent, 'results_actor'):
        #         if self.parent.results_actor:
        #             visible = self.parent.results_actor.GetVisibility()
        #             self.parent.results_actor.SetVisibility(not visible)
        #             self.parent.scalar_bar.SetVisibility(not visible)
        #             if hasattr(self.parent, 'mesh_actor'):
        #                 self.parent.mesh_actor.SetVisibility(visible)
        #             self.parent.vtkWidget.GetRenderWindow().Render()
    
    def generate_mesh(self):
        """Generate mesh from geometry"""
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "Error", "No geometry loaded")
            return
            
        try:
            num_elements = self.elements_spin.value()
            self.parent.generate_analysis_mesh(num_elements)
            
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
                
            self.parent.update_button_icon("Analysis", "check")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate mesh: {str(e)}")

    def run_thermal_analysis(self, params):
        # Implementation for thermal analysis
        pass

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
                    
                    # Process mesh and create boundary conditions
                    _, _, self.parent.boundary_conditions = MainWindow.ProcessDataforSolver(
                        existing_mesh=mesh,
                        fixed_nodes=fixed_nodes,
                        load_data=load_data,
                        youngs_modulus=self.parent.material_data['young_modulus'],
                        poissons_ratio=self.parent.material_data['poisson_ratio']
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Error", f"Failed to create boundary conditions: {str(e)}")
                    return

            # Continue with structural analysis
            import struct_fea as fea
            import linear_solvers as lin_solv
            import jax
            
            # Enable double precision
            jax.config.update("jax_enable_x64", True)
            
            # Create FE solver using the data from generate_analysis_mesh
            fe_solver = fea.StructFEA(
                mesh=self.parent.analysis_mesher,
                mat_prop=self.parent.material_props,
                bc=self.parent.boundary_conditions,
                solver=lin_solv.Solvers.PARDISO
            )

            # Run analysis
            self.parent.message_text.append("\nRunning structural analysis...")
            youngs_modulus = np.ones((fe_solver.mesh.num_elems,))
            startTime = time.time()
            u = np.asarray(fe_solver.solve(elem_youngs_modulus=youngs_modulus))
            
            # Calculate displacements
            delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
            deltaMax = np.max(delta)
            nDOF = 3*fe_solver.mesh.num_nodes
            
            # Print results
            self.parent.message_text.append('-----------------------------')
            self.parent.message_text.append(f"nDof: {nDOF}")
            self.parent.message_text.append(f'Solver: {fe_solver.solver.name}')
            self.parent.message_text.append(f"FEA time: {time.time() - startTime}")
            self.parent.message_text.append(f'Max displacement: {deltaMax}')
            self.parent.message_text.append('-----------------------------')
            
            # Store results
            self.parent.analysis_results = {
                'displacements': u,
                'max_displacement': deltaMax,
                'delta': delta
            }
            
            # Update button status
            self.parent.update_button_icon("Analysis", "check")

            self.visualize_results()  # Add this line   

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
            
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(mapper.GetLookupTable())
            scalar_bar.SetTitle("Displacement (m)")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetPosition(0.85, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.8)
            scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
            scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
            self.parent.scalar_bar = scalar_bar
            
            # Hide original mesh
            if hasattr(self.parent, 'mesh_actor'):
                self.parent.mesh_actor.SetVisibility(False)
            
            # Add actors to renderer
            self.parent.renderer.AddActor(self.parent.results_actor)
            self.parent.renderer.AddActor(self.parent.scalar_bar)
            
            # Reset camera and render
            self.parent.renderer.ResetCamera()
            self.parent.vtkWidget.GetRenderWindow().Render()
            
            # Add results summary
            max_disp = np.max(delta)
            self.parent.message_text.append(f"\nVisualization updated:")
            self.parent.message_text.append(f"Maximum displacement: {max_disp:.6f} m")
            self.parent.message_text.append(f"Scale factor: {scale_factor:.2f}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to visualize results: {str(e)}")


class Analysis:
    def __init__(self):
        self.load_set = 0
        self.mesh_quality = "Very Coarse"
        self.num_elements = 10000
        self.deflation_type = "Rigid"
        self.num_deflation_groups = 100
        self.num_modes = 1
        self.include_thermal = False
        self.zero_strain_temp = 300.0
        self.remesh = False
        
    

class TopOptConstraintsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Constraints")
        self.resize(300, 500)
        
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
        
        self.am_build_check = QtWidgets.QCheckBox("AM Build")
        self.am_build_combo = QtWidgets.QComboBox()
        self.am_build_combo.addItems(["+XDir", "+YDir", "+ZDir", "-XDir", "-YDir", "-ZDir"])
        self.form_layout.addRow(self.am_build_check, self.am_build_combo)
        
        self.draw_direction_check = QtWidgets.QCheckBox("DrawDirection")
        self.draw_direction_combo = QtWidgets.QComboBox()
        self.draw_direction_combo.addItems(["XDir", "YDir", "ZDir"])
        self.form_layout.addRow(self.draw_direction_check, self.draw_direction_combo)
        
        self.cyclic_sym_check = QtWidgets.QCheckBox("CyclicSym(Z)")
        self.cyclic_sym_combo = QtWidgets.QComboBox()
        self.cyclic_sym_combo.addItems(["+90 deg", "-90 deg"])
        self.form_layout.addRow(self.cyclic_sym_check, self.cyclic_sym_combo)
        
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
        """Gather and apply all constraint settings"""
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
        
        # Update icon in main window
        if hasattr(self.parent(), 'update_button_icon'):
            self.parent().update_button_icon("TopOpt Constraints", "check")
        
        # Store constraints in parent window
        if hasattr(self.parent(), 'topopt_constraints'):
            self.parent().topopt_constraints = constraints
            
        self.parent().message_text.append("TopOpt constraints applied")
        self.close()


class OptimizeTopologyWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimize Topology")
        self.resize(300, 400)
        self.parent = parent
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Objective selection
        obj_layout = QtWidgets.QHBoxLayout()
        obj_label = QtWidgets.QLabel("Objective")
        self.obj_combo = QtWidgets.QComboBox()
        self.obj_combo.addItems(["Min. Compliance", "Min. Mass", "Min. Stress"])
        obj_layout.addWidget(obj_label)
        obj_layout.addWidget(self.obj_combo)
        layout.addLayout(obj_layout)
        
        # Checkboxes
        self.use_all_loads = QtWidgets.QCheckBox("Use all Loads?")
        self.use_all_loads.setChecked(True)
        layout.addWidget(self.use_all_loads)
        
        self.use_simp = QtWidgets.QCheckBox("Use SIMP Method?")
        layout.addWidget(self.use_simp)
        
        self.smooth_surface = QtWidgets.QCheckBox("Smooth surface?")
        self.smooth_surface.setChecked(True)
        layout.addWidget(self.smooth_surface)
        
        # Load Set
        load_set_layout = QtWidgets.QHBoxLayout()
        load_set_label = QtWidgets.QLabel("Use Load Set")
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setRange(0, 100)
        self.load_set_spin.setValue(0)
        load_set_layout.addWidget(load_set_label)
        load_set_layout.addWidget(self.load_set_spin)
        layout.addLayout(load_set_layout)
        
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
        
        # Action buttons
        self.optimize_button = QtWidgets.QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.optimize_button)
        
        self.stop_button = QtWidgets.QPushButton("STOP OPTIMIZATION!")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)  # Initially disabled
        layout.addWidget(self.stop_button)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        # Initialize optimization state
        self.optimization_running = False
        
    def start_optimization(self):
        """Start the topology optimization process"""
        if not self.check_prerequisites():
            return
            
        optimization_params = {
            'objective': self.obj_combo.currentText(),
            'use_all_loads': self.use_all_loads.isChecked(),
            'use_simp': self.use_simp.isChecked(),
            'smooth_surface': self.smooth_surface.isChecked(),
            'load_set': self.load_set_spin.value(),
            'volume_fraction': self.vol_frac_spin.value(),
            'volume_step': self.vol_step_spin.value(),
            'save_intermediate': self.save_intermediate.isChecked()
        }
        
        # Update UI state
        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Store optimization parameters in parent window
        if hasattr(self.parent, 'optimization_params'):
            self.parent.optimization_params = optimization_params
            
        self.parent.message_text.append("Starting topology optimization...")
        
        # Update sidebar button icon
        if hasattr(self.parent, 'update_button_icon'):
            self.parent.update_button_icon("Structural TopOpt", "check")
            
        # Here you would typically start the actual optimization process
        # This would likely involve a separate thread or process
        # For now, we'll just simulate it with a message
        self.parent.message_text.append(f"Optimization parameters set:")
        self.parent.message_text.append(f"Objective: {optimization_params['objective']}")
        self.parent.message_text.append(f"Volume Fraction: {optimization_params['volume_fraction']}")
        
    def stop_optimization(self):
        """Stop the ongoing optimization process"""
        if self.optimization_running:
            self.optimization_running = False
            self.optimize_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.parent.message_text.append("Optimization stopped by user")
            
    def check_prerequisites(self):
        """Check if all required conditions are met for optimization"""
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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.settings = Settings()
        self.stl_geom = None
        self.stl_filepath = None    
        self.constraint_actors = []
        self.force_actors = []
        self.material_data = None
        self.structuralBC = None    # Added initialization for structural BC
        self.analysis = Analysis()
        self.topopt_constraints = None
        self.optimization_params = None
        self.thermal_optimization_params = None
        # self.cantilever_problem = CantileverProblem()
        self.results_actor = None
        self.scalar_bar = None
        self.analysis_results = None
        
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        # Set size policy for the main widget
        self.main_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        
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

         # Add geometry info text actor
        self.setup_geometry_info()

    def setup_geometry_info(self):
        """Setup the geometry information text overlay"""
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetPosition(10, 10)
        self.text_actor.GetTextProperty().SetFontSize(12)
        self.text_actor.GetTextProperty().SetColor(0, 0, 0)  # Black text
        self.text_actor.GetTextProperty().SetFontFamilyToArial()
        self.renderer.AddActor2D(self.text_actor)

    def calculate_geometry_metrics(self):
        """Calculate geometry metrics for the loaded STL file"""
        if not self.stl_geom:
            return None

        # Get filename without path
        filename = os.path.basename(self.stl_geom.file_path)
        
        # Calculate volume and center of mass
        volume = 0
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3

        for vertices in self.stl_geom.mesh.vectors:
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
        if not self.stl_geom:
            self.text_actor.SetInput("")
            return

        metrics = self.calculate_geometry_metrics()
        if metrics:
            info_text = (
                f"Model: {metrics['model']}\n"
                f"Length: {metrics['length']:.2f} (meter)\n"
                f"Volume: {metrics['volume']:.2e} (m^3)"
            )
            self.text_actor.SetInput(info_text)

    def load_stl_file(self, file_path):
        # Keep existing loading code
        
        self.stl_filepath = file_path
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
        
        # Create mapper and main actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)
        self.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        
        # Extract only feature edges
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.BoundaryEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.FeatureEdgesOn()
        featureEdges.SetFeatureAngle(30)
        featureEdges.Update()
        
        edgeMapper = vtk.vtkPolyDataMapper()
        edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
        
        edgeActor = vtk.vtkActor()
        edgeActor.SetMapper(edgeMapper)
        edgeActor.GetProperty().SetColor(0, 0, 0)
        edgeActor.GetProperty().SetLineWidth(1)
        
        # Remove existing actors and add new ones
        self.renderer.RemoveAllViewProps()
        self.setup_geometry_info()  # Recreate text actor after clearing
        self.renderer.AddActor(self.stl_actor)
        self.renderer.AddActor(edgeActor)
        
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
        
        # Update geometry information
        self.update_geometry_info()
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        self.message_text.setText(f"Model loaded with {self.stl_geom.stl_n_triangles} triangles")

        # Call the function to update the sidebar
        self.on_geometry_loaded()

    def toggle_edges(self):
        if hasattr(self, 'stl_actor'):
            prop = self.stl_actor.GetProperty()
            # prop.SetEdgeVisibility(not prop.GetEdgeVisibility())
            # prop.SetEdgeColor(0, 0, 0)  # Black edges
            self.vtkWidget.GetRenderWindow().Render()

    def open_display_options(self):
        dialog = DisplayOptionsWindow(self)
        dialog.exec_()

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

        # Define buttons with default icons
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
        # Route the button action based on its name
        if name == "Units":
            self.open_units_window()
        elif name == "Geometry":
            self.open_geometry_window()
        elif name == "Material":
            self.open_material_window()
        elif name == "Structural Loads":
            self.open_structural_loads()
        elif name == "Display Options":
            self.open_display_options()
        elif name == "Analysis":
            self.open_analysis_window()
        elif name == "TopOpt Constraints":
            self.open_topopt_constraints_window()  # Fixed method name to match definition
        elif name == "Structural TopOpt":
            self.open_structural_topopt_window()
        elif name == "Thermal TopOpt":
            self.open_thermal_topopt_window()
        elif name == "Projects":
            dialog = ProjectsWindow(self)
            dialog.exec_()
        

    def get_icon(self, icon_type):
        import os
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

    def update_button_icon(self, button_name, new_icon):
        """
        Dynamically update the icon of a specific sidebar button.
        """
        if button_name in self.sidebar_buttons:
            button = self.sidebar_buttons[button_name]
            button.setIcon(self.get_icon(new_icon))

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
        
        version_label = QtWidgets.QLabel("Pareto Version 2025.01")
        build_label = QtWidgets.QLabel("Build Date 2.19")
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
        
        # Enable other buttons if needed
        self.sidebar_buttons["Material"].setEnabled(True)
        self.sidebar_buttons["Structural Loads"].setEnabled(True)

    def generate_analysis_mesh(self, num_elements):
           if not self.stl_geom:
                return
                
           try:
                # Get material properties from the material data
                if not hasattr(self, 'material_data'):
                    QtWidgets.QMessageBox.warning(self, "Error", "No material assigned")
                    return
                    
                # Use the mesher to create the mesh
                self.analysis_mesher = Mesher()
                self.analysis_mesher.createMeshFromSTLFile(self.stl_filepath, num_elements)
                self.analysis_mesher.createEdofMatStructural()
                
                # Debug info about mesh
                self.message_text.append(f"\nMesh generated with {num_elements} elements")
                self.message_text.append(f"Total nodes in mesh: {self.analysis_mesher.num_nodes}")
                
                # Get boundary nodes
                boundary_nodes = self.analysis_mesher.get_boundary_nodes()
                boundary_points = self.analysis_mesher.node_xyz[boundary_nodes]
                self.message_text.append(f"Found {len(boundary_nodes)} boundary nodes")
                
                # Helper function to find nodes for faces
                def find_nodes_for_faces(faces):
                    if not faces:
                        return set()
                    nodes = set()
                    for face in faces:
                        distances = self.stl_geom.find_points_triangle_distances_vectorized(boundary_points, face['index'])
                        tolerance = min(self.analysis_mesher.elem_size)*0.9
                        close_points_mask = distances < tolerance
                        nodes.update(boundary_nodes[close_points_mask])
                    return nodes
                
                # Process fixed constraint faces
                fixed_nodes_xyz = find_nodes_for_faces(self.fixed_faces if hasattr(self, 'fixed_faces') else None)
                fixed_nodes_x = find_nodes_for_faces(self.fixed_faces_x if hasattr(self, 'fixed_faces_x') else None)
                fixed_nodes_y = find_nodes_for_faces(self.fixed_faces_y if hasattr(self, 'fixed_faces_y') else None)
                fixed_nodes_z = find_nodes_for_faces(self.fixed_faces_z if hasattr(self, 'fixed_faces_z') else None)
                
                # Process load faces
                load_nodes_groups = []
                if hasattr(self, 'load_faces_groups') and hasattr(self, 'load_forces'):
                    for i, faces_group in enumerate(self.load_faces_groups):
                        nodes = find_nodes_for_faces(faces_group)
                        load_nodes_groups.append(nodes)
                        force = self.load_forces[i]
                        self.message_text.append(f"\nLoad group {i+1}: {len(nodes)} nodes with force ({force[0]}, {force[1]}, {force[2]})N")
                
                # Store nodes for later use
                self.fixed_nodes = {
                    'xyz': fixed_nodes_xyz,
                    'x': fixed_nodes_x,
                    'y': fixed_nodes_y,
                    'z': fixed_nodes_z
                }
                self.load_nodes_groups = load_nodes_groups
                
                # Summary of fixed nodes
                self.message_text.append(f"\nFixed nodes summary:")
                self.message_text.append(f"- XYZ fixed: {len(fixed_nodes_xyz)} nodes")
                self.message_text.append(f"- X fixed: {len(fixed_nodes_x)} nodes")
                self.message_text.append(f"- Y fixed: {len(fixed_nodes_y)} nodes")
                self.message_text.append(f"- Z fixed: {len(fixed_nodes_z)} nodes")
                
                

                # # Prepare data for cantilever problem
                # problem_data = {
                #     'mesh': self.analysis_mesher,
                #     'fixed_nodes_data': self.fixed_nodes,
                #     'load_nodes_groups': self.load_nodes_groups,
                #     'load_forces': self.load_forces,
                #     'youngs_modulus': self.material_data['young_modulus'],
                #     'poissons_ratio': self.material_data['poisson_ratio']
                # }
                
                # # Store results
                # self.analysis_mesher = mesh
                # self.material_props = mat_props
                # self.boundary_conditions = boundary_conditions


                # Create visualization
                points = vtk.vtkPoints()
                cells = vtk.vtkCellArray()
                
                # Add points
                for node in self.analysis_mesher.node_xyz:
                    points.InsertNextPoint(node)
                
                # Add cells (hex elements)
                for elem in self.analysis_mesher.elemArray:
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
                for i in range(self.analysis_mesher.num_nodes):
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
                    else:
                        colors.InsertNextTuple3(200, 200, 200)  # Gray for unfixed/unloaded
                
                mesh.GetPointData().SetScalars(colors)
                
                # Create mapper and actor
                mapper = vtk.vtkDataSetMapper()
                mapper.SetInputData(mesh)
                
                if hasattr(self, 'mesh_actor'):
                    self.renderer.RemoveActor(self.mesh_actor)
                
                self.mesh_actor = vtk.vtkActor()
                self.mesh_actor.SetMapper(mapper)
                self.mesh_actor.GetProperty().SetOpacity(1.0)
                self.mesh_actor.GetProperty().EdgeVisibilityOn()
                self.mesh_actor.GetProperty().SetEdgeColor(0, 0, 0)
                
                # Hide STL geometry
                if hasattr(self, 'stl_actor'):
                    self.stl_actor.SetVisibility(False)
                
                # Add mesh actor to renderer
                self.renderer.AddActor(self.mesh_actor)
                
                # Reset camera and render
                self.renderer.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()

                # Prepare load data
                load_data = {
                    'load_nodes_groups': self.load_nodes_groups,
                    'load_forces': self.load_forces
                }
            
                # Call processDataforSolver to get mesh, material props, and boundary conditions
                # It looks like you're working on a comprehensive GUI application for structural and thermal topology optimization using VTK for visualization. Below, I'll provide a more detailed implementation of the `MainWindow` class, including the `generate_analysis_mesh` method, and the `ProcessDataforSolver` method for preparing data for the solver.
                # To ensure the `MainWindow` class and its methods are fully functional, let's break down the implementation step by step. Here's the complete implementation of the `MainWindow` class, including the `generate_analysis_mesh` and `ProcessDataforSolver` methods:
                # Certainly! Let's break down the implementation of the `MainWindow` class and the `generate_analysis_mesh` and `ProcessDataforSolver` methods step by step.
                self.analysis_mesher, self.material_props, self.boundary_conditions = MainWindow.ProcessDataforSolver(
                    existing_mesh=self.analysis_mesher,
                    fixed_nodes=self.fixed_nodes,
                    load_data=load_data,
                    youngs_modulus=self.material_data['young_modulus'],
                    poissons_ratio=self.material_data['poisson_ratio']
                )

           except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate analysis mesh: {str(e)}")

    @staticmethod
    def ProcessDataforSolver(existing_mesh, fixed_nodes, load_data, youngs_modulus, poissons_ratio):
        """""
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

        # Create boundary conditions and material properties
        bc = bound_cond.BC(force=force,
                        fixed_dofs=fixed_dofs,
                        dirichlet_values=dirichlet_values)
        mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                                            poissons_ratio=poissons_ratio)
        
        return mesh, mat_prop, bc

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
        
        # Add coordinate axes
        axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
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
        
        # Create mapper and main actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)
        self.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        # Removed triangle edge highlighting:
        # self.stl_actor.GetProperty().EdgeVisibilityOn()
        # self.stl_actor.GetProperty().SetEdgeColor(0, 0, 0)
        
        # Extract only feature (planar) edges using vtkFeatureEdges
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly_data)
        featureEdges.BoundaryEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.FeatureEdgesOn()
        featureEdges.SetFeatureAngle(30)  # Adjust if necessary
        featureEdges.Update()
        
        edgeMapper = vtk.vtkPolyDataMapper()
        edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
        
        edgeActor = vtk.vtkActor()
        edgeActor.SetMapper(edgeMapper)
        edgeActor.GetProperty().SetColor(0, 0, 0)
        edgeActor.GetProperty().SetLineWidth(1)
        
        # Remove existing actors and add new ones
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(self.stl_actor)
        self.renderer.AddActor(edgeActor)
        
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

        # Call the function to update the sidebar
        self.on_geometry_loaded()
    
    def open_material_window(self):
            dialog = MaterialWindow(self)
            dialog.exec_()
    
    def open_structural_loads(self):
        dialog = StructuralLoadsWindow(self)
        dialog.show()  # Use show() instead of exec_() to allow main window interaction

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

    def on_right_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        cell_id = self.picker.GetCellId()
        if cell_id >= 0: 
            self.clear_selections()
        
        
        self.interactor.GetInteractorStyle().OnRightButtonDown()

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self, self)
        dialog.exec_()

    def open_analysis_window(self):
        dialog = AnalysisWindow(self)
        dialog.exec_()

    def open_topopt_constraints_window(self):
        dialog = TopOptConstraintsWindow(self)
        dialog.exec_()

    def open_optimize_topology_window(self):
        dialog = OptimizeTopologyWindow(self)
        dialog.exec_()

    def open_structural_topopt_window(self):
        """Open the structural topology optimization window"""
        dialog = OptimizeTopologyWindow(self)
        dialog.exec_()

    def open_thermal_topopt_window(self):
        dialog = ThermalTopOptWindow(self)
        dialog.exec_()

    def sidebar_button_clicked(self, name):
        # Route the button action based on its name
        if name == "Units":
            self.open_units_window()
        elif name == "Geometry":
            self.open_geometry_window()
        elif name == "Material":
            self.open_material_window()
        elif name == "Structural Loads":
            self.open_structural_loads()
        elif name == "Display Options":
            self.open_display_options()
        elif name == "Analysis":
            self.open_analysis_window()
        elif name == "TopOpt Constraints":
            self.open_topopt_constraints_window()  # Fixed method name to match definition
        elif name == "Structural TopOpt":
            self.open_structural_topopt_window()
        elif name == "Thermal TopOpt":
            self.open_thermal_topopt_window()
        elif name == "Projects":
            dialog = ProjectsWindow(self)
            dialog.exec_()

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

    def save_project(self):
        if not self.stl_geom:
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry loaded to save")
            return
            
        project = ProjectData()
        project.stl_file_path = self.stl_geom.file_path.split('/')[-1]
        project.settings = self.settings
        project.material_data = self.material_data  # Add this line
        project.structuralBC = self.structuralBC
     

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
                with open(file_path, 'w') as f:
                    json.dump(project.to_dict(), f, indent=4)
                self.message_text.append(f"Project saved successfully to {file_path}")
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
                    
                    # Add after settings restoration:
                    if project.material_data:
                        self.material_data = project.material_data
                        # Update geometry color based on material
                        if hasattr(self, 'stl_actor'):
                            if project.material_data['name'] == "Custom":
                                self.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                            elif project.material_data['name'] == "AlloySteel":
                                self.stl_actor.GetProperty().SetColor(0.7, 0.7, 0.8)
                            elif project.material_data['name'] == "Aluminum":
                                self.stl_actor.GetProperty().SetColor(0.9, 0.9, 0.9)
                            elif project.material_data['name'] == "Titanium":
                                self.stl_actor.GetProperty().SetColor(0.8, 0.8, 0.7)
                            elif project.material_data['name'] == "StainlessSteel":
                                self.stl_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
                            self.vtkWidget.GetRenderWindow().Render()
                        
                        self.message_text.append(f"Loaded material: {project.material_data['name']}")
                    
                    self.message_text.append(f"Project loaded successfully from {file_path}")
                    
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
            self.renderer.AddActor(constraint_actor)
            self.constraint_actors.append(constraint_actor)
            
            # Update display
            self.vtkWidget.GetRenderWindow().Render()

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

class ProjectsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Projects")
        self.resize(300, 150)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        save_button = QtWidgets.QPushButton("Save Project")
        save_button.clicked.connect(self.parent.save_project)
        layout.addWidget(save_button)
        
        load_button = QtWidgets.QPushButton("Load Project")
        load_button.clicked.connect(self.parent.load_project)
        layout.addWidget(load_button)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

def get_screen_geometry():
    """Get the primary screen geometry"""
    screen = QtWidgets.QApplication.primaryScreen()
    geometry = screen.availableGeometry()
    return geometry.width(), geometry.height()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Pareto STL Viewer")
    
    # Get screen dimensions
    screen_width, screen_height = get_screen_geometry()
    
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

