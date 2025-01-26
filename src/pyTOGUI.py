import sys
from PyQt5 import QtWidgets #pip install PyQt5
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import STLGeom

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.settings = Settings()
        self.stl_geom = None
        
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

    def setup_sidebar(self):
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setFixedWidth(200)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(1)
        
        buttons = [
            ("Units", "#0066CC", self.open_units_window),
            ("Geometry", "#0066CC", self.open_geometry_window),
            ("Material", "#FF0000", None),
            ("Structural Loads", "#FF0000", None),
            ("Thermal Loads", "#FF0000", None),
            ("Body force", "#FF0000", None),
            ("Display Options", "#0066CC", None),
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
        """
        Sets up a message frame in the GUI.
        This method creates and configures a QFrame to display messages with the following components:
        - A QFrame with a box style border
        - A QTextEdit widget with fixed height of 80 pixels
        - Custom styling including gray background, Segoe UI font at 10pt
        - Read-only text display initialized with "Welcome to Pareto!"
        The message frame provides feedback and status updates to the user.
        """

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
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        # Setup picker and interaction style
        self.picker = vtk.vtkCellPicker()

        self.picker.SetTolerance(0.0005)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Add coordinate axes
        axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)

        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOn()

        self.renderer.ResetCamera()
        self.interactor.Initialize()

    def load_stl_file(self, file_path):
        self.stl_geom = STLGeom.STLGeom(file_path)
        
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
        
        # The next few lines ensure that the highlighting does not disappear at certain angles
        render_window = self.vtkWidget.GetRenderWindow()
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)
        self.renderer.UseDepthPeelingOn()
        self.renderer.SetMaximumNumberOfPeels(100) 
     

        # Setup interactions
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

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

    def on_left_button_press(self, obj, event):
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.tri_highlight = [False] * self.stl_geom.stl_n_triangles
        cell_id = self.picker.GetCellId()
        if self.stl_geom and cell_id >= 0:
            self.stl_geom.highlight_triangles_recursive(
                seed_triangle=cell_id,
                depth=500,  
                cutoff_angle_degrees=30
            )
            self.update_highlights()

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