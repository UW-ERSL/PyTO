import sys
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
import pyvista as pv
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
            return QIcon()
        return QIcon(icon_file)

    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self)
        dialog.exec_()
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
                stl_geom = STLGeom(file_path)
                stl_geom.plotGeometry(plotter = self.parent.plotter)
                
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load geometry: {str(e)}")

    def setup_geometry_info(self, mesh, file_path):
        """Display geometry info as a text overlay in the PyVista plotter."""
        if mesh is None:
            return
        # Calculate metrics
        volume = mesh.volume if hasattr(mesh, "volume") else 0.0
        area = mesh.area if hasattr(mesh, "area") else 0.0
        bounds = mesh.bounds if hasattr(mesh, "bounds") else None 
        length_unit = self.parent.settings.get_length_unit_string() if hasattr(self.parent, "settings") else "m"

        info_lines = [
            f"Model: {os.path.basename(file_path)}",
            f"Volume: {volume:.2e} {length_unit}³",
            f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
            f"Surface Area: {area:.2e} {length_unit}²"
        ]

        #self.parent.plotter.remove_actor("geometry_info", reset_camera=False)
        # Add text overlay
        self.parent.plotter.add_text(
            "\n".join(info_lines),
            position="upper_left",
            font_size=12,
            color="black",
            name="geometry_info",
            font="arial",
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())