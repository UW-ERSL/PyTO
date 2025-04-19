import os
import sys
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import argparse
from glob import glob
import traceback 
import scipy.sparse as scipy_sparse
import linear_solvers as lin_solv
import struct_fea as fea
import mat_lib
import bound_cond
from STLGeom import STLGeom
from mesher import Mesher
import plots
import thermal_fea

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Add src directory to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Configuration paths
PROJECTS_DIR = "C:\\Semester_5\\Research\\Test\\projects"
MODELS_DIR = "C:\\Semester_5\\Research\\Test\\models"
RESULTS_DIR = "C:\\Semester_5\\Research\\Test\\results"


# ======================================================== Validation utility functions =========================================================

def validate_structural_bc(bc_data, stl_geom):
    """
    Validate structural boundary conditions to ensure they're properly defined
    
    Args:
        bc_data: Boundary condition data from project file
        stl_geom: STL geometry object
        
    Returns:
        bool: True if boundary conditions are valid, False otherwise
    """
    if 'fixed_faces_indices' not in bc_data or not bc_data.get('fixed_faces_indices'):
        return False
        
    if 'load_faces_indices' not in bc_data or not bc_data.get('load_faces_indices'):
        return False
    
    # Check if load faces exist and have corresponding forces
    if not 'load_forces' in bc_data or len(bc_data['load_faces_indices']) != len(bc_data['load_forces']):
        return False
        
    return True


def validate_thermal_bc(bc_data):
    """
    Validate thermal boundary conditions to ensure they're properly defined
    
    Args:
        bc_data: Thermal boundary condition data from project file
        
    Returns:
        bool: True if thermal boundary conditions are valid, False otherwise
    """
    # Check if fixed temperatures, heat sources, or total heat sources are defined
    fixed_temps = bc_data.get('fixed_temps', [])
    heat_sources = bc_data.get('heat_sources', [])
    total_heat_sources = bc_data.get('total_heat_sources', [])
    
    has_bcs = len(fixed_temps) > 0 or len(heat_sources) > 0 or len(total_heat_sources) > 0
    
    return has_bcs

#=================================================================================================


#================================================================================================
class BaseAnalysis:
    """Base class for FEA analysis."""
    
    def __init__(self, project_file, n_elements=10000, output_dir=None):
        """Initialize the analysis with project file and settings."""
        self.project_file = project_file
        self.n_elements = n_elements
        self.start_time = time.time()
        
        # Set output directory
        if not output_dir:
            self.output_dir = os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(project_file))[0])
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.base_name = os.path.splitext(os.path.basename(project_file))[0]
        self.log_file = os.path.join(self.output_dir, f"{self.analysis_type}_analysis_log.txt")
        self.mesh = None
        self.stl_path = None
        self.stl_geom = None
        self.solver = lin_solv.Solvers.PARDISO
        
    def log_print(self, message):
        """Write message to both console and log file."""
        print(message)
        with open(self.log_file, 'a') as log:
            log.write(message + "\n")
            
    def load_project_data(self):
        """Load and validate project data."""
        try:
            with open(self.project_file, 'r') as f:
                self.project_data = json.load(f)
            
            # Get STL file path
            stl_file = self.project_data.get('stl_file_path', '')
            if not os.path.isabs(stl_file):
                self.stl_path = os.path.join(MODELS_DIR, stl_file)
                if not os.path.exists(self.stl_path):
                    self.stl_path = os.path.join(os.path.dirname(os.path.abspath(self.project_file)), stl_file)
            
            if not os.path.exists(self.stl_path):
                self.log_print(f"STL file not found: {self.stl_path}")
                return False
                
            # Get material properties
            if 'material_data' not in self.project_data:
                self.log_print("No material data found in project file")
                return False
                
            return True
        except Exception as e:
            self.log_print(f"Error loading project file: {e}")
            return False
            
    def generate_mesh(self):
        """Generate mesh from STL file."""
        try:
            self.log_print(f"Generating mesh with target {self.n_elements} elements...")
            self.mesh = Mesher()
            self.mesh.createMeshFromSTLFile(self.stl_path, nElemsDesired=self.n_elements)
            self.mesh_type_setup()
            
            # Load STL geometry
            self.stl_geom = STLGeom(self.stl_path)
            
            if hasattr(self.mesh, 'num_elems'):
                if self.mesh.num_elems != self.n_elements:
                    self.log_print(f"Note: Requested {self.n_elements} elements, but mesh was created with {self.mesh.num_elems} elements")
                    
            return True
        except Exception as e:
            self.log_print(f"Error generating mesh: {e}")
            return False
            
    def mesh_type_setup(self):
        pass
        
    def get_boundary_nodes(self):
        """Get boundary nodes from mesh."""
        try:
            self.boundary_nodes = self.mesh.get_boundary_nodes()
            self.boundary_points = self.mesh.node_xyz[self.boundary_nodes]
            self.tol = min(self.mesh.elem_size) * 0.9
            return True
        except Exception as e:
            self.log_print(f"Error getting boundary nodes: {e}")
            return False
            
    def validate_boundary_conditions(self):
        pass
        
    def process_boundary_conditions(self):
        pass
        
    def create_solver(self):
        pass
        
    def run_analysis(self):
        pass
        
    def generate_plots(self):
        pass
        
    def save_results(self):
        pass
        
    def execute(self):
        """Execute the full analysis pipeline."""
        # Initialize log file
        with open(self.log_file, 'w') as log:
            log.write(f"PyTO {self.analysis_type.capitalize()} Analysis Log\n")
            log.write("=======================\n\n")
            log.write(f"Project file: {os.path.basename(self.project_file)}\n")
            log.write(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        self.log_print(f"\n{'='*60}")
        self.log_print(f"PROCESSING {self.analysis_type.upper()} ANALYSIS: {os.path.basename(self.project_file)}")
        self.log_print(f"{'='*60}")
        
        # Run analysis pipeline
        if not self.load_project_data():
            return False
            
        if not self.validate_boundary_conditions():
            return False
            
        if not self.generate_mesh():
            return False
            
        if not self.get_boundary_nodes():
            return False
            
        if not self.process_boundary_conditions():
            return False
            
        if not self.create_solver():
            return False
            
        if not self.run_analysis():
            return False
            
        self.generate_plots()
        self.save_results()
        
        self.log_print(f"{self.analysis_type.capitalize()} analysis completed in {time.time() - self.start_time:.2f} seconds")
        return True
#==================================================================================================================================


#==================================================================================================================================
class StructuralAnalysis(BaseAnalysis):
    """Class for structural FEA analysis."""
    
    def __init__(self, project_file, n_elements=10000, output_dir=None):
        self.analysis_type = "structural"
        super().__init__(project_file, n_elements, output_dir)
        
    def mesh_type_setup(self):
        """Setup mesh for structural analysis."""
        self.mesh.createEdofMatStructural()
        
    def validate_boundary_conditions(self):
        """Validate structural boundary conditions."""
        if 'structuralBC' not in self.project_data:
            self.log_print("No structural boundary conditions found in project file")
            return False
            
        self.bc_data = self.project_data['structuralBC']
        
        # Use validation utility function
        if not validate_structural_bc(self.bc_data, None):  # STL geom not needed here
            self.log_print("Invalid structural boundary conditions")
            return False
            
        # Extract validated data for processing
        self.fixed_faces = self.bc_data.get('fixed_faces_indices', [])
        self.load_faces_groups = self.bc_data.get('load_faces_indices', [])
        self.load_forces = self.bc_data.get('load_forces', [])
            
        return True
        
    def process_boundary_conditions(self):
        """Process structural boundary conditions."""
        try:
            self.fixed_nodes_dict = {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
            self.load_nodes_groups = []
            
            # Process fixed faces
            if self.fixed_faces:
                batch_size = 50
                for start_idx in range(0, len(self.fixed_faces), batch_size):
                    batch_end = min(start_idx + batch_size, len(self.fixed_faces))
                    batch_faces = self.fixed_faces[start_idx:batch_end]
                    
                    for face_idx in batch_faces:
                        if face_idx < self.stl_geom.stl_n_triangles:
                            distances = self.stl_geom.find_points_triangle_distances_vectorized(self.boundary_points, face_idx)
                            close_nodes_mask = distances < self.tol
                            nodes_for_face = self.boundary_nodes[close_nodes_mask]
                            self.fixed_nodes_dict['xyz'].update(nodes_for_face)
            
            # Process load groups
            for face_indices, force_values in zip(self.load_faces_groups, self.load_forces):
                if not face_indices or not force_values:
                    self.load_nodes_groups.append(set())
                    continue
                    
                load_nodes = set()
                batch_size = 50
                for start_idx in range(0, len(face_indices), batch_size):
                    batch_end = min(start_idx + batch_size, len(face_indices))
                    batch_faces = face_indices[start_idx:batch_end]
                    
                    for face_idx in batch_faces:
                        if face_idx < self.stl_geom.stl_n_triangles:
                            distances = self.stl_geom.find_points_triangle_distances_vectorized(self.boundary_points, face_idx)
                            close_nodes_mask = distances < self.tol
                            load_nodes.update(self.boundary_nodes[close_nodes_mask])
                
                self.load_nodes_groups.append(load_nodes)
            
            # Create boundary conditions
            self.bc = self.process_data_for_solver()
            return True
        except Exception as e:
            self.log_print(f"Error processing boundary conditions: {e}")
            return False
            
    def process_data_for_solver(self):
        """Process mesh data and create boundary conditions for structural solver."""
        # Process fixed nodes
        fixed_dofs = []
        if 'xyz' in self.fixed_nodes_dict:
            for node in self.fixed_nodes_dict['xyz']:
                fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
                if hasattr(self.mesh, 'node_indices'):
                    self.mesh.node_indices[node, 3] = 1
        
        if 'x' in self.fixed_nodes_dict:
            for node in self.fixed_nodes_dict['x']:
                fixed_dofs.append(3*node)
                if hasattr(self.mesh, 'node_indices'):
                    self.mesh.node_indices[node, 3] = 2
        
        if 'y' in self.fixed_nodes_dict:
            for node in self.fixed_nodes_dict['y']:
                fixed_dofs.append(3*node + 1)
                if hasattr(self.mesh, 'node_indices'):
                    self.mesh.node_indices[node, 3] = 3
        
        if 'z' in self.fixed_nodes_dict:
            for node in self.fixed_nodes_dict['z']:
                fixed_dofs.append(3*node + 2)
                if hasattr(self.mesh, 'node_indices'):
                    self.mesh.node_indices[node, 3] = 4
        
        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
        
        # Process loads
        force = np.zeros(3*self.mesh.num_nodes)
        for nodes, force_vector in zip(self.load_nodes_groups, self.load_forces):
            if nodes:
                force_per_node = np.array(force_vector) / len(nodes)
                for node in nodes:
                    force[3*node:3*node + 3] += force_per_node
                    if hasattr(self.mesh, 'node_indices'):
                        self.mesh.node_indices[node, 3] = 5

        # Create boundary conditions
        return bound_cond.BC(
            force=force,
            fixed_dofs=fixed_dofs,
            dirichlet_values=dirichlet_values
        )
        
    def create_solver(self):
        """Create structural FEA solver."""
        try:
            # Get material properties
            mat_data = self.project_data['material_data']
            self.mat_prop = mat_lib.StructuralMaterial(
                youngs_modulus=mat_data.get('young_modulus', 210e9),
                poissons_ratio=mat_data.get('poisson_ratio', 0.3)
            )
            
            self.fe_solver = fea.StructFEA(
                mesh=self.mesh,
                mat_prop=self.mat_prop,
                bc=self.bc,
                solver=self.solver
            )
            return True
        except Exception as e:
            self.log_print(f"Error creating FEA solver: {e}")
            return False
            
    def run_analysis(self):
        """Run structural analysis."""
        try:
            solve_time_start = time.time()
            self.u = np.asarray(self.fe_solver.solve())
            self.solve_time = time.time() - solve_time_start
            
            self.delta = np.sqrt(self.u[0::3]**2 + self.u[1::3]**2 + self.u[2::3]**2)
            self.delta_max = np.max(self.delta)
            
            self.log_print("\n" + "="*50)
            self.log_print("FEA Results:")
            self.log_print(f"Number of nodes: {self.mesh.num_nodes}")
            self.log_print(f"Number of elements: {self.mesh.num_elems}")
            self.log_print(f"Solution time: {self.solve_time:.2f} seconds")
            self.log_print(f"Maximum displacement: {self.delta_max:.4e} m")
            self.log_print("="*50)
            return True
        except Exception as e:
            self.log_print(f"Error during solver execution: {e}")
            traceback.print_exc() 
            return False
            
    def generate_plots(self):
        """Generate structural analysis plots."""
        try:
            plt.figure(figsize=(12, 8))
            plots.plotMesh(self.mesh, bc=None, u=self.u,
                        title=f'Maximum displacement: {self.delta_max:.3e} m')
            plt.savefig(os.path.join(self.output_dir, f"{self.base_name}_displacement.png"), dpi=300)
            plt.close('all')
        except Exception as e:
            self.log_print(f"Error generating plots: {e}")
            
    def save_results(self):
        """Save structural analysis results."""
        try:
            np.save(os.path.join(self.output_dir, f"{self.base_name}_displacement.npy"), self.u)
            
            mat_data = self.project_data['material_data']
            with open(os.path.join(self.output_dir, f"{self.base_name}_summary.txt"), 'w') as f:
                f.write("PyTO FEA Results Summary\n")
                f.write("=======================\n\n")
                f.write(f"Project file: {os.path.basename(self.project_file)}\n")
                f.write(f"STL model: {os.path.basename(self.stl_path)}\n\n")
                f.write(f"Material: {mat_data.get('name', 'Unnamed')}\n")
                f.write(f"Young's modulus: {self.mat_prop.youngs_modulus:.3e} Pa\n")
                f.write(f"Poisson's ratio: {self.mat_prop.poissons_ratio:.3f}\n\n")
                f.write(f"Mesh nodes: {self.mesh.num_nodes}\n")
                f.write(f"Mesh elements: {self.mesh.num_elems}\n\n")
                f.write(f"Load case: {', '.join([str(f) for f in self.load_forces])}\n\n")
                f.write(f"Solver: {self.fe_solver.solver.name}\n")
                f.write(f"Solution time: {self.solve_time:.2f} seconds\n\n")
                f.write(f"Maximum displacement: {self.delta_max:.6e} m\n")
        except Exception as e:
            self.log_print(f"Error saving results: {e}")
#==================================================================================================================================


#==================================================================================================================================
class ThermalAnalysis(BaseAnalysis):
    """Class for thermal FEA analysis."""
    
    def __init__(self, project_file, n_elements=10000, output_dir=None):
        self.analysis_type = "thermal"
        super().__init__(project_file, n_elements, output_dir)
        
    def mesh_type_setup(self):
        """Setup mesh for thermal analysis."""
        self.mesh.createEdofMatThermal()
        
    def validate_boundary_conditions(self):
        """Validate thermal boundary conditions."""
        if 'thermalBC' not in self.project_data:
            self.log_print("No thermal boundary conditions found in project file")
            return False
            
        self.thermal_bc_data = self.project_data['thermalBC']
        
        # Use validation utility function
        if not validate_thermal_bc(self.thermal_bc_data):
            self.log_print("Invalid thermal boundary conditions")
            return False
            
        # Extract validated data for processing
        self.fixed_temps_data = self.thermal_bc_data.get('fixed_temps', [])
        self.heat_sources_data = self.thermal_bc_data.get('heat_sources', [])
        self.total_heat_sources_data = self.thermal_bc_data.get('total_heat_sources', [])
            
        return True
        
    def process_boundary_conditions(self):
        """Process thermal boundary conditions."""
        try:
            # Process fixed temperature triangles (Dirichlet BC)
            self.fixed_temps = {}
            
            # Process fixed temperature boundary conditions
            for temp_group in self.fixed_temps_data:
                triangle_indices = temp_group.get('triangles', [])
                temperature = temp_group.get('temperature', 300.0)
                
                batch_size = 50
                for start_idx in range(0, len(triangle_indices), batch_size):
                    batch_end = min(start_idx + batch_size, len(triangle_indices))
                    batch_faces = triangle_indices[start_idx:batch_end]
                    
                    for face_idx in batch_faces:
                        if face_idx < self.stl_geom.stl_n_triangles:
                            distances = self.stl_geom.find_points_triangle_distances_vectorized(self.boundary_points, face_idx)
                            close_nodes_mask = distances < self.tol
                            nodes_for_face = self.boundary_nodes[close_nodes_mask]
                            
                            for node in nodes_for_face:
                                self.fixed_temps[node] = temperature
            
            # Process heat flux boundary conditions
            self.heat_flux_groups = []
            self.heat_flux_values = []
            
            for flux_group in self.heat_sources_data:
                triangle_indices = flux_group.get('triangles', [])
                heat_flux = flux_group.get('heat_flux', 0.0)
                
                heat_flux_nodes = set()
                batch_size = 50
                for start_idx in range(0, len(triangle_indices), batch_size):
                    batch_end = min(start_idx + batch_size, len(triangle_indices))
                    batch_faces = triangle_indices[start_idx:batch_end]
                    
                    for face_idx in batch_faces:
                        if face_idx < self.stl_geom.stl_n_triangles:
                            distances = self.stl_geom.find_points_triangle_distances_vectorized(self.boundary_points, face_idx)
                            close_nodes_mask = distances < self.tol
                            heat_flux_nodes.update(self.boundary_nodes[close_nodes_mask])
                
                self.heat_flux_groups.append(heat_flux_nodes)
                self.heat_flux_values.append(heat_flux)
            
            # Process total heat boundary conditions
            self.total_heat_groups = []
            self.total_heat_values = []
            
            for heat_group in self.total_heat_sources_data:
                triangle_indices = heat_group.get('triangles', [])
                total_heat = heat_group.get('total_heat', 0.0)
                
                total_heat_nodes = set()
                batch_size = 50
                for start_idx in range(0, len(triangle_indices), batch_size):
                    batch_end = min(start_idx + batch_size, len(triangle_indices))
                    batch_faces = triangle_indices[start_idx:batch_end]
                    
                    for face_idx in batch_faces:
                        if face_idx < self.stl_geom.stl_n_triangles:
                            distances = self.stl_geom.find_points_triangle_distances_vectorized(self.boundary_points, face_idx)
                            close_nodes_mask = distances < self.tol
                            total_heat_nodes.update(self.boundary_nodes[close_nodes_mask])
                
                self.total_heat_groups.append(total_heat_nodes)
                self.total_heat_values.append(total_heat)
            
            # Create thermal boundary conditions
            self.bc = self.process_data_for_thermal()
            return True
        except Exception as e:
            self.log_print(f"Error processing thermal boundary conditions: {e}")
            return False
            
    def process_data_for_thermal(self):
        """Process mesh data and create boundary conditions for thermal solver."""
        # Process fixed temperature nodes
        fixed_dofs = []
        fixed_values_list = []
        for node, temp in self.fixed_temps.items():
            fixed_dofs.append(node)  # one dof per node in thermal analysis
            fixed_values_list.append(temp)
            if hasattr(self.mesh, 'node_indices'):
                self.mesh.node_indices[node, 3] = 1  # flag for fixed temperature
        
        fixed_dofs = np.array(fixed_dofs, dtype=int)
        dirichlet_values = np.array(fixed_values_list, dtype=float)
        
        # Process thermal loads 
        thermal_load = np.zeros(self.mesh.num_nodes)  # one load per node
        
        # Apply heat flux loads
        for nodes, flux_value in zip(self.heat_flux_groups, self.heat_flux_values):
            if nodes:
                flux_per_node = float(flux_value) / len(nodes)
                for node in nodes:
                    thermal_load[node] += flux_per_node
                    if hasattr(self.mesh, 'node_indices'):
                        self.mesh.node_indices[node, 3] = 5  # flag for heat flux load
        
        # Apply total heat loads
        for nodes, heat_value in zip(self.total_heat_groups, self.total_heat_values):
            if nodes:
                heat_per_node = float(heat_value) / len(nodes)
                for node in nodes:
                    thermal_load[node] += heat_per_node
                    if hasattr(self.mesh, 'node_indices'):
                        self.mesh.node_indices[node, 3] = 6  # flag for total heat load
        
        # Create thermal boundary conditions
        return bound_cond.BC(
            force=thermal_load,
            fixed_dofs=fixed_dofs,
            dirichlet_values=dirichlet_values
        )
        
    def create_solver(self):
        """Create thermal FEA solver."""
        try:
            # Get material properties
            mat_data = self.project_data['material_data']
            
            # Create thermal material properties
            self.thermal_mat_prop = mat_lib.ThermalMaterial(
                thermal_conductivity=mat_data.get('thermal_conductivity', 50.0),
                specific_heat=mat_data.get('specific_heat', 450.0),
                mass_density=mat_data.get('density', 7800.0)
            )
            
            self.fe_solver = thermal_fea.ThermalFEA(
                mesh=self.mesh,
                mat_prop=self.thermal_mat_prop,
                bc=self.bc,
                solver=self.solver
            )
            return True
        except Exception as e:
            self.log_print(f"Error creating thermal FEA solver: {e}")
            return False
            
    def run_analysis(self):
        """Run thermal analysis."""
        try:
            solve_time_start = time.time()
            self.temperatures = np.asarray(self.fe_solver.solve())
            self.solve_time = time.time() - solve_time_start
            
            self.temp_min = np.min(self.temperatures)
            self.temp_max = np.max(self.temperatures)
            
            self.log_print("\n" + "="*50)
            self.log_print("Thermal FEA Results:")
            self.log_print(f"Number of nodes: {self.mesh.num_nodes}")
            self.log_print(f"Number of elements: {self.mesh.num_elems}")
            self.log_print(f"Solution time: {self.solve_time:.2f} seconds")
            self.log_print(f"Temperature range: {self.temp_min:.2f}K to {self.temp_max:.2f}K")
            self.log_print("="*50)
            return True
        except Exception as e:
            self.log_print(f"Error during thermal solver execution: {e}")
            traceback.print_exc() 
            return False
            
    def generate_plots(self):
        """Generate thermal analysis plots."""
        try:
            plt.figure(figsize=(12, 8))
            plots.plotMesh(self.mesh, bc=None, u=self.temperatures,
                        title=f'Temperature Distribution (min: {self.temp_min:.2f}K, max: {self.temp_max:.2f}K)')
            plt.savefig(os.path.join(self.output_dir, f"{self.base_name}_temperature.png"), dpi=300)
            plt.close('all')
        except Exception as e:
            self.log_print(f"Error generating thermal plots: {e}")
            
    def save_results(self):
        """Save thermal analysis results."""
        try:
            np.save(os.path.join(self.output_dir, f"{self.base_name}_temperature.npy"), self.temperatures)
            
            mat_data = self.project_data['material_data']
            with open(os.path.join(self.output_dir, f"{self.base_name}_thermal_summary.txt"), 'w') as f:
                f.write("PyTO Thermal Analysis Results Summary\n")
                f.write("=======================\n\n")
                f.write(f"Project file: {os.path.basename(self.project_file)}\n")
                f.write(f"STL model: {os.path.basename(self.stl_path)}\n\n")
                f.write(f"Material: {mat_data.get('name', 'Unnamed')}\n")
                f.write(f"Thermal conductivity: {self.thermal_mat_prop.thermal_conductivity:.2f} W/(m·K)\n")
                f.write(f"Specific heat: {self.thermal_mat_prop.specific_heat:.2f} J/(kg·K)\n")
                f.write(f"Density: {self.thermal_mat_prop.mass_density:.2f} kg/m³\n\n")
                f.write(f"Mesh nodes: {self.mesh.num_nodes}\n")
                f.write(f"Mesh elements: {self.mesh.num_elems}\n\n")
                f.write(f"Fixed temperature count: {len(self.fixed_temps)}\n")
                f.write(f"Heat flux source count: {sum(len(group) for group in self.heat_flux_groups)}\n")
                f.write(f"Total heat source count: {sum(len(group) for group in self.total_heat_groups)}\n\n")
                f.write(f"Solver: {self.fe_solver.solver.name}\n")
                f.write(f"Solution time: {self.solve_time:.2f} seconds\n\n")
                f.write(f"Minimum temperature: {self.temp_min:.2f}K\n")
                f.write(f"Maximum temperature: {self.temp_max:.2f}K\n")
        except Exception as e:
            self.log_print(f"Error saving thermal results: {e}")
#==================================================================================================================================

#==================================================================================================================================
class AnalysisManager:
    """Class to manage analysis selection and execution."""
    
    @staticmethod
    def determine_analysis_type(project_file):
        """Determine what types of analysis are available in the project file with validation."""
        try:
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            # Check if sections exist
            has_structural_section = 'structuralBC' in project_data
            has_thermal_section = 'thermalBC' in project_data
            
            # Load STL geometry for validating boundary conditions
            stl_file = project_data.get('stl_file_path', '')
            if not os.path.isabs(stl_file):
                stl_path = os.path.join(MODELS_DIR, stl_file)
                if not os.path.exists(stl_path):
                    stl_path = os.path.join(os.path.dirname(os.path.abspath(project_file)), stl_file)
            
            if not os.path.exists(stl_path):
                print(f"STL file not found: {stl_path}")
                return False, False
                
            stl_geom = STLGeom(stl_path)
            
            # Validate boundary conditions
            has_valid_structural = False
            has_valid_thermal = False
            
            if has_structural_section:
                has_valid_structural = validate_structural_bc(project_data['structuralBC'], stl_geom)
                
            if has_thermal_section:
                has_valid_thermal = validate_thermal_bc(project_data['thermalBC'])
            
            return has_valid_structural, has_valid_thermal
        except Exception as e:
            print(f"Error determining analysis type: {e}")
            return False, False
    
    @staticmethod
    def get_user_analysis_choice(project_file):
        """Selection of analysis type."""
        has_structural, has_thermal = AnalysisManager.determine_analysis_type(project_file)
        
        available_types = []
        if has_structural:
            available_types.append("structural")
        if has_thermal:
            available_types.append("thermal")
        
        if not available_types:
            print("\nNo valid analysis types found in the project file.")
            return None
            
        print("\n" + "="*60)
        print("ANALYSIS TYPE SELECTION")
        print("="*60)
        print("\nAvailable analysis types for this project:")
        
        # Always ask for analysis type even if only one is available
        for i, a_type in enumerate(available_types):
            print(f"{i+1}. {a_type.capitalize()} Analysis")
        
        while True:
            try:
                choice = input("\nSelect analysis type (number): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_types):
                    selected_type = available_types[choice_idx]
                    print(f"\nSelected: {selected_type.capitalize()} Analysis")
                    return selected_type
                else:
                    print(f"Please enter a number between 1 and {len(available_types)}")
            except ValueError:
                print("Please enter a valid number")
    
    @staticmethod
    def get_user_mesh_elements():
        """Selection of mesh element count."""
        print("\n" + "="*60)
        print("MESH SETTINGS")
        print("="*60)
        
        while True:
            try:
                elements_input = input("\nEnter number of mesh elements (default: 10000): ")
                if elements_input.strip() == "":
                    return 10000
                else:
                    n_elements = int(elements_input)
                    if n_elements > 0:
                        return n_elements
                    else:
                        print("Number of elements must be positive")
            except ValueError:
                print("Please enter a valid number")
    
    @staticmethod
    def get_user_solver_choice():
        """Selection of solver type."""
        print("\n" + "="*60)
        print("SOLVER SELECTION")
        print("="*60)
        
        solvers = [
            ("PARDISO", lin_solv.Solvers.PARDISO, "PARDISO Direct Solver"),
            ("DPCG", lin_solv.Solvers.DPCG, "Diagonal Preconditioned Conjugate Gradient"),
            ("CG", lin_solv.Solvers.CG, "Conjugate Gradient"),
            ("PYAMG", lin_solv.Solvers.PYAMG, "Algebraic Multigrid"),
            ("SPSOLVE", lin_solv.Solvers.SPSOLVE, "SciPy Sparse Direct Solver")
        ]
        
        print("\nAvailable solvers:")
        for i, (name, _, desc) in enumerate(solvers):
            print(f"{i+1}. {name}: {desc}")
        
        while True:
            try:
                choice = input("\nSelect solver (number, default is 1 for PARDISO): ")
                if choice.strip() == "":
                    return solvers[0][1]  # Return PARDISO enum
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(solvers):
                    selected_solver = solvers[choice_idx][1]
                    print(f"\nSelected solver: {solvers[choice_idx][0]}")
                    return selected_solver
                else:
                    print(f"Please enter a number between 1 and {len(solvers)}")
            except ValueError:
                print("Please enter a valid number")
    
    @staticmethod
    def run_interactive_analysis(project_file, output_dir=None):
        """Run analysis with interactive user input for all parameters."""
        # Make sure directories exist
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Standardize path if not absolute
        if not os.path.isabs(project_file):
            project_file = os.path.join(PROJECTS_DIR, project_file)
            
        if not os.path.exists(project_file):
            print(f"Error: Project file not found: {project_file}")
            return False
            
        # Set output directory
        if not output_dir:
            output_dir = os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(project_file))[0])
        
        # Print project information
        print(f"\n{'='*60}")
        print(f"PROJECT: {os.path.basename(project_file)}")
        print(f"{'='*60}")
        
        # Get user selections
        analysis_type = AnalysisManager.get_user_analysis_choice(project_file)
        if not analysis_type:
            return False
            
        n_elements = AnalysisManager.get_user_mesh_elements()
        solver = AnalysisManager.get_user_solver_choice()
        
        # Confirm selections
        print("\n" + "="*60)
        print("ANALYSIS CONFIGURATION:")
        print(f"Project: {os.path.basename(project_file)}")
        print(f"Analysis Type: {analysis_type.capitalize()}")
        print(f"Mesh Elements: {n_elements}")
        print(f"Solver: {solver.name}")
        print("="*60)
        
        confirm = input("\nProceed with analysis? (Y/n): ")
        if confirm.lower() == 'n':
            print("Analysis cancelled by user")
            return False
            
        # Run analysis with selected parameters
        if analysis_type == 'structural':
            analyzer = StructuralAnalysis(project_file, n_elements, output_dir)
            analyzer.solver = solver
            return analyzer.execute()
        elif analysis_type == 'thermal':
            analyzer = ThermalAnalysis(project_file, n_elements, output_dir)
            analyzer.solver = solver
            return analyzer.execute()
        
        return False
#==================================================================================================================================


#==================================================================================================================================
def main():
    """Parse command line arguments and run the analysis"""
    parser = argparse.ArgumentParser(description='Run FEA from PyTO project files')
    parser.add_argument('-p', '--project', 
                        help='Path to .pyto project file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for results (if omitted, uses RESULTS_DIR)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all available project files')
    
    args = parser.parse_args()
    
    # Make sure directories exist
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # List projects if requested
    if args.list:
        project_files = glob(os.path.join(PROJECTS_DIR, "*.pyto"))
        if not project_files:
            print(f"No .pyto files found in {PROJECTS_DIR}")
        else:
            print(f"\nAvailable project files ({len(project_files)}):")
            for i, pfile in enumerate(project_files):
                print(f"{i+1}. {os.path.basename(pfile)}")
        return 0
    
    # Interactive selection of project if no specific project provided
    if not args.project:
        project_files = glob(os.path.join(PROJECTS_DIR, "*.pyto"))
        if not project_files:
            print(f"No .pyto files found in {PROJECTS_DIR}")
            return 1
            
        print(f"\nAvailable project files ({len(project_files)}):")
        for i, pfile in enumerate(project_files):
            print(f"{i+1}. {os.path.basename(pfile)}")
            
        while True:
            try:
                choice = input("\nSelect project to analyze (number): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(project_files):
                    selected_project = project_files[choice_idx]
                    print(f"\nSelected project: {os.path.basename(selected_project)}")
                    
                    # Run interactive analysis on selected project
                    return 0 if AnalysisManager.run_interactive_analysis(
                        selected_project, 
                        output_dir=args.output
                    ) else 1
                else:
                    print(f"Please enter a number between 1 and {len(project_files)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Process specified project
    else:
        project_path = args.project
        if not os.path.isabs(project_path):
            project_path = os.path.join(PROJECTS_DIR, project_path)
            
        if not os.path.exists(project_path):
            print(f"Error: Project file not found: {project_path}")
            return 1
            
        # Always run in interactive mode
        return 0 if AnalysisManager.run_interactive_analysis(
            project_path, 
            output_dir=args.output
        ) else 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("FEA Automation - Run structural and thermal analysis from PyTO project files")
        print("\nUsage options:")
        print("  1. Run interactively (select from available projects):")
        print(f"     python {os.path.basename(__file__)}")
        print("\n  2. Analyze a specific project:")
        print(f"     python {os.path.basename(__file__)} -p project_name.pyto")
        print("\n  3. List available project files:")
        print(f"     python {os.path.basename(__file__)} -l")
        print("\n  4. Show help message:")
        print(f"     python {os.path.basename(__file__)} -h")
        print("\nOptions:")
        print("  -p, --project FILE        Path to specific .pyto project file")
        print("  -l, --list                List all available project files")
        print("  -o, --output DIR          Directory to save results")
    sys.exit(main())