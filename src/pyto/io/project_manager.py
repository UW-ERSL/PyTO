import os
import sys
import json
import time
import numpy as np
import argparse
import linear_solvers as lin_solv
import hex_structural_fea as fea
import mat_lib
import bound_cond
from stl_reader import STLGeom
from hex_mesher import HexMesher
import hex_thermal_fea
import glob

class ProjectManager:
    def __init__(self):
        pass

    def execute_structural_analysis(self, project_data, stl_path, n_elements, output_dir):
        start_time = time.time()
        print(f"Running structural analysis...")

        mesh = HexMesher()
        mesh.createMeshFromSTLFile(stl_path, nElemsDesired=n_elements)
        mesh.createEdofMatStructural()

        stl_geom = STLGeom(stl_path)

        bc_data = project_data['structuralBC']
        fixed_faces = bc_data.get('fixed_faces_indices', [])
        load_faces_groups = bc_data.get('load_faces_indices', [])
        load_forces = bc_data.get('load_forces', [])

        boundary_nodes = mesh.get_boundary_nodes()
        boundary_points = mesh.node_xyz[boundary_nodes]
        tol = min(mesh.elem_size) * 0.9

        fixed_nodes = set()
        for face_idx in fixed_faces:
            if face_idx < stl_geom.stl_n_triangles:
                distances = stl_geom.find_points_triangle_distances_vectorized(boundary_points, face_idx)
                close_nodes_mask = distances < tol
                nodes_for_face = boundary_nodes[close_nodes_mask]
                fixed_nodes.update(nodes_for_face)

        load_nodes_groups = []
        for face_indices in load_faces_groups:
            load_nodes = set()
            for face_idx in face_indices:
                if face_idx < stl_geom.stl_n_triangles:
                    distances = stl_geom.find_points_triangle_distances_vectorized(boundary_points, face_idx)
                    close_nodes_mask = distances < tol
                    load_nodes.update(boundary_nodes[close_nodes_mask])
            load_nodes_groups.append(load_nodes)

        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])

        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

        force = np.zeros(3*mesh.num_nodes)
        for nodes, force_vector in zip(load_nodes_groups, load_forces):
            if nodes:
                force_per_node = np.array(force_vector) / len(nodes)
                for node in nodes:
                    force[3*node:3*node + 3] += force_per_node

        bc = bound_cond.BC(
            force=force,
            fixed_dofs=fixed_dofs,
            dirichlet_values=dirichlet_values
        )

        mat_data = project_data['material_data']
        if 'properties' in mat_data:
            mat_prop = mat_lib.Material(
                name="Applied_Material",
                youngs_modulus=mat_data['properties'].get("Young's Modulus", 210e9),
                poissons_ratio=mat_data['properties'].get("Poisson's Ratio", 0.3),
                mass_density=mat_data['properties'].get('Density', 7800.0),
                thermal_conductivity=mat_data['properties'].get('Thermal Conductivity', 50.0),
                specific_heat=mat_data['properties'].get('Specific Heat Capacity', 450.0),
                thermal_expansion=mat_data['properties'].get('Thermal Expansion', 12e-6),
                cost=mat_data['properties'].get('Price', 1.0),
                yield_strength=mat_data['properties'].get('Yield Strength', 250e6)
            )
        else:
            mat_prop = mat_lib.Material(
                name="Applied_Material",
                youngs_modulus=mat_data.get('young_modulus', 210e9),
                poissons_ratio=mat_data.get('poisson_ratio', 0.3),
                mass_density=mat_data.get('density', 7800.0),
                thermal_conductivity=mat_data.get('thermal_conductivity', 50.0),
                specific_heat=mat_data.get('specific_heat', 450.0),
                thermal_expansion=mat_data.get('thermal_expansion', 12e-6),
                cost=mat_data.get('price', 1.0),
                yield_strength=mat_data.get('yield_strength', 250e6)
            )

        solver_type = project_data.get('analysis_settings', {}).get('solver_type', 'PARDISO')
        solver_map = {
            "PARDISO": lin_solv.Solvers.PARDISO,
            "DPCG": lin_solv.Solvers.DPCG,
            "PCG": lin_solv.Solvers.PCG,
            "PYAMG": lin_solv.Solvers.PYAMG,
            "SPSOLVE": lin_solv.Solvers.SPSOLVE
        }
        solver = solver_map.get(solver_type, lin_solv.Solvers.PARDISO)

        fe_solver = fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver
        )

        solve_time_start = time.time()
        u = np.asarray(fe_solver.solve())
        solve_time = time.time() - solve_time_start

        fe_solver.postprocess()  # Calculate stresses

        total_time = time.time() - start_time

        max_deformation = fe_solver.max_deformation
        max_vonmises = np.max(fe_solver.vonMisesStress)

        return {
            'solution_time': solve_time,
            'total_time': total_time,
            'max_displacement': max_deformation,
            'max_stress': max_vonmises
        }

    def execute_thermal_analysis(self, project_data, stl_path, n_elements, output_dir):
        start_time = time.time()
        print(f"Running thermal analysis...")

        mesh = HexMesher()
        mesh.createMeshFromSTLFile(stl_path, nElemsDesired=n_elements)
        mesh.createEdofMatThermal()

        stl_geom = STLGeom(stl_path)

        boundary_nodes = mesh.get_boundary_nodes()
        boundary_points = mesh.node_xyz[boundary_nodes]
        tol = min(mesh.elem_size) * 0.9

        thermal_bc_data = project_data['thermalBC']

        fixed_temps = {}
        for temp_group in thermal_bc_data.get('fixed_temps', []):
            triangle_indices = temp_group.get('triangles', [])
            temperature = temp_group.get('temperature', 300.0)

            for face_idx in triangle_indices:
                if face_idx < stl_geom.stl_n_triangles:
                    distances = stl_geom.find_points_triangle_distances_vectorized(boundary_points, face_idx)
                    close_nodes_mask = distances < tol
                    nodes_for_face = boundary_nodes[close_nodes_mask]

                    for node in nodes_for_face:
                        fixed_temps[node] = temperature

        thermal_load = np.zeros(mesh.num_nodes)

        for flux_group in thermal_bc_data.get('heat_sources', []):
            triangle_indices = flux_group.get('triangles', [])
            heat_flux = flux_group.get('heat_flux', 0.0)

            flux_nodes = set()
            for face_idx in triangle_indices:
                if face_idx < stl_geom.stl_n_triangles:
                    distances = stl_geom.find_points_triangle_distances_vectorized(boundary_points, face_idx)
                    close_nodes_mask = distances < tol
                    flux_nodes.update(boundary_nodes[close_nodes_mask])

            if flux_nodes:
                flux_per_node = heat_flux / len(flux_nodes)
                for node in flux_nodes:
                    thermal_load[node] += flux_per_node

        for heat_group in thermal_bc_data.get('total_heat_sources', []):
            triangle_indices = heat_group.get('triangles', [])
            total_heat = heat_group.get('total_heat', 0.0)

            heat_nodes = set()
            for face_idx in triangle_indices:
                if face_idx < stl_geom.stl_n_triangles:
                    distances = stl_geom.find_points_triangle_distances_vectorized(boundary_points, face_idx)
                    close_nodes_mask = distances < tol
                    heat_nodes.update(boundary_nodes[close_nodes_mask])

            if heat_nodes:
                heat_per_node = total_heat / len(heat_nodes)
                for node in heat_nodes:
                    thermal_load[node] += heat_per_node

        fixed_dofs = []
        fixed_values = []
        for node, temp in fixed_temps.items():
            fixed_dofs.append(node)
            fixed_values.append(temp)

        fixed_dofs = np.array(fixed_dofs, dtype=int)
        dirichlet_values = np.array(fixed_values, dtype=float)

        bc = bound_cond.BC(
            force=thermal_load,
            fixed_dofs=fixed_dofs,
            dirichlet_values=dirichlet_values
        )

        mat_data = project_data['material_data']

        if 'properties' in mat_data:
            thermal_mat_prop = mat_lib.Material(
                name="Applied_Material",
                youngs_modulus=mat_data['properties'].get("Young's Modulus", 210e9),
                poissons_ratio=mat_data['properties'].get("Poisson's Ratio", 0.3),
                mass_density=mat_data['properties'].get('Density', 7800.0),
                thermal_conductivity=mat_data['properties'].get('Thermal Conductivity', 50.0),
                specific_heat=mat_data['properties'].get('Specific Heat Capacity', 450.0),
                thermal_expansion=mat_data['properties'].get('Thermal Expansion', 12e-6),
                cost=mat_data['properties'].get('Price', 1.0),
                yield_strength=mat_data['properties'].get('Yield Strength', 250e6)
            )
        else:
            thermal_mat_prop = mat_lib.Material(
                name="Applied_Material",
                youngs_modulus=mat_data.get('young_modulus', 210e9),
                poissons_ratio=mat_data.get('poisson_ratio', 0.3),
                mass_density=mat_data.get('density', 7800.0),
                thermal_conductivity=mat_data.get('thermal_conductivity', 50.0),
                specific_heat=mat_data.get('specific_heat', 450.0),
                thermal_expansion=mat_data.get('thermal_expansion', 12e-6),
                cost=mat_data.get('price', 1.0),
                yield_strength=mat_data.get('yield_strength', 250e6)
            )

        solver_type = project_data.get('analysis_settings', {}).get('solver_type', 'PARDISO')
        solver_map = {
            "PARDISO": lin_solv.Solvers.PARDISO,
            "DPCG": lin_solv.Solvers.DPCG,
            "PCG": lin_solv.Solvers.PCG,
            "PYAMG": lin_solv.Solvers.PYAMG,
            "SPSOLVE": lin_solv.Solvers.SPSOLVE
        }
        solver = solver_map.get(solver_type, lin_solv.Solvers.PARDISO)

        fe_solver = hex_thermal_fea.HexThermalFEA(
            mesh=mesh,
            mat_prop=thermal_mat_prop,
            bc=bc,
            solver=solver
        )

        solve_time_start = time.time()
        temperatures = np.asarray(fe_solver.solve())
        solve_time = time.time() - solve_time_start

        total_time = time.time() - start_time

        temp_min = np.min(temperatures)
        temp_max = np.max(temperatures)

        return {
            'solution_time': solve_time,
            'total_time': total_time,
            'min_temperature': temp_min,
            'max_temperature': temp_max
        }

    def execute_project(self, project_file, output_dir=None):
        try:
            with open(project_file, 'r') as f:
                project_data = json.load(f)

            stl_file = project_data.get('stl_file_path', '')
            stl_path = None

            if os.path.isabs(stl_file) and os.path.exists(stl_file):
                stl_path = stl_file
            else:
                project_dir = os.path.dirname(os.path.abspath(project_file))
                candidate_path = os.path.join(project_dir, stl_file)
                if os.path.exists(candidate_path):
                    stl_path = candidate_path
                else:
                    model_name = os.path.basename(project_dir)
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    pyto_root = os.path.dirname(script_dir)
                    models_path = os.path.join(pyto_root, "Models")
                    candidate_path = os.path.join(models_path, model_name, stl_file)
                    if os.path.exists(candidate_path):
                        stl_path = candidate_path
                    else:
                        candidate_path = os.path.join(models_path, model_name, f"{model_name}.STL")
                        if os.path.exists(candidate_path):
                            stl_path = candidate_path

            if not stl_path:
                print(f"Error: STL file not found: {stl_file}")
                print(f"Tried searching in project directory and in ../Models/{os.path.basename(os.path.dirname(project_file))}/")
                return False

            print(f"Using STL file: {stl_path}")

            n_elements = project_data.get('analysis_settings', {}).get('n_elements', 10000)

            # Check for valid structural BCs
            has_structural = False
            if 'structuralBC' in project_data:
                bc = project_data['structuralBC']
                if bc.get('fixed_faces_indices') or bc.get('load_faces_indices'):
                    has_structural = True

            # Check for valid thermal BCs
            has_thermal = False
            if 'thermalBC' in project_data:
                tbc = project_data['thermalBC']
                if tbc.get('fixed_temps') or tbc.get('heat_sources') or tbc.get('total_heat_sources'):
                    # Only solve if at least one BC list is non-empty
                    if (tbc.get('fixed_temps') and len(tbc.get('fixed_temps')) > 0) or \
                    (tbc.get('heat_sources') and len(tbc.get('heat_sources')) > 0) or \
                    (tbc.get('total_heat_sources') and len(tbc.get('total_heat_sources')) > 0):
                        has_thermal = True

            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(project_file), "results")

            if has_structural:
                results = self.execute_structural_analysis(project_data, stl_path, n_elements, output_dir)
                print(f"Time: {results['solution_time']:.2f} s")
                print(f"Maximum deformation: {results['max_displacement']:.4e}")
                print(f"Maximum von Mises stress: {results['max_stress']:.4e}")

            if has_thermal:
                results = self.execute_thermal_analysis(project_data, stl_path, n_elements, output_dir)
                print(f"Time: {results['solution_time']:.2f} s")
                print(f"Temperature range: {results['min_temperature']:.2f}K to {results['max_temperature']:.2f}K")

            if not has_structural and not has_thermal:
                print("No valid boundary conditions found for structural or thermal analysis.")

            return True
        except Exception as e:
            print(f"Error executing project: {e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    def run_from_cli(cls):
        parser = argparse.ArgumentParser(description="Execute PyTO project file")
        parser.add_argument("project", help="Path to .pyto project file")
        parser.add_argument("--output", type=str, help="Output directory for results")

        args = parser.parse_args()

        if not os.path.exists(args.project):
            print(f"Error: Project file not found: {args.project}")
            sys.exit(1)

        manager = cls()
        success = manager.execute_project(args.project, args.output)
        sys.exit(0 if success else 1)

def execute_all_projects_in_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), "Models")
    # Search recursively for all .pyto files
    pyto_files = glob.glob(os.path.join(models_dir, "**", "*.pyto"), recursive=True)
    print(f"Found {len(pyto_files)} project files in {models_dir}")
    manager = ProjectManager()
    for project_file in pyto_files:
        print(f"\n--- Executing project: {project_file} ---")
        manager.execute_project(project_file)

if __name__ == "__main__":
    # If no arguments, run all projects in ../Models
    if len(sys.argv) == 1:
        execute_all_projects_in_models()
    else:
        ProjectManager.run_from_cli()