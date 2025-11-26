"""
Complete Thermo-Elastic Sensitivity Example
Following the user's coding style and BC setup patterns
"""

import numpy as np
import os
import sys
import time

# Import required modules
try:
    from hex_thermal_fea import HexThermalFEA
    from hex_structural_fea import HexStructuralFEA
    from hex_thermoelastic_sensitivity import ThermoElasticSensitivity
    import hex_mesher
    import linear_solvers
    import bound_cond
    import mat_lib
    from topopt_material_model import MaterialModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are in the same directory.")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))


def createThermoElasticBarProblem(nDOFDesired: int = 10000,
                                   hot_temperature: float = 100.0,
                                   cold_temperature: float = 20.0,
                                   tensile_force: float = 1000.0):
    """
    Creates a coupled thermo-elastic problem on a bar.
    
    Problem description:
    - Bar fixed at left end (x=0)
    - Hot temperature applied at left end
    - Cold temperature applied at right end
    - Optional tensile force at right end
    - Temperature gradient causes thermal expansion
    - Combined thermal + mechanical loading
    
    Parameters:
    -----------
    nDOFDesired : int
        Desired number of degrees of freedom
    hot_temperature : float
        Temperature at left end (°C)
    cold_temperature : float
        Temperature at right end (°C)
    tensile_force : float
        Applied tensile force at right end (N)
        
    Returns:
    --------
    tuple : (mesh, mat_prop, bc_thermal, bc_structural, reference_temp)
    """
    
    # Load STL geometry
    stl_file = os.path.join(script_dir, '../Models/Beam/beam.STL')
    
    # Create mesh
    nElemsDesired = nDOFDesired / 3  # Estimate for structural DOFs
    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    
    print("\n" + "="*70)
    print("THERMO-ELASTIC BAR PROBLEM SETUP")
    print("="*70)
    
    # =========================================================================
    # THERMAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create thermal DOF mapping
    mesh.createEdofMatThermal()
    print(f"Thermal DOFs (nodes): {mesh.num_nodes}")
    
    # Fixed temperature at left end (hot)
    hot_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    hot_dofs = hot_nodes.astype(int)
    hot_values = hot_temperature * np.ones_like(hot_dofs, dtype=float)
    
    # Fixed temperature at right end (cold)
    cold_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = xMax plane
    cold_dofs = cold_nodes.astype(int)
    cold_values = cold_temperature * np.ones_like(cold_dofs, dtype=float)
    
    # Combine thermal BCs
    thermal_fixed_dofs = np.concatenate([hot_dofs, cold_dofs])
    thermal_dirichlet_values = np.concatenate([hot_values, cold_values])
    
    # No thermal body forces (heat generation)
    thermal_force = np.zeros(mesh.num_nodes)
    
    bc_thermal = bound_cond.BC(
        force=thermal_force,
        fixed_dofs=thermal_fixed_dofs,
        dirichlet_values=thermal_dirichlet_values
    )
    
    print(f"\nThermal BCs:")
    print(f"  Hot nodes (T={hot_temperature}°C): {len(hot_nodes)}")
    print(f"  Cold nodes (T={cold_temperature}°C): {len(cold_nodes)}")
    print(f"  Temperature gradient: {hot_temperature - cold_temperature}°C")
    
    # =========================================================================
    # STRUCTURAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create structural DOF mapping
    mesh.createEdofMatStructural()
    print(f"\nStructural DOFs: {mesh.num_nodes * 3}")
    
    # Fixed displacement at left end (all directions)
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    fixed_dofs = np.array([
        3 * fixed_nodes,
        3 * fixed_nodes + 1,
        3 * fixed_nodes + 2
    ]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    
    # Mark fixed nodes in mesh
    mesh.node_indices[fixed_nodes, 3] = 1
    
    # Initialize force vector
    force = np.zeros(3 * mesh.num_nodes)
    
    # Apply tensile force at right end if specified
    if abs(tensile_force) > 0:
        load_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = xMax plane
        
        # Classify nodes as corner, edge, or face nodes
        edge_nodes = []
        corner_nodes = []
        face_nodes = []
        
        for node in load_nodes:
            coords = mesh.node_xyz[node]
            num_extremes = 0
            
            # Check if on y extremes
            if (abs(coords[1] - min(mesh.node_xyz[:, 1])) < mesh.elem_size[1] / 2 or
                abs(coords[1] - max(mesh.node_xyz[:, 1])) < mesh.elem_size[1] / 2):
                num_extremes += 1
            
            # Check if on z extremes
            if (abs(coords[2] - min(mesh.node_xyz[:, 2])) < mesh.elem_size[2] / 2 or
                abs(coords[2] - max(mesh.node_xyz[:, 2])) < mesh.elem_size[2] / 2):
                num_extremes += 1
            
            if num_extremes == 2:
                corner_nodes.append(node)
            elif num_extremes == 1:
                edge_nodes.append(node)
            else:
                face_nodes.append(node)
        
        # Convert to arrays
        corner_nodes = np.array(corner_nodes)
        edge_nodes = np.array(edge_nodes)
        face_nodes = np.array(face_nodes)
        
        # Apply forces according to node type (consistent with surface integration)
        force[3 * face_nodes] = 4.0
        force[3 * edge_nodes] = 2.0
        force[3 * corner_nodes] = 1.0
        
        # Normalize to achieve desired total load
        total_load = np.sum(force[3 * load_nodes])
        if total_load > 0:
            force[3 * load_nodes] *= tensile_force / total_load
        
        # Mark loaded nodes
        all_load_nodes = np.union1d(np.union1d(face_nodes, edge_nodes), corner_nodes)
        mesh.node_indices[all_load_nodes, 3] = 2
        
        print(f"\nMechanical Loading:")
        print(f"  Load nodes: {len(load_nodes)}")
        print(f"    Face nodes: {len(face_nodes)}")
        print(f"    Edge nodes: {len(edge_nodes)}")
        print(f"    Corner nodes: {len(corner_nodes)}")
        print(f"  Total tensile force: {tensile_force} N")
    
    bc_structural = bound_cond.BC(
        force=force,
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )
    
    print(f"\nStructural BCs:")
    print(f"  Fixed nodes: {len(fixed_nodes)}")
    print(f"  Fixed DOFs: {len(fixed_dofs)}")
    
    # =========================================================================
    # MATERIAL PROPERTIES
    # =========================================================================
    
    # Get material properties (ensure single material, not list)
    mat_prop = mat_lib.get_material("Steel")
    
    # If mat_prop is returned as a list, extract the single material
    if isinstance(mat_prop, list):
        mat_prop = mat_prop[0]
    
    # Reference temperature for thermal expansion
    reference_temp = cold_temperature  # Use cold temperature as reference
    
    print(f"\nMaterial Properties (Steel):")
    print(f"  Young's modulus: {mat_prop.youngs_modulus:.2e} Pa")
    print(f"  Poisson's ratio: {mat_prop.poissons_ratio:.3f}")
    print(f"  Thermal expansion: {mat_prop.thermal_expansion_coefficient:.2e} /K")
    print(f"  Thermal conductivity: {mat_prop.thermal_conductivity:.2f} W/(m·K)")
    print(f"  Reference temperature: {reference_temp}°C")
    
    # =========================================================================
    # THEORETICAL ESTIMATES
    # =========================================================================
    
    if abs(tensile_force) > 0:
        # Estimate cross-sectional area
        bbox = mesh.bbox
        area = (bbox.y.max - bbox.y.min) * (bbox.z.max - bbox.z.min)
        length = bbox.x.max - bbox.x.min
        
        # Mechanical stress and displacement
        stress_mech = tensile_force / area
        disp_mech = (tensile_force * length) / (mat_prop.youngs_modulus * area)
        
        # Thermal strain and displacement
        delta_T = hot_temperature - cold_temperature
        strain_thermal = mat_prop.thermal_expansion_coefficient * delta_T
        disp_thermal = strain_thermal * length
        
        print(f"\nTheoretical Estimates:")
        print(f"  Cross-sectional area: {area:.4f} m²")
        print(f"  Length: {length:.4f} m")
        print(f"  Mechanical stress: {stress_mech:.2e} Pa")
        print(f"  Mechanical displacement: {disp_mech:.2e} m")
        print(f"  Thermal strain: {strain_thermal:.2e}")
        print(f"  Thermal displacement: {disp_thermal:.2e} m")
        print(f"  Total displacement (approx): {disp_mech + disp_thermal:.2e} m")
    
    print("="*70 + "\n")
    
    return mesh, mat_prop, bc_thermal, bc_structural, reference_temp


def run_coupled_analysis(nDOFDesired: int = 10000,
                         hot_temp: float = 100.0,
                         cold_temp: float = 20.0,
                         tensile_force: float = 1000.0,
                         p_simp: float = 3.0,
                         q_simp: float = 1.0,
                         verify_fd: bool = True):
    """
    Run complete coupled thermo-elastic analysis with sensitivity computation.
    
    Parameters:
    -----------
    nDOFDesired : int
        Target number of DOFs
    hot_temp : float
        Hot temperature (°C)
    cold_temp : float
        Cold temperature (°C)
    tensile_force : float
        Applied tensile force (N)
    p_simp : float
        Structural SIMP penalty
    q_simp : float
        Thermal SIMP penalty
    verify_fd : bool
        Whether to verify with finite differences
    """
    
    # =========================================================================
    # PROBLEM SETUP
    # =========================================================================
    
    mesh, mat_prop, bc_thermal, bc_structural, reference_temp = \
        createThermoElasticBarProblem(
            nDOFDesired=nDOFDesired,
            hot_temperature=hot_temp,
            cold_temperature=cold_temp,
            tensile_force=tensile_force
        )
    
    # Solver
    solver = linear_solvers.Solvers.PARDISO
    
    # =========================================================================
    # THERMAL FEA
    # =========================================================================
    
    print("\n" + "="*70)
    print("THERMAL ANALYSIS")
    print("="*70)
    
    thermal_fea = HexThermalFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_thermal,
        solver=solver,
        thermoElasticReferenceTemperature=reference_temp
    )
    
    # Initialize design variables (uniform density)
    x = 0.5*np.ones(mesh.num_elems) 
    
    # DIAGNOSTIC: Check array sizes before solving
    print(f"\nDiagnostic checks:")
    print(f"  mesh.num_elems: {mesh.num_elems}")
    print(f"  x.shape: {x.shape}")
    print(f"  thermal_fea.elem_stiff.shape: {thermal_fea.elem_stiff.shape}")
    print(f"  thermal_fea.node_idx.shape: {thermal_fea.node_idx.shape}")
    print(f"  Expected node_idx entries: {mesh.num_elems * 64} (8x8 per element)")
    
    # Verify thermal_fea has single material
    if thermal_fea.elem_stiff.shape[0] > 1:
        print(f"  ⚠ WARNING: thermal_fea.elem_stiff has {thermal_fea.elem_stiff.shape[0]} materials")
        print(f"     This may cause assembly issues!")
        print(f"     Solution: Ensure mat_prop is a single Material, not a list")
    
    # Solve thermal problem
    start_time = time.time()
    T = thermal_fea.solve(x, material_model=MaterialModel.SIMP)
    thermal_time = time.time() - start_time
    
    print(f"\nThermal solution computed in {thermal_time:.3f} seconds")
    print(f"Temperature field:")
    print(f"  Min: {T.min():.2f}°C")
    print(f"  Max: {T.max():.2f}°C")
    print(f"  Mean: {T.mean():.2f}°C")
    
    # =========================================================================
    # COMPUTE THERMAL FORCES
    # =========================================================================
    
    print("\n" + "="*70)
    print("THERMAL FORCE COMPUTATION")
    print("="*70)
    
    f_thermal = thermal_fea.get_thermoelastic_force(x, material_model=MaterialModel.SIMP)
    
    print(f"\nThermal forces:")
    print(f"  Min: {f_thermal.min():.3e} N")
    print(f"  Max: {f_thermal.max():.3e} N")
    print(f"  Norm: {np.linalg.norm(f_thermal):.3e} N")
    print(f"  Non-zero entries: {np.count_nonzero(f_thermal)}/{len(f_thermal)}")
    
    # =========================================================================
    # STRUCTURAL FEA
    # =========================================================================
    
    print("\n" + "="*70)
    print("STRUCTURAL ANALYSIS")
    print("="*70)
    
    structural_fea = HexStructuralFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_structural,
        solver=solver,
        thermo_elastic_force=f_thermal
    )
    
    # Solve structural problem
    start_time = time.time()
    d = structural_fea.solve(x, material_model=MaterialModel.SIMP)
    
    structural_time = time.time() - start_time
    
    print(f"\nStructural solution computed in {structural_time:.3f} seconds")
    print(f"Displacement field:")
    print(f"  Min: {d.min():.3e} m")
    print(f"  Max: {d.max():.3e} m")
    print(f"  Max deformation: {structural_fea.max_deformation:.3e} m")
    
    # Compute compliance
    J_compliance = d.T @ structural_fea.stiff_mtrx @ d
    print(f"\nStructural compliance: {J_compliance:.6e} J")
    
    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    
    print(f"External force norm: {np.linalg.norm(structural_fea.bc.force):.6e}")
    print(f"Thermal force norm: {np.linalg.norm(f_thermal):.6e}")
    print(f"Ratio f_thermal/f_ext: {np.linalg.norm(f_thermal)/np.linalg.norm(structural_fea.bc.force):.2f}")
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Initialize sensitivity analyzer
    sensitivity = ThermoElasticSensitivity(thermal_fea, structural_fea)
    
    # Compute analytical sensitivities
    print("\nComputing analytical sensitivities using adjoint method...")


    # === ADD THIS SANITY CHECK HERE ===
    # Assemble K_S for verification
    K_S = sensitivity.assemble_structural_stiffness(x, material_model=MaterialModel.SIMP)

    # Method 1: Matrix form
    J_matrix = d.T @ K_S @ d

    # Method 2: Work form (using equilibrium K*d = f)
    # Get the actual force vector used in the solve
    f_total = structural_fea.total_force  # This should be f_ext + f_thermal
    J_work = d.T @ f_total

    print("\n" + "="*60)
    print("COMPLIANCE VERIFICATION")
    print("="*60)
    print(f"J (matrix form):     {J_matrix:.12e}")
    print(f"J (work form):       {J_work:.12e}")
    print(f"Relative difference: {abs(J_matrix - J_work)/abs(J_matrix):.2e}")
    print("="*60)

    if abs(J_matrix - J_work)/abs(J_matrix) > 1e-10:
        print("WARNING: Matrix and work forms don't match!")
        print("This indicates an equilibrium or assembly issue.")
    else:
        print("✓ Equilibrium satisfied correctly")
    print("="*60 + "\n")
    # === END SANITY CHECK ===

    start_time = time.time()
    dJdx = sensitivity.compute_compliance_sensitivity(
        x=x,
        T=T,
        d=d,
        p=p_simp,
        q=q_simp,
        material_model=MaterialModel.SIMP,
        solver=solver,
        verbose=True
    )
    sensitivity_time = time.time() - start_time
    
    print(f"\nSensitivity computation time: {sensitivity_time:.3f} seconds")
    print(f"\nSensitivity statistics:")
    print(f"  Min: {dJdx.min():.6e}")
    print(f"  Max: {dJdx.max():.6e}")
    print(f"  Mean: {dJdx.mean():.6e}")
    print(f"  Std: {dJdx.std():.6e}")
    print(f"  Positive: {np.sum(dJdx > 0)}/{len(dJdx)}")
    print(f"  Negative: {np.sum(dJdx < 0)}/{len(dJdx)}")
    
    # =========================================================================
    # FINITE DIFFERENCE VERIFICATION
    # =========================================================================
    
    if verify_fd:
        print("\n" + "="*70)
        print("FINITE DIFFERENCE VERIFICATION")
        print("="*70)
        
        # Select elements to verify
        num_verify = min(10, mesh.num_elems)
        element_indices = np.linspace(0, mesh.num_elems - 1, num_verify, dtype=int)
        
        print(f"\nVerifying {num_verify} elements with finite differences...")
        
        start_time = time.time()
        rel_errors, analytical, fd = sensitivity.verify_sensitivity_fd(
            x=x,
            T=T,
            d=d,
            p=p_simp,
            q=q_simp,
            perturbation=1e-4,
            element_indices=element_indices,
            material_model=MaterialModel.SIMP,
            verbose=True
        )
        verification_time = time.time() - start_time
        
        print(f"\nVerification time: {verification_time:.3f} seconds")
        
        # Check verification results
        max_error = np.max(rel_errors)
        mean_error = np.mean(rel_errors)
        
        print(f"\nVerification results:")
        print(f"  Max relative error: {max_error:.2e}")
        print(f"  Mean relative error: {mean_error:.2e}")
        
        if max_error < 1e-4:
            print(f"  ✓ EXCELLENT - Verification passed (< 1e-4)")
        elif max_error < 1e-3:
            print(f"  ✓ GOOD - Verification passed (< 1e-3)")
        elif max_error < 1e-2:
            print(f"  ⚠ ACCEPTABLE - Verification warning (< 1e-2)")
        else:
            print(f"  ✗ POOR - Verification failed (>= 1e-2)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nProblem size:")
    print(f"  Elements: {mesh.num_elems}")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Structural DOFs: {mesh.num_nodes * 3}")
    print(f"  Thermal DOFs: {mesh.num_nodes}")
    
    print(f"\nComputational cost:")
    print(f"  Thermal analysis: {thermal_time:.3f} s")
    print(f"  Structural analysis: {structural_time:.3f} s")
    print(f"  Sensitivity analysis: {sensitivity_time:.3f} s")
    print(f"  Total: {thermal_time + structural_time + sensitivity_time:.3f} s")
    
    print(f"\nResults:")
    print(f"  Temperature range: {T.min():.2f} - {T.max():.2f}°C")
    print(f"  Max displacement: {structural_fea.max_deformation:.3e} m")
    print(f"  Compliance: {J_compliance:.6e} J")
    
    print(f"\nSensitivity:")
    print(f"  Range: {dJdx.min():.3e} to {dJdx.max():.3e}")
    print(f"  Elements to add material: {np.sum(dJdx < 0)}")
    print(f"  Elements to remove material: {np.sum(dJdx > 0)}")
    
    if verify_fd:
        print(f"\nVerification:")
        print(f"  Max FD error: {max_error:.2e}")
        print(f"  Status: {'PASS' if max_error < 1e-3 else 'WARNING' if max_error < 1e-2 else 'FAIL'}")
    
    print("="*70 + "\n")
    
    return {
        'mesh': mesh,
        'x': x,
        'T': T,
        'd': d,
        'compliance': J_compliance,
        'sensitivity': dJdx,
        'thermal_fea': thermal_fea,
        'structural_fea': structural_fea,
        'sensitivity_analyzer': sensitivity
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("THERMO-ELASTIC SENSITIVITY ANALYSIS - COMPLETE EXAMPLE")
    print("Following user's coding style and BC setup")
    print("="*70)
    
    # Problem parameters
    nDOFDesired = 10000
    hot_temp = 100.0  # °C
    cold_temp = 20.0  # °C
    tensile_force = 5000.0  # N
    
    # SIMP penalties
    p_simp = 3.0  # Structural
    q_simp = 1.0  # Thermal
    
    # Run analysis
    try:
        results = run_coupled_analysis(
            nDOFDesired=nDOFDesired,
            hot_temp=hot_temp,
            cold_temp=cold_temp,
            tensile_force=tensile_force,
            p_simp=p_simp,
            q_simp=q_simp,
            verify_fd=True
        )
        
        print("\n✓ Analysis completed successfully!")
        print("\nResults available in 'results' dictionary:")
        for key in results.keys():
            print(f"  - {key}")
        
        # Optional: Plot results if available
        try:
            print("\nGenerating visualization...")
            results['thermal_fea'].plot_temperature()
            results['structural_fea'].plot_deformation()
        except Exception as e:
            print(f"Visualization skipped: {e}")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()