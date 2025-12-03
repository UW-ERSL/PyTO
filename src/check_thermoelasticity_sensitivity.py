"""
Complete Thermo-Elastic Sensitivity Example
"""
try:
    import numpy as np
    import os
    import sys
    import scipy.sparse as sp
    import linear_solvers
    from topopt_material_model import MaterialModel
    import bound_cond
    from hex_thermal_fea import HexThermalFEA
    from hex_structural_fea import HexStructuralFEA
    import hex_mesher
    import linear_solvers
    import bound_cond
    import mat_lib
    from topopt_material_model import MaterialModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ThermoElasticSensitivity:
    """Computes sensitivities for coupled thermo-elastic problems."""
    
    def __init__(self, 
                 thermal_fea,  # HexThermalFEA instance
                 structural_fea):  # HexStructuralFEA instance
        """
        Initialize the sensitivity analyzer.
        
        Parameters:
        -----------
        thermal_fea : HexThermalFEA
            Thermal finite element analysis solver
        structural_fea : HexStructuralFEA
            Structural finite element analysis solver
        """
        self.thermal_fea = thermal_fea
        self.structural_fea = structural_fea
        self.thermalMesh = thermal_fea.mesh
        self.structuralMesh = structural_fea.mesh
        if (structural_fea.mesh.num_elems != thermal_fea.mesh.num_elems):
            raise ValueError("Structural and thermal meshes must have the same number of elements.")
        # Cache element stiffness matrices (solid material)
        self.ke_bar_structural = structural_fea.elem_stiff[0]  # 24x24
        self.kt_bar_thermal = thermal_fea.elem_stiff[0]  # 8x8
        
        # Get material properties
        self.mat_prop = structural_fea.mat_prop
        if isinstance(self.mat_prop, list):
            self.E0 = self.mat_prop[0].youngs_modulus
            self.nu = self.mat_prop[0].poissons_ratio
            self.alpha = self.mat_prop[0].thermal_expansion_coefficient
        else:
            self.E0 = self.mat_prop.youngs_modulus
            self.nu = self.mat_prop.poissons_ratio
            self.alpha = self.mat_prop.thermal_expansion_coefficient
        
        # Get reference temperature from thermal FEA
        self.T_ref = thermal_fea.thermoElasticReferenceTemperature
        # Compute H matrix (24x8) for thermal forces
        dx, dy, dz = self.thermalMesh.elem_size
        self.H = thermal_fea.getHMatrix( dx, dy, dz, self.nu)

    def compute_compliance_sensitivity(self,
                                      x,
                                      T,
                                      d,
                                      p=3.0,
                                      q=1.0,
                                      material_model=MaterialModel.SIMP,
                                      solver=linear_solvers.Solvers.PARDISO,
                                      verbose=False):
        """
        Compute compliance sensitivity: dJ_S / dx_e
        
        Uses J =  d^T K d (strain energy definition).
        
        The sensitivity includes three terms:
        1. Structural stiffness: -p * xi^(p-1) * (1/2) * d_e^T * ke_bar * d_e
        2. Thermal force: p * xi^(p-1) * E0 * alpha * d_e^T * H * (T_e - T_ref)
        3. Thermal adjoint: q * xi^(q-1) * lambda_T_e^T * kt_bar * T_e
        
        Parameters:
        -----------
        x : ndarray (num_elems,)
            Design variables (pseudo-densities)
        T : ndarray (num_nodes,)
            Temperature field
        d : ndarray (num_dofs_structural,)
            Displacement field
        p : float
            Structural SIMP penalty (default: 3.0)
        q : float
            Thermal SIMP penalty (default: 1.0)
        material_model : MaterialModel
            Material interpolation model
        solver : linear_solvers.Solvers
            Linear solver for adjoint system
        verbose : bool
            Print detailed information
            
        Returns:
        --------
        dJdx : ndarray (num_elems,)
            Compliance sensitivity with respect to design variables
        """
        nelem = self.thermalMesh.num_elems
        dJdx = np.zeros(nelem)
        term1 = np.zeros(nelem)
        term2 = np.zeros(nelem) 
        term3 = np.zeros(nelem)
        
        # Step 1: Solve thermal adjoint equation
        # K_T^T * lambda_T = -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        lambda_T = self.solve_thermal_adjoint(d, x, p, solver, verbose)
        
        print(np.max(np.abs(self.kt_bar_thermal)), np.max(np.abs(lambda_T)))
        # Step 2: Compute element-wise sensitivities
        for e in range(nelem):
            # Get element DOFs
            edof_s = self.structuralMesh.edofMatStructural[e, :]
            edof_t = self.thermalMesh.edofMatThermal[e, :]
            d_e = d[edof_s]  # Total displacement
            T_e = T[edof_t]
            lambda_T_e = lambda_T[edof_t]
            
            # Term 1: Direct structural stiffness contribution 
            term1[e] = - p * x[e]**(p - 1) * d_e.T @ self.ke_bar_structural @ d_e
            
            # Term 2: Direct thermal force contribution
            T_diff = T_e - self.T_ref
            term2[e] = 2* p * x[e]**(p - 1) * self.E0 * self.alpha * d_e.T @ self.H @ T_diff

            # Term 3: Adjoint thermal contribution
            #term3[e] = q * x[e]**(q - 1) * lambda_T_e.T @ self.kt_bar_thermal @ T_e
            term3[e]  = q * x[e]**(q - 1)  * lambda_T_e.T @ self.kt_bar_thermal @ T_e

            dJdx[e] = term1[e] + term2[e] + term3[e]

        print(f"Max sensitivity terms: Term1={np.max(np.abs(term1)):.4e}, Term2={np.max(np.abs(term2)):.4e}, Term3={np.max(np.abs(term3)):.4e}")
        return dJdx
    
    def solve_thermal_adjoint(self,
                             d,
                             x,
                             p=3.0,
                             solver=linear_solvers.Solvers.PARDISO,
                             verbose=False):
        """
        Solve the thermal adjoint equation:
        K_T^T * lambda_T = -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        
        Since K_T is symmetric, this reduces to:
        K_T * lambda_T = -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        
        Parameters:
        -----------
        d : ndarray (num_dofs_structural,)
            Displacement field
        x : ndarray (num_elems,)
            Design variables
        p : float
            Structural SIMP penalty
        solver : linear_solvers.Solvers
            Linear solver to use
        verbose : bool
            Print information
            
        Returns:
        --------
        lambda_T : ndarray (num_nodes,)
            Thermal adjoint variable
        """
        nelem = self.thermalMesh.num_elems
        num_thermal_dofs = self.thermalMesh.num_nodes
        
        # Assemble RHS: -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        rhs = np.zeros(num_thermal_dofs)
        for e in range(nelem):
            edof_s = self.structuralMesh.edofMatStructural[e, :]
            edof_t = self.thermalMesh.edofMatThermal[e, :]
            d_e = d[edof_s]
            # Contribution from this element
            rhs_e = -2*x[e]**p * self.E0 * self.alpha * self.H.T @ d_e
            
            # Assemble into global RHS
            rhs[edof_t] += rhs_e
        
        
        # Get thermal stiffness matrix from thermal FEA
        # We need to assemble it with current design variables
        K_T = self.assemble_thermal_stiffness(x)
        bcAdjoint = bound_cond.BC(force = 0*self.thermal_fea.bc.force,fixed_dofs = self.thermal_fea.bc.fixed_dofs,
                                  dirichlet_values = 0.0*self.thermal_fea.bc.dirichlet_values) 
        # Solve adjoint system
        lambda_T = linear_solvers.solve(
            K_T,
            rhs,
            solver,
            bcAdjoint,
            **self.thermal_fea.kwargs
        )
        
        
        return lambda_T
    
    def assemble_structural_stiffness(self, x, material_model=MaterialModel.SIMP):
        """
        Assemble global structural stiffness matrix K_S(x).
        
        Parameters:
        -----------
        x : ndarray (num_elems,)
            Design variables
        material_model : MaterialModel
            Material interpolation model
            
        Returns:
        --------
        K_S : sp.coo_matrix
            Global structural stiffness matrix
        """
        from topopt_material_model import get_structural_material_model_scaling
        
        nelem = self.structuralMesh.num_elems
        
        # Get material scaling
        elem_material_scaling = get_structural_material_model_scaling(x, material_model)
        
        # Scale element stiffness matrices
        if self.structural_fea.elem_stiff.shape[0] == 1:
            # Single material case
            elem_stiff_mtrx = np.einsum('ij, e -> eij',
                                       self.structural_fea.elem_stiff[0],
                                       elem_material_scaling).flatten(order='C')
        else:
            # Multi-material case
            elem_stiff_mtrx = np.einsum('mij, m -> mij',
                                       self.structural_fea.elem_stiff,
                                       elem_material_scaling).flatten(order='C')
        
        # Assemble global matrix
        K_S = sp.coo_matrix(
            (elem_stiff_mtrx, 
             (self.structural_fea.node_idx[:, 0], self.structural_fea.node_idx[:, 1])),
            shape=(self.structural_fea.bc.num_dofs, self.structural_fea.bc.num_dofs)
        )
        
        return K_S
    
    def assemble_thermal_stiffness(self, x, material_model=MaterialModel.SIMP):
        """
        Assemble the global thermal stiffness matrix.
        
        Parameters:
        -----------
        x : ndarray (num_elems,)
            Design variables
        q : float
            Thermal SIMP penalty
        material_model : MaterialModel
            Material interpolation model
            
        Returns:
        --------
        K_T : sp.coo_matrix
            Global thermal stiffness matrix
        """
        from topopt_material_model import get_thermal_material_model_scaling
        
        # Get material scaling
        elem_material_scaling = get_thermal_material_model_scaling(x, material_model)
        
        # Scale element stiffness matrices
        if self.thermal_fea.elem_stiff.shape[0] == 1:
            # Single material case
            elem_stiff_mtrx = np.einsum('ij, e -> eij',
                                       self.thermal_fea.elem_stiff[0],
                                       elem_material_scaling).flatten(order='C')
        else:
            # Multi-material case
            elem_stiff_mtrx = np.einsum('mij, m -> mij',
                                       self.thermal_fea.elem_stiff,
                                       elem_material_scaling).flatten(order='C')
        
        # Assemble global matrix
        K_T = sp.coo_matrix(
            (elem_stiff_mtrx, 
             (self.thermal_fea.node_idx[:, 0], self.thermal_fea.node_idx[:, 1])),
            shape=(self.thermal_fea.bc.num_dofs, self.thermal_fea.bc.num_dofs)
        )
        
        return K_T
    
    def verify_sensitivity_fd(self,
                             x,
                             T,
                             d,
                             p=3.0,
                             q=1.0,
                             perturbation=1e-6,
                             element_indices=None,
                             material_model=MaterialModel.SIMP,
                             verbose=True):
        """
        Verify analytical sensitivity using finite differences.
        
        Parameters:
        -----------
        x : ndarray (num_elems,)
            Design variables
        T : ndarray (num_nodes,)
            Temperature field
        d : ndarray (num_dofs_structural,)
            Displacement field
        p : float
            Structural SIMP penalty
        q : float
            Thermal SIMP penalty
        perturbation : float
            Finite difference step size
        element_indices : list or None
            Elements to verify (if None, verify all)
        material_model : MaterialModel
            Material interpolation model
        verbose : bool
            Print detailed comparison
            
        Returns:
        --------
        relative_error : ndarray
            Relative error for each verified element
        analytical : ndarray
            Analytical sensitivities
        fd : ndarray
            Finite difference sensitivities
        """
 
        # Compute baseline compliance
        J0 = self.compute_compliance(d, x, material_model)
        
        # Compute analytical sensitivities
        dJdx_analytical = self.compute_compliance_sensitivity(
            x, T, d, p, q, material_model, verbose=False
        )
    
        
        # Select elements to verify
        if element_indices is None:
            # Verify a subset of elements for efficiency
            nelem = min(10, self.thermalMesh.num_elems)
            element_indices = np.linspace(0, self.thermalMesh.num_elems - 1, nelem, dtype=int)
        
        dJdx_fd = np.zeros(len(element_indices))
        relative_errors = np.zeros(len(element_indices))
        
        if verbose:
            print(f"\nVerifying {len(element_indices)} elements...")
            print(f"{'Element':<10} {'Analytical':<15} {'FD':<15} {'Rel Error':<15}")
            print("-" * 60)
        
        for idx, e in enumerate(element_indices):
            # Perturb design variable
            x_pert = x.copy()
            x_pert[e] += perturbation
            
            # Resolve thermal problem
            self.thermal_fea.solve(x_pert, material_model)
            
            # Compute new thermal forces
            f_th_pert = self.thermal_fea.get_thermoelastic_force(x_pert, material_model)
            
            # Update structural FEA with new thermal forces
            self.structural_fea.thermo_elastic_force = f_th_pert
            
            # Resolve structural problem
            d_pert = self.structural_fea.solve(x_pert, material_model)
            
            # Compute perturbed compliance with perturbed design
            J_pert = self.compute_compliance(d_pert, x_pert, material_model)
            
            # Finite difference approximation
            dJdx_fd[idx] = (J_pert - J0) / perturbation
            
            # Compute relative error
            if np.abs(dJdx_analytical[e]) > 1e-12:
                relative_errors[idx] = np.abs(dJdx_fd[idx] - dJdx_analytical[e]) / np.abs(dJdx_analytical[e])
            else:
                relative_errors[idx] = np.abs(dJdx_fd[idx] - dJdx_analytical[e])
            
            if verbose:
                print(f"{e:<10} {dJdx_analytical[e]:<15.6e} {dJdx_fd[idx]:<15.6e} {relative_errors[idx]:<15.6e}")
        
        
        return relative_errors, dJdx_analytical[element_indices], dJdx_fd
    
    def compute_compliance(self, d, x, material_model=MaterialModel.SIMP):
        """
        Compute structural compliance (strain energy): J_S = (1/2) d^T K_S(x) d
        
        Parameters:
        -----------
        d : ndarray (num_dofs_structural,)
            Displacement field
        x : ndarray (num_elems,)
            Design variables
        material_model : MaterialModel
            Material interpolation model
            
        Returns:
        --------
        J_S : float
            Structural compliance
        """
        # Assemble structural stiffness matrix with current design
        K_S = self.assemble_structural_stiffness(x, material_model)
        
        # Compute compliance 
        J_S =  d.T @ K_S @ d

        return J_S


# Import required modules


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

    # =========================================================================
    # THERMAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create thermal DOF mapping
    mesh.createEdofMatThermal()
   
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

    # =========================================================================
    # STRUCTURAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create structural DOF mapping
    mesh.createEdofMatStructural()
    # Fixed displacement at left end (all directions)
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    fixed_dofs = np.array([
        3 * fixed_nodes,
        3 * fixed_nodes + 1,
        3 * fixed_nodes + 2
    ]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)

    mesh.node_indices[fixed_nodes, 3] = 1# for plotting

    force = np.zeros(3 * mesh.num_nodes)
    load_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = xMax plane
    
    force[3 * load_nodes] = tensile_force/len(load_nodes)  # Distribute load evenly
    mesh.node_indices[load_nodes, 3] = 2 # for plotting

    bc_structural = bound_cond.BC(
        force=force,
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )
    mat_prop = mat_lib.get_material("Steel")
    # Reference temperature for thermal expansion
    reference_temp = cold_temperature  # Use cold temperature as reference
   
    return mesh, mat_prop, bc_thermal, bc_structural, reference_temp


def run_coupled_analysis(nDOFDesired: int = 10000,
                         hot_temp: float = 100.0,
                         cold_temp: float = 20.0,
                         tensile_force: float = 1000.0,
                         p_simp: float = 3.0,
                         q_simp: float = 1.0,
                         perturbation=1e-4,
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

    thermal_fea = HexThermalFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_thermal,
        solver=solver,
        thermoElasticReferenceTemperature=reference_temp
    )
    
    # Initialize design variables (uniform density)
    x = 0.1*np.ones(mesh.num_elems) 

    # Solve thermal problem
    T = thermal_fea.solve(x, material_model=MaterialModel.SIMP)
 
    # =========================================================================
    # COMPUTE THERMAL FORCES
    # =========================================================================
    f_thermal = thermal_fea.get_thermoelastic_force(x, material_model=MaterialModel.SIMP)

    # =========================================================================
    # STRUCTURAL FEA
    # =========================================================================
    
    # Add thermal forces to structural FEA
    structural_fea = HexStructuralFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_structural,
        solver=solver,
        thermo_elastic_force=f_thermal
    )
    # Solve structural problem
    d = structural_fea.solve(x, material_model=MaterialModel.SIMP)

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    
    # Initialize sensitivity analyzer
    sensitivity = ThermoElasticSensitivity(thermal_fea, structural_fea)
    
    # Compute analytical sensitivities
    print("\nComputing analytical sensitivities using adjoint method...")


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

    # =========================================================================
    # FINITE DIFFERENCE VERIFICATION
    # =========================================================================
    
    if verify_fd:
        print("\nComputing finite difference sensitivities and comparing with analytical results...")
        # Select elements to verify
        num_verify = min(10, mesh.num_elems)
        element_indices = np.linspace(0, mesh.num_elems - 1, num_verify, dtype=int)
            
        sensitivity.verify_sensitivity_fd(
            x=x,
            T=T,
            d=d,
            p=p_simp,
            q=q_simp,
            perturbation=perturbation,
            element_indices=element_indices,
            material_model=MaterialModel.SIMP,
            verbose=True
        )
       

if __name__ == "__main__":

    # Problem parameters
    nDOFDesired = 10000
    hot_temp = 30.0  # °C
    cold_temp = 23.0  # °C
    tensile_force = 0.0  # N
    
    perturbation = 1e-5  # Finite difference perturbation size

    run_coupled_analysis(
            nDOFDesired=nDOFDesired,
            hot_temp=hot_temp,
            cold_temp=cold_temp,
            tensile_force=tensile_force,
            perturbation=perturbation,
            verify_fd=True
        )
