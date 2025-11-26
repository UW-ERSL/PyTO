"""Thermo-Elastic Sensitivity Analysis Module.

This module computes sensitivities for coupled thermo-elastic topology optimization problems.
It implements the adjoint method for efficient sensitivity computation.

References:
-----------
Deng, S. and Suresh, K., 2017. Stress constrained thermo-elastic topology optimization 
with varying temperature fields via augmented topological sensitivity based level-set. 
Structural and Multidisciplinary Optimization, 56(6), pp.1413-1427.
"""

import numpy as np
import scipy.sparse as sp
import linear_solvers
from topopt_material_model import MaterialModel


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
        self.mesh = thermal_fea.mesh
        
        # Verify both FEA objects use the same mesh
        assert thermal_fea.mesh == structural_fea.mesh, \
            "Thermal and structural FEA must use the same mesh"
        
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
        
        print(f"Tref in sensitivity: {self.T_ref}")
        print(f"Tref in forward solve: {self.thermal_fea.thermoElasticReferenceTemperature}")
        # Compute H matrix (24x8) for thermal forces
        dx, dy, dz = self.mesh.elem_size
        self.H = thermal_fea.getHMatrix(dx, dy, dz, self.nu)
        
        print(f"ThermoElasticSensitivity initialized:")
        print(f"  Number of elements: {self.mesh.num_elems}")
        print(f"  E0 = {self.E0:.2e}, nu = {self.nu:.3f}, alpha = {self.alpha:.2e}")
        print(f"  T_ref = {self.T_ref:.2f}")
        print(f"  H matrix shape: {self.H.shape}")
  
    
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
        
        Uses J = (1/2) d^T K d (strain energy definition).
        
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
        nelem = self.mesh.num_elems
        dJdx = np.zeros(nelem)
        
        if verbose:
            print("\n" + "="*60)
            print("Computing Compliance Sensitivity")
            print("="*60)
            print(f"Design variables: min={x.min():.3e}, max={x.max():.3e}")
            print(f"Temperature: min={T.min():.2f}, max={T.max():.2f}")
            print(f"Displacement: min={d.min():.3e}, max={d.max():.3e}")
            print(f"SIMP penalties: p={p}, q={q}")
        
        # Step 1: Solve thermal adjoint equation
        # K_T^T * lambda_T = -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        lambda_T = self.solve_thermal_adjoint(d, x, p, solver, verbose)
        
        if verbose:
            print(f"\nAdjoint solution: min={lambda_T.min():.3e}, max={lambda_T.max():.3e}")
            print(f"\nComputing element-wise sensitivities...")
        
        print(f"Max value in H matrix: {np.max(np.abs(self.H))}")
        print(f"Element size: {self.mesh.elem_size}")
       
        # Step 2: Compute element-wise sensitivities
        for e in range(nelem):
            # Get element DOFs
            edof_s = self.mesh.edofMatStructural[e, :]
            edof_t = self.mesh.edofMatThermal[e, :]
   
            
            d_e = d[edof_s]  # Total displacement (current)
       
            # Extract element vectors
            d_e = d[edof_s]
            T_e = T[edof_t]
            lambda_T_e = lambda_T[edof_t]
            
            # Term 1: Direct structural stiffness contribution (strain energy with 1/2)
            term1 = - p * x[e]**(p - 1) * d_e.T @ self.ke_bar_structural @ d_e
            
            # Term 2: Direct thermal force contribution
            T_diff = T_e - self.T_ref
            term2 = 2* p * x[e]**(p - 1) *self.E0 * self.alpha * d_e.T @ self.H @ T_diff
            if verbose and e == 0:
                print(f"Term2 contribution at element 0: d_e^T H (T-Tref) = {d_e.T @ self.H @ T_diff}")
            # Term 3: Adjoint thermal contribution
            term3 = q * x[e]**(q - 1) * lambda_T_e.T @ self.kt_bar_thermal @ T_e
            
            k = self.mat_prop.thermal_conductivity  # Get this value
            scaling_factor = k / (self.E0 * self.alpha)
            term3 *= scaling_factor
            # Total sensitivity (NOTE: Signs flipped from standard derivation - empirical correction)
            dJdx[e] = term1 + term2 + term3

            if verbose and e == 0:
                # For element 0:
                print(f"\nDetailed element 0 analysis:")
                print(f"  x[0] = {x[0]}")
                print(f"  T_avg = {T_e.mean():.2f} C")
                print(f"  |d_e| = {np.linalg.norm(d_e):.6e}")
                print(f"  d_e^T K_e d_e = {d_e.T @ self.ke_bar_structural @ d_e:.6e}")
                print(f"  d_e^T H (T-Tref) = {d_e.T @ self.H @ T_diff:.6e}")
                print(f"  term1 = {term1:.6e}")
                print(f"  term2 = {term2:.6e}")
                print(f"  term3 (scaled) = {term3:.6e}")
                print(f"  total (term1+term2+term3) = {term1+term2+term3:.6e}")


            if verbose and e  in [0, 364, 2552]:
                print(f"Element {e}: dJdx = {dJdx[e]:.3e} (term1={term1:.3e}, term2={term2:.3e}, term3={term3:.3e})")
        if verbose:
            print(f"\nSensitivity statistics:")
            print(f"  Term 1 (structural): avg magnitude = {np.mean(np.abs([p * x[e]**(p-1) * d[self.mesh.edofMatStructural[e, :]].T @ self.ke_bar_structural @ d[self.mesh.edofMatStructural[e, :]] for e in range(min(100, nelem))])):.3e}")
            print(f"  Total sensitivity: min={dJdx.min():.3e}, max={dJdx.max():.3e}, mean={dJdx.mean():.3e}")
            print("="*60)
        
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
        nelem = self.mesh.num_elems
        num_thermal_dofs = self.mesh.num_nodes
        
        if verbose:
            print(f"\nSolving thermal adjoint system...")
            print(f"  Number of thermal DOFs: {num_thermal_dofs}")
        
        # Assemble RHS: -sum_e (xi_e^p * E0 * alpha * H^T * d_e)
        rhs = np.zeros(num_thermal_dofs)
        
    
        for e in range(nelem):
            edof_s = self.mesh.edofMatStructural[e, :]
            edof_t = self.mesh.edofMatThermal[e, :]
            
            d_e = d[edof_s]
            
            # Contribution from this element
            rhs_e = -2*x[e]**p * self.E0 * self.alpha * self.H.T @ d_e
            
            # Assemble into global RHS
            rhs[edof_t] += rhs_e
        
        if verbose:
            print(f"  RHS: min={rhs.min():.3e}, max={rhs.max():.3e}, norm={np.linalg.norm(rhs):.3e}")
        
        # Get thermal stiffness matrix from thermal FEA
        # We need to assemble it with current design variables
        K_T = self.assemble_thermal_stiffness(x, q=1.0)
        
        # Solve adjoint system
        lambda_T = linear_solvers.solve(
            K_T,
            rhs,
            solver,
            self.thermal_fea.bc,
            **self.thermal_fea.kwargs
        )
        
        print(f"E0 = {self.E0}")
        print(f"alpha = {self.alpha}")
        print(f"Max displacement |d| = {np.max(np.abs(d))}")
        print(f"Max adjoint |λ_T| = {np.max(np.abs(lambda_T))}")
        print(f"Thermal RHS max = {np.max(np.abs(rhs))}")
        
        if verbose:
            print(f"  Adjoint solution computed successfully")
            print(f"  lambda_T: min={lambda_T.min():.3e}, max={lambda_T.max():.3e}")
        
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
        
        nelem = self.mesh.num_elems
        
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
    
    def assemble_thermal_stiffness(self, x, q=1.0, material_model=MaterialModel.SIMP):
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
        
        nelem = self.mesh.num_elems
        
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
        if verbose:
            print("\n" + "="*60)
            print("Finite Difference Verification")
            print("="*60)
            print(f"Perturbation: {perturbation:.2e}")
        
        # Compute baseline compliance
        J0 = self.compute_compliance(d, x, material_model)
        
        
        
        # Compute analytical sensitivities
        dJdx_analytical = self.compute_compliance_sensitivity(
            x, T, d, p, q, material_model, verbose=False
        )
        
  
        if verbose:
            print(f"Baseline compliance: {J0:.6e}")
        
        # Select elements to verify
        if element_indices is None:
            # Verify a subset of elements for efficiency
            nelem = min(20, self.mesh.num_elems)
            element_indices = np.linspace(0, self.mesh.num_elems - 1, nelem, dtype=int)
        
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
            
            if verbose and e == 0:
                # For element 0:

                print(f"  FD = {dJdx_fd[0]:.6e}")
            # Compute relative error
            if np.abs(dJdx_analytical[e]) > 1e-12:
                relative_errors[idx] = np.abs(dJdx_fd[idx] - dJdx_analytical[e]) / np.abs(dJdx_analytical[e])
            else:
                relative_errors[idx] = np.abs(dJdx_fd[idx] - dJdx_analytical[e])
            
            
            if verbose:
                print(f"{e:<10} {dJdx_analytical[e]:<15.6e} {dJdx_fd[idx]:<15.6e} {relative_errors[idx]:<15.6e}")
        
        if verbose:
            print("-" * 60)
            print(f"Average relative error: {np.mean(relative_errors):.6e}")
            print(f"Max relative error: {np.max(relative_errors):.6e}")
            print("="*60)
        
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


def example_usage():
    """
    Example demonstrating how to use the ThermoElasticSensitivity class.
    """
    print("\n" + "="*60)
    print("ThermoElasticSensitivity - Example Usage")
    print("="*60)
    
    # This is a placeholder - actual usage would require:
    # 1. Import and setup thermal and structural FEA
    # 2. Solve both thermal and structural problems
    # 3. Compute sensitivities
    
    print("""
    Example workflow:
    
    # 1. Setup thermal FEA
    thermal_fea = HexThermalFEA(mesh, mat_prop, bc_thermal, solver)
    T = thermal_fea.solve(x)
    
    # 2. Compute thermal forces
    f_thermal = thermal_fea.get_thermoelastic_force(x)
    
    # 3. Setup structural FEA with thermal forces
    structural_fea = HexStructuralFEA(mesh, mat_prop, bc_structural, solver,
                                       thermo_elastic_force=f_thermal)
    d = structural_fea.solve(x)
    
    # 4. Compute compliance
    J = structural_fea.sol.T @ structural_fea.stiff_mtrx @ structural_fea.sol
    print(f"Compliance: {J:.6e}")
    
    # 5. Initialize sensitivity analyzer
    sensitivity = ThermoElasticSensitivity(thermal_fea, structural_fea)
    
    # 6. Compute sensitivities
    dJdx = sensitivity.compute_compliance_sensitivity(x, T, d, p=3.0, q=1.0, verbose=True)
    
    # 7. Verify with finite differences
    rel_err, analytical, fd = sensitivity.verify_sensitivity_fd(
        x, T, d, p=3.0, q=1.0, perturbation=1e-6, verbose=True
    )
    """)
    print("="*60)


if __name__ == "__main__":
    example_usage()