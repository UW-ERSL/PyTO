import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import mat_lib
import bound_cond
import pyvista as pv  # pip install pyvista
from numba import njit
from scipy.sparse import csr_matrix



import deflation
import pypardiso

dsolver = deflation.DeflationSolver()

def gauss_pts_wts():
    a = 0.1381966011250105
    b = 1 - 3 * a
    gp = np.array([
    [a, a, a],
    [b, a, a],
    [a, b, a],
    [a, a, b]])
    w  = np.full(4, 1.0/24.0)
    return gp.T, w.T

def shape_function(shape: str, xi):
    """
    Shape functions N and their gradients ∇N in natural coordinates.
    Parameters
    ----------
    shape : {'Tet10', ...}
    xi    : 1-D array‐like of length 1, 2, or 3
    Returns
    -------
    N       : (nNodes,)   shape-function values
    grad_N  : (nDim, nNodes)    derivatives w.r.t (r,s,t)
    """
    xi = np.asarray(xi, dtype=float)
    dim = xi.size
    # 3-D Tet10 
    if shape == 'Tet10' and dim == 3:
        r, s, t = xi
        L4 = 1.0 - r - s - t
        L1, L2, L3 = r, s, t

        N = np.array([
            L1 * (2*L1 - 1),                 # N1
            L2 * (2*L2 - 1),                 # N2
            L3 * (2*L3 - 1),                 # N3
            L4 * (2*L4 - 1),                 # N4
            4 * L1 * L2,                     # N5
            4 * L2 * L3,                     # N6
            4 * L1 * L3,                     # N7
            4 * L1 * L4,                     # N8
            4 * L2 * L4,                     # N9
            4 * L3 * L4                      # N10
        ])

        # Partial derivatives w.r.t r, s, t
        
        dN_dr = np.array([
            4*L1 - 1,
            0,
            0,
            -4*L4 + 1,
            4*L2,
            0,
            4*L3,
            4*(L4 - L1),
            -4*L2,
            -4*L3
        ])

        dN_ds = np.array([
            0,
            4*L2 - 1,
            0,
            -4*L4 + 1,
            4*L1,
            4*L3,
            0,
            -4*L1,
            4*(L4 - L2),
            -4*L3
        ])

        dN_dt = np.array([
            0,
            0,
            4*L3 - 1,
            -4*L4 + 1,
            0,
            4*L2,
            4*L1,
            -4*L1,
            -4*L2,
            4*(L4 - L3)
        ])

        grad_N = np.vstack((dN_dr, dN_ds, dN_dt))
        return N, grad_N

    raise ValueError("Unsupported element type or bad xi size")


class StructuralFEATet:
    """Linear Structural Finite Element Analysis using 10-noded quadratic tet elements."""

    def __init__(
        self,
        quadratic_tet_mesh,
        mat_prop: mat_lib.StructuralMaterial,
        bc: bound_cond.BC,
        solver: lin_sol.Solvers,
        **kwargs,
    ):
        self.mesh = quadratic_tet_mesh
        self.mat_prop, self.bc = mat_prop, bc
        self.solver, self.kwargs = solver, kwargs

        self.shape = "Tet10"
        self.xi, self.wt = gauss_pts_wts()
        self.n, self.grad_n = self.get_shapefunction_gradients_at_quadpts()
        self.createEdofMatStructural()
        self.nodes_per_element = np.size(self.n[0])
        self.dof_per_node = 3
        self.dof_per_elem = self.dof_per_node * self.nodes_per_element

        self.material_model = 'GeneralizedNeoHookean1'
        E = self.mat_prop.youngs_modulus  
        nu = self.mat_prop.poissons_ratio
        self.bulkModulus =  E/(3*(1-2*nu)) 
        self.shearModulus = E/(2*(1+nu))

        total_dofs = self.mesh.num_nodes * self.dof_per_node
        self.neuman_force = np.zeros((total_dofs, 1))
        self.nodal_dirichlet = np.zeros((total_dofs, 1))
        self.F = np.zeros((total_dofs, 1))

    def createEdofMatStructural(self):
        """
        Create the element degree of freedom matrix for structural analysis.
        This function creates a matrix that maps each element to its corresponding degrees of freedom.
        The matrix is structured such that each node in an element contributes three degrees of freedom (x, y, z).
        Attributes:
        self.edofMat (numpy.ndarray): The element degree of freedom matrix.
        """
        self.edofMat = np.zeros((self.mesh.num_elems, 30), dtype=int)
        node_indices = np.arange(10)
        dof_indices = np.arange(3)
        for i in range(self.mesh.num_elems):
            # Use broadcasting to create all indices at once
            self.edofMat[i] = (
                3 * self.mesh.elems[i, node_indices, None] + dof_indices
            ).flatten()
    
    
    def get_shapefunction_gradients_at_quadpts(self):
        """
        Get all shape function gradients at quadrature points
        """
        # Precompute shape functions at quadrature points
        n_cell = []
        grad_n_cell = []
        for i in range(len(self.xi[0])):
            N, grad_N = shape_function(self.shape, self.xi[:,i])
            n_cell.append(N)
            grad_n_cell.append(grad_N)
        return n_cell, grad_n_cell


    def assemble_k(self):
        """
        Assemble global stiffness matrix and force vector
        """
        n_dof = self.dof_per_node * self.mesh.num_nodes
        n_elements = self.mesh.num_elems
        nzmax = self.dof_per_elem**2 * n_elements # Maximum number of non-zero entries
        
        row_triplets = np.zeros(nzmax)
        col_triplets = np.zeros(nzmax)
        entry_triplets = np.zeros(nzmax)
        f = np.zeros(n_dof)
        
        grad_n_cell = self.grad_n
        dof_per_elem = self.dof_per_elem
        wt = self.wt
        nodes_per_element = self.nodes_per_element
        material_model = self.material_model
        shearModulus = self.shearModulus
        bulkModulus = self.bulkModulus

        elemArray = self.mesh.elems
        node_xyz = self.mesh.nodes
        sol = self.sol
        

        compute_K_global(n_elements, elemArray, node_xyz, sol, grad_n_cell, dof_per_elem, wt, nodes_per_element, material_model, shearModulus, bulkModulus,
                                row_triplets, col_triplets, entry_triplets, f)
            
        self.K = csr_matrix((entry_triplets, (row_triplets, col_triplets)), 
                            shape=(n_dof, n_dof))
        self.F = f

    def plot_deformation(self):
        """Plot the deformed shape of the mesh."""
        # Create a PyVista plotter
        plotter = pv.Plotter()
        # Create a PyVista mesh from the tetrahedral mesh
        pv_mesh = pv.UnstructuredGrid(
            {pv.CellType.TETRA: self.mesh.elems[:, 0:4]}, self.mesh.nodes
        )

        sol = self.sol.copy()
        sol = sol.reshape((-1, 3))

        deltaMax = self.max_deformation
        L = np.max(
            [
                np.max(self.mesh.nodes[:, i]) - np.min(self.mesh.nodes[:, i])
                for i in range(3)
            ]
        )
        scale = float(0.2 * L / deltaMax)
        deformed_mesh = pv_mesh.copy()
        deformed_mesh.points += scale * sol[:, 0:4]
        # Add both original and deformed mesh to the plotter
        plotter.add_mesh(
            pv_mesh, show_edges=True, color="red", opacity=0.2, label="Original"
        )
        plotter.add_mesh(
            deformed_mesh,
            show_edges=True,
            color="lightblue",
            opacity=1,
            label="Deformed",
        )

        # Add legend
        plotter.add_legend()
        # Add axes widget
        plotter.add_axes()
        # Show the plot
        plotter.show()

    def solve_nonlinear_fem_force_control(self,verbose = True, n_steps = 5,max_iter = 30,tol = 1e-8):
        """
        Solve nonlinear finite element problem using Newton-Raphson method.
        """
        print('Force Control')
        success = False
        n_dof = self.dof_per_node*self.mesh.num_nodes
        self.sol = np.zeros(n_dof)
        #areaof_forceappliedface = self.mesh.bbox.ly*self.mesh.bbox.lz #integrate_over_surface_triangles
        self.neuman_force=self.bc.force
        neuman_force_on_face = self.neuman_force #self.neuman_force * areaof_forceappliedface
        for step in range(1, n_steps + 1):
            if verbose:
                print(f'Load Step {step}/{n_steps}')
            load_factor = step/n_steps
            iter = 1
            err = 1
            while (iter < max_iter) and (err > tol):
                self.assemble_k()  # Assembles K, M matrices and F vector
                all_dof = np.arange(n_dof)
                self.delta_sol = np.zeros(n_dof)
                self.free_dof = np.setdiff1d(all_dof, self.bc.fixed_dofs)
                b = self.F - (load_factor * neuman_force_on_face).flatten()
                fixed_dofs = self.bc.fixed_dofs
                free_dofs  = self.free_dof

                u_fixed = np.empty_like(fixed_dofs, dtype=b.dtype)
                for idx, i in enumerate(fixed_dofs):
                    u_fixed[idx] = -load_factor * self.nodal_dirichlet[i].item() + self.sol[i].item()

                K_global = self.K
                K_ff = K_global[free_dofs, :][:, free_dofs]
                K_fd = K_global[free_dofs, :][:, fixed_dofs]

                b_free = b[free_dofs] - K_fd.dot(u_fixed)
                if (self.solver == lin_sol.Solvers.DPCG):
                    M = lin_sol._jacobi_preconditioner(K_ff)
                    d_u_free = -dsolver.deflatedPCG(K_ff,
                                b_free,
                                W = dsolver.W,
                                M = M,
                                rtol = 1e-8)
                elif (self.solver == lin_sol.Solvers.PARDISO):
                    d_u_free = -pypardiso.spsolve(K_ff, b_free)
                else:
                    print('Solver not implemented')

                d_u = np.zeros(n_dof, dtype=b.dtype)
                d_u[free_dofs] = d_u_free

                self.delta_sol = d_u
                self.sol += self.delta_sol
             
                self.u = self.sol[::3]
                self.v = self.sol[1::3]
                self.w = self.sol[2::3]
                
                err = np.linalg.norm(self.delta_sol)/np.linalg.norm(self.sol)
                
                if verbose:
                    print(f'\tIteration: {iter}\t {err:E}')
                iter += 1
        
        success = True if err < tol else False
        self.deformation = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.max_deformation = np.max(self.deformation)
        
        return  success

@njit(cache=True)
def compute_K_global(n_elements, elemArray, node_xyz, sol, grad_n, dof_per_elem, wt, nodes_per_element, material_model, shearModulus, bulkModulus,
                    row_triplets, col_triplets, entry_triplets, f):
    index = 0
    for elem in range(n_elements):
        elem_nodes = elemArray[elem]
        position_nodes = node_xyz[elem_nodes, :]
        sol_elem = np.vstack((sol[3*elem_nodes], sol[3*elem_nodes+1], sol[3*elem_nodes+2]))

        k_elem, f_elem = compute_element_stiffness_finite_strain_spatial_conf(grad_n, dof_per_elem, wt, nodes_per_element, sol_elem, position_nodes, material_model, shearModulus, bulkModulus)
        
        if (k_elem is None):
            print(f'Element {elem}: det(F) < 0')
            continue
        dof = np.vstack((3*elem_nodes, 3*elem_nodes + 1, 3*elem_nodes + 2))
        #dof = dof.reshape(-1, order='F')
        dof = dof.T.flatten()

        # Create temp matrix by replicating dof myDOFPerElem times
        temp = np.empty((dof_per_elem, dof.size), dtype=dof.dtype)
        for i in range(dof_per_elem):
            temp[i, :] = dof
        # Transpose temp to prepare for Fortran-like flattening
        temp_T = temp.T

        # Flatten the transposed array
        row_index = temp_T.flatten()

        # Since temp_T is the transpose, its flattening gives the desired column-major order
        col_index = temp.flatten()

        # For k_elem, transpose before flattening to achieve Fortran-like order
        entries = k_elem.T.flatten()

        row_triplets[index:index+dof_per_elem**2] = row_index
        col_triplets[index:index+dof_per_elem**2] = col_index
        entry_triplets[index:index+dof_per_elem**2] = entries
        
        index += dof_per_elem**2
        f[dof] += f_elem
            
@njit(cache=True)
def compute_element_stiffness_finite_strain_spatial_conf(grad_n, dof_per_elem, wt, nodes_per_element, sol_elem, position_nodes, material_model, shearModulus, bulkModulus):
    """
    Computes elemental stiffness matrix and force vector for given element
    Args:
        elem: Element number
    Returns:
        tuple: (K_elem, f_elem) Elemental stiffness matrix and force vector
    """
    grad_n_cell = grad_n  # Shape function gradient values at quadrature points
    k_material = np.zeros((dof_per_elem, dof_per_elem))
    k_geometric = np.zeros((dof_per_elem, dof_per_elem))
    f_elem = np.zeros(dof_per_elem)
    #xi_gq = xi  # Quadrature points
    wt_gq = wt  # Quadrature weights
    num_gq = len(wt_gq)
    
    nodes = nodes_per_element
 
    k_elem = compute_k_elem(num_gq, grad_n_cell, wt_gq, position_nodes, sol_elem, nodes, k_material, k_geometric, f_elem, material_model, shearModulus, bulkModulus)
    if (k_elem is None):
        return None, None
    return k_elem, f_elem


@njit(cache=True)
def compute_k_elem(num_gq, grad_n_cell, wt_gq, position_nodes, sol, nodes, k_material, k_geometric, f_elem, material_model, shearModulus, bulkModulus):
    for g in range(num_gq):
        grad_n_all = grad_n_cell[g]
        j_total = grad_n_all @ position_nodes
        grad_ndx = np.linalg.solve(j_total, grad_n_all)
        F = np.eye(3) + sol@ grad_ndx.T
        b = F @ F.T  # Left green deformation tensor
        F_inv = np.linalg.inv(F)
        
        grad_ndxs = np.zeros_like(grad_ndx)
        for k in range(nodes):
            for i in range(3):
                grad_ndxs[i,k] = np.sum(grad_ndx[:,k] * F_inv[:,i])

        J_F = np.linalg.det(F)
        if J_F < 0:
            #print('Determinant of elem F negative')
            return None
            
        stress = kirchhoff_stress(material_model, shearModulus, bulkModulus, b, J_F)

        C = compute_elasticity_tensor_generalized_neo_hookean(material_model, shearModulus, bulkModulus, b, J_F)
        dJ = abs(np.linalg.det(j_total))

        compute_K_material_geometric(nodes, g, dJ, grad_ndxs, wt_gq, stress, C, k_material, k_geometric, f_elem)

    k_elem = k_material + k_geometric
    return k_elem


@njit(cache=True)
def kirchhoff_stress(material_model, shearModulus, bulkModulus, B, J):
    """     Calculate Kirchhoff stress tensor for hyperelastic materials.
    This function implements two variations of the Generalized Neo-Hookean material model:
    GeneralizedNeoHookean1:
        Uses a volumetric energy term k*J*(J-1)^2/2
        Suitable for moderate compressibility
        More commonly used in literature
    GeneralizedNeoHookean2:
        Uses a volumetric energy term k*(J^2-1-2*ln(J))/4 
        Better suited for highly compressible materials
        More numerically stable at large deformations
    Parameters
    ----------
    B : ndarray, shape (3,3)
        Left Cauchy-Green deformation tensor
    J : float
        Determinant of the deformation gradient
    Returns
    -------
    stress : ndarray, shape (3,3)
        Kirchhoff stress tensor """
    stress = np.zeros((3, 3))
    delta = np.eye(3)
    Bkk = np.trace(B)
    
    if material_model == 'GeneralizedNeoHookean1':
        kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress)
                
    elif material_model == 'GeneralizedNeoHookean2':
        kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress)
                
    return stress

@njit(cache=True)
def kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress):
    for i in range(3):
            for j in range(3):
                stress[i,j] = shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + bulkModulus * J * (J-1) * delta[i,j]

@njit(cache=True)
def kirchoff_stress_generalized_neo_hookean2(B, J, Bkk, delta, shearModulus, bulkModulus, stress):
    for i in range(3):
            for j in range(3):
                stress[i,j] = shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + 0.5 * bulkModulus * J * (J-1/J) * delta[i,j]
                
                
@njit(cache=True)
def compute_elasticity_tensor_generalized_neo_hookean(material_model, shearModulus, bulkModulus, B, J):
    """
    3D Elasticity tensor for hyperelastic material modeled by Generalized neohookean model
    """
    delta = np.eye(3)
    Bqq = np.trace(B)
    
    C = np.zeros((3, 3, 3, 3))
    
    if material_model == 'GeneralizedNeoHookean1':
        elasticity_tensor_generalised_neo_hookean_1(shearModulus, bulkModulus, Bqq, delta, B, J, C)
                        
    elif material_model == 'GeneralizedNeoHookean2':
        elasticity_tensor_generalised_neo_hookean_2(shearModulus, bulkModulus, Bqq, delta, B, J, C)

    
    return C

@njit(cache=True)
def elasticity_tensor_generalised_neo_hookean_1(shearModulus, bulkModulus, Bqq, delta, B, J, C):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k] 
                                    - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                    + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                + bulkModulus * (2*J - 1) * J * delta[i,j] * delta[k,l]
               
@njit(cache=True)
def elasticity_tensor_generalised_neo_hookean_2(shearModulus, bulkModulus, Bqq, delta, B, J, C):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k]
                                    - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                    + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                + bulkModulus * J * J * delta[i,j] * delta[k,l]
  
@njit(cache=True)
def compute_K_material_geometric(nodes, g, dJ, grad_ndxs, wt_gq, stress, C, k_material, k_geometric, f_elem):
    """
    Compute material and geometric stiffness matrices for an element
    Args:
        elem (int): Element number
        xi_gq (ndarray): Quadrature points
        wt_gq (ndarray): Quadrature weights
    Returns:
        tuple: (K_material, K_geometric) Material and geometric stiffness matrices
    """
    for A in range(nodes):
            for i in range(3):
                for B in range(nodes):
                    for k in range(3):
                        for j in range(3):
                            for l in range(3):
                                k_material[3*A+i,3*B+k] += wt_gq[g]*dJ*grad_ndxs[j,A]*C[i,j,k,l]*grad_ndxs[l,B]
                            k_geometric[3*A+i,3*B+k] -= wt_gq[g]*dJ*grad_ndxs[k,A]*grad_ndxs[j,B]*stress[i,j]

                for J in range(3):
                    f_elem[3*A + i] += wt_gq[g]*dJ*stress[i,J]*grad_ndxs[J,A]


if __name__ == "__main__":
    import jax  # import jax to enable 64 bit precision
    import time
    from tet_structural_examples import TetStructuralExamples, getTetStructuralProblem

    jax.config.update("jax_enable_x64", True)

    problem = (
        TetStructuralExamples.TensileBar
    )  # CubeCompression, TensileBar, BeamBending
    nForceSteps = 2

    quadratic_tet_mesh, mat_prop, bc, elem_body_force = getTetStructuralProblem(
        problem, nDOFDesired=1000
    )

    solver = lin_sol.Solvers.PARDISO  # typically DPCG or PARDISO
    fe_solver = StructuralFEATet(
        quadratic_tet_mesh, mat_prop=mat_prop, bc=bc, solver=solver
    )

    startTime = time.time()
    fe_solver.solve_nonlinear_fem_force_control(n_steps=nForceSteps, verbose=True)

    delta = np.max(np.abs(fe_solver.deformation))
    

    # Store results
    nDOF = fe_solver.mesh.num_nodes

    print("-----------------------------")
    print("nDof: ", nDOF)
    print("Solver: ", fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print("Max deformation: ", delta)
    print("-----------------------------")
    fe_solver.plot_deformation()
