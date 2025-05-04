import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, norm
from scipy.special import roots_legendre
import scipy.io
import os
import time

class largeDeformationFEA:
    def __init__(self,mesh, mat_prop, bc):
        self.material_model = 'GeneralizedNeoHookean1'
        self.mesh = mesh
        self.material_properties = mat_prop
        self.boundary_conditions = bc
        E = self.material_properties.youngs_modulus  
        nu = self.material_properties.poissons_ratio
        self.bulkModulus =  E/(3*(1-2*nu)) 
        self.shearModulus = E/(2*(1+nu))
        self.neuman_force = np.zeros((self.mesh.num_nodes * 3, 1))
        self.nodal_dirichlet = np.zeros((self.mesh.num_nodes * 3, 1))
        self.F = np.zeros((self.mesh.num_nodes * 3, 1))



    def solve_nonlinear_fem(self,verbose = False, n_steps = 5,max_iter = 30,tol = 1e-9):
        """
        Solve nonlinear finite element problem using Newton-Raphson method.
        """
        success = False
        n_dof = 3*self.mesh.num_nodes
        self.sol = np.zeros(n_dof)
        areaof_forceappliedface = self.mesh.bbox.ly*self.mesh.bbox.lz
        self.neuman_force=self.boundary_conditions.force
        neuman_force_on_face = self.neuman_force*areaof_forceappliedface
        for step in range(1, n_steps + 1):
            if verbose:
                print(f'Load Step {step}/{n_steps}')
            load_factor = step/n_steps
            
            iter = 0
            err = 1
            while (iter < max_iter) and (err > tol):
                if verbose:
                    print(f'Load Step {step}/{n_steps} Iteration {iter}')
                iter += 1
                self.assemble_k()  # Assembles K, M matrices and F vector
                all_dof = np.arange(n_dof)
                self.delta_sol = np.zeros(n_dof)
                self.free_dof = np.setdiff1d(all_dof, self.boundary_conditions.fixed_dofs)
                b = self.F - (load_factor * neuman_force_on_face).flatten()
                
                # Fix constrained nodes
                self.K = self.K.tolil()
                for i in self.boundary_conditions.fixed_dofs:
                    self.K[i, :] = 0
                    self.K[i, i] = 1
                    b[i] = -load_factor * self.nodal_dirichlet[i].item() + self.sol[i].item()

                self.K = self.K.tocsr()
                d_sol = -spsolve(self.K, b)
                self.delta_sol = d_sol[:n_dof]
                self.sol += self.delta_sol
             
                self.u = self.sol[::3]
                self.v = self.sol[1::3]
                self.w = self.sol[2::3]
                
                err = np.linalg.norm(self.delta_sol)/np.linalg.norm(self.sol)
                
                #if verbose:
                    #print(f'{iter}\t {err:E}\t {np.linalg.norm(b)/n_coord/self.num_nodes:E}')
        
        success = True if err < tol else False
        self.deformation = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.max_delta = np.max(self.deformation)
        
        return  success

    def shape_function(self, xi):
        """
        Compute shape functions and gradients at parametric coordinates
        Args:
            xi: Parametric coordinates (1D, 2D or 3D)
        Returns:
            tuple: (N, grad_N) Shape functions and gradients
        """
        if np.size(xi) == 1:
            if self.shape == 'Linear':
                N = np.array([(1-xi)/2, (1+xi)/2])
                grad_N = np.array([-1/2, 1/2])
            elif self.shape == 'Quadratic':
                N = np.array([xi*(xi-1)/2, (1-xi)*(1+xi), xi*(xi+1)/2])
                grad_N = np.array([(2*xi-1)/2, -2*xi, (2*xi+1)/2])
        elif np.size(xi) == 2:
            N = 0.25 * np.array([
                (1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]),
                (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])
            ])
            grad_N = 0.25 * np.array([
                [xi[1]-1, 1-xi[1], xi[1]+1, -xi[1]-1],
                [xi[0]-1, -xi[0]-1, xi[0]+1, 1-xi[0]]
            ])
        elif np.size(xi) == 3:
            N = 0.125 * np.array([
                (1-xi[0])*(1-xi[1])*(1-xi[2]), (1+xi[0])*(1-xi[1])*(1-xi[2]),
                (1+xi[0])*(1+xi[1])*(1-xi[2]), (1-xi[0])*(1+xi[1])*(1-xi[2]),
                (1-xi[0])*(1-xi[1])*(1+xi[2]), (1+xi[0])*(1-xi[1])*(1+xi[2]),
                (1+xi[0])*(1+xi[1])*(1+xi[2]), (1-xi[0])*(1+xi[1])*(1+xi[2])
            ])
            grad_N = 0.125 * np.array([
                [(xi[1]-1)*(1-xi[2]), (1-xi[1])*(1-xi[2]), (xi[1]+1)*(1-xi[2]), (-xi[1]-1)*(1-xi[2]),
                    (xi[1]-1)*(1+xi[2]), (1-xi[1])*(1+xi[2]), (xi[1]+1)*(1+xi[2]), (-xi[1]-1)*(1+xi[2])],
                [(xi[0]-1)*(1-xi[2]), (-xi[0]-1)*(1-xi[2]), (xi[0]+1)*(1-xi[2]), (1-xi[0])*(1-xi[2]),
                    (xi[0]-1)*(1+xi[2]), (-xi[0]-1)*(1+xi[2]), (xi[0]+1)*(1+xi[2]), (1-xi[0])*(1+xi[2])],
                [-(1-xi[0])*(1-xi[1]), -(1+xi[0])*(1-xi[1]), -(1+xi[0])*(1+xi[1]), -(1-xi[0])*(1+xi[1]),
                    (1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]), (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])]
            ])
        return N, grad_N

    def jacobian(self, elem, xi):
        """
        Compute Jacobian matrix at parametric coordinates for an element
        Args:
            elem: Element number
            xi: Parametric coordinates
        Returns:
            ndarray: Jacobian matrix
        """
        nodes = self.mesh.elemArray[elem]
        position_nodes = self.mesh.node_xyz[nodes, :]
        _, grad_N = self.shape_function(xi)
        return grad_N @ position_nodes

    def assemble_k(self):
        """
        Assemble global stiffness matrix and force vector
        """
        self.dof_per_node = 3
        n_dof = self.dof_per_node * self.mesh.num_nodes
        
        self.nodes_per_element = 8
        self.dof_per_elem = self.dof_per_node * self.nodes_per_element
        n_elements = self.mesh.num_elems
        nzmax = self.dof_per_elem**2 * n_elements # Maximum number of non-zero entries
        
        row_triplets = np.zeros(nzmax)
        col_triplets = np.zeros(nzmax)
        entry_triplets = np.zeros(nzmax)
        f = np.zeros(n_dof)
        
        # Initialize quadrature points and weights
        nQuadPts1D = 2
        self.xi, self.wt = self.gauss_quad_3d(nQuadPts1D)
        
        # Precompute shape functions at quadrature points
        n_cell = []
        grad_n_cell = []
        for i in range(len(self.xi[0])):
            N, grad_N = self.shape_function(self.xi[:,i])
            n_cell.append(N)
            grad_n_cell.append(grad_N)
        
        self.n = n_cell
        self.grad_n = grad_n_cell
        
        index = 0
        for elem in range(n_elements):
            nodes = self.mesh.elemArray[elem]
            k_elem, f_elem = self.compute_element_stiffness_finite_strain_spatial_conf(elem)
            
            dof = np.vstack((3*nodes, 3*nodes + 1, 3*nodes + 2))
            dof = dof.reshape(-1, order='F')

            # Create temp matrix by replicating dof myDOFPerElem times
            temp = np.tile(dof, (self.dof_per_elem, 1))

            row_index = temp.reshape(1, self.dof_per_elem**2, order='F').flatten()
            col_index = temp.T.reshape(1, self.dof_per_elem**2, order='F').flatten()
            entries   = k_elem.T.reshape(1, self.dof_per_elem**2, order='F').flatten()
            
            row_triplets[index:index+self.dof_per_elem**2] = row_index
            col_triplets[index:index+self.dof_per_elem**2] = col_index
            entry_triplets[index:index+self.dof_per_elem**2] = entries
            
            index += self.dof_per_elem**2
            f[dof] += f_elem
            
        self.K = csr_matrix((entry_triplets, (row_triplets, col_triplets)), 
                            shape=(n_dof, n_dof))
        self.F = f

    def compute_element_stiffness_finite_strain_spatial_conf(self, elem):
        """
        Computes elemental stiffness matrix and force vector for given element
        Args:
            elem: Element number
        Returns:
            tuple: (K_elem, f_elem) Elemental stiffness matrix and force vector
        """
        grad_n_cell = self.grad_n  # Shape function gradient values at quadrature points
        k_material = np.zeros((self.dof_per_elem, self.dof_per_elem))
        k_geometric = np.zeros((self.dof_per_elem, self.dof_per_elem))
        f_elem = np.zeros(self.dof_per_elem)
        xi_gq = self.xi  # Quadrature points
        wt_gq = self.wt  # Quadrature weights
        num_gq = len(wt_gq)
        
        nodes = self.nodes_per_element
        elem_nodes = self.mesh.elemArray[elem,:]
        sol = np.vstack((self.sol[3*elem_nodes], self.sol[3*elem_nodes+1], self.sol[3*elem_nodes+2]))

        for g in range(num_gq):
            grad_n_all = grad_n_cell[g]
            j_total = self.jacobian(elem, xi_gq[:,g])
            grad_ndx = np.linalg.solve(j_total, grad_n_all)
            F = np.eye(3) + sol@ grad_ndx.T
            b = F @ F.T  # Left green deformation tensor
            F_inv = np.linalg.inv(F)
            
            grad_ndxs = np.zeros_like(grad_ndx)
            for k in range(nodes):
                for i in range(3):
                    grad_ndxs[i,k] = np.sum(grad_ndx[:,k] * F_inv[:,i])

            J_F = np.linalg.det(F)
            stress = self.kirchhoff_stress( b, J_F)
            C = self.compute_elasticity_tensor_generalized_neo_hookean(b, J_F)
            dJ = abs(np.linalg.det(j_total))

            if J_F < 0:
                print('Determinant of F negative')
                break

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

        k_elem = k_material + k_geometric
        return k_elem, f_elem
    
    def compute_strain_energy(self):
        """
        Compute strain energy
        """
        self.strain_energy = 0.5 * self.sol @ (self.K[:3*self.mesh.num_nodes, :3*self.mesh.num_nodes] @ self.sol)
        return self
    
    def gauss_quad_2d_quad(self, num_gq=4):
        """
        Gauss quadrature points for 2D quadrilateral elements using tensor product.
        Args:
            num_gq (int): Number of quadrature points per direction.
        Returns:
            tuple: (xi_GQ, wt_GQ) 2D Parametric coordinates and weights.
        """
        N = int(np.sqrt(num_gq))
        x, w = roots_legendre(N)
        
        # Reverse the nodes and weights to get positives first
        x = x[::-1]
        w = w[::-1]
        
        xi_GQ = np.zeros((2, N * N))
        xi_GQ[0, :] = np.repeat(x, N)
        xi_GQ[1, :] = np.tile(x, N)
        
        wt_GQ = np.outer(w, w).flatten()
        
        return xi_GQ, wt_GQ
    
    def gauss_quad_3d(self, num_gq=1):
        """
        Gauss quadrature points for 3D elements using tensor product
        Args:
            num_gq (int): Number of quadrature points per direction
        Returns:
            tuple: (xi_GQ, wt_GQ) 3D Parametric coordinates and weights
        """
        # Get 1D Gauss points and weights
        x, w = self.lgwt(num_gq, -1, 1)
        # Reverse the nodes and weights to get positives first
        x = x[::-1]
        w = w[::-1]
        # X, Y, Z are each of shape (N, N, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        # Flatten the grids and stack them into a (3, N^3) array.
        xi_gq = np.vstack((X.flatten(order='F'),
                    Y.flatten(order='F'),
                    Z.flatten(order='F')))
        # Form the 3D quadrature weights as the tensor product of the 1D weights.
        wt_gq = np.kron(np.kron(w, w), w)
        return xi_gq, wt_gq


    def kirchhoff_stress(self,  B, J):
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
        
        if self.material_model == 'GeneralizedNeoHookean1':
            for i in range(3):
                for j in range(3):
                    stress[i,j] = self.shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + self.bulkModulus * J * (J-1) * delta[i,j]
                    
        elif self.material_model == 'GeneralizedNeoHookean2':
            for i in range(3):
                for j in range(3):
                    stress[i,j] = self.shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + 0.5 * self.bulkModulus * J * (J-1/J) * delta[i,j]
                    
        return stress

    def compute_elasticity_tensor_generalized_neo_hookean(self, B, J):
        """
        3D Elasticity tensor for hyperelastic material modeled by Generalized neohookean model
        """
        delta = np.eye(3)
        Bqq = np.trace(B)
        
        C = np.zeros((3, 3, 3, 3))
        
        if self.material_model == 'GeneralizedNeoHookean1':
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            C[i,j,k,l] = self.shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k] 
                                            - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                            + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                        + self.bulkModulus * (2*J - 1) * J * delta[i,j] * delta[k,l]
                            
        elif self.material_model == 'GeneralizedNeoHookean2':
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            C[i,j,k,l] = self.shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k]
                                            - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                            + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                        + self.bulkModulus * J * J * delta[i,j] * delta[k,l]
        
        return C
    def lgwt(self,N, a, b):
        """
        Compute Legendre-Gauss nodes and weights for interval [a,b]
        
        Args:
            N (int): Number of points
            a (float): Left endpoint
            b (float): Right endpoint
        
        Returns:
            tuple: (x,w) nodes and weights
        """
        x, w = roots_legendre(N)
        x = 0.5 * (b - a) * x + 0.5 * (a + b)
        w = 0.5 * (b - a) * w
        return x, w
    
    def print_sparse_matrix(self):
        """
        Print the non-zero values of the sparse matrix K in (i, j) value format.
        """
        coo = self.K.tocoo()  # Convert to COO format
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if v != 0:
                print(f"({i}, {j}) {v}")


if __name__ == "__main__":
    import hex_structural_examples as examplesStructural
    #import plots

    #nDOFDesired=42
    #nDOFDesired=10500
    nDOFDesired=3000
    mesh, mat_prop, bc, elem_body_force = examplesStructural.createBeamSurfaceLoadProblem(nDOFDesired)	
    #mesh, mat_prop, bc, elem_body_force = examplesStructural.createBeamSurfaceLoadProblem(nDOFDesired, L = [0.5, 0.05, 0.05], youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 10000)	
    
    #plots.plotMesh(mesh, bc, title = 'Large Deformation Beam')
    # Create test instance
    ldFEA = largeDeformationFEA(mesh, mat_prop, bc)
    
    start_time = time.time()
    ldFEA.solve_nonlinear_fem(verbose = True)
    end_time = time.time()
    
    ldFEA.compute_strain_energy()
    print(f'Strain Energy: {ldFEA.strain_energy}')
    
    # Log the time taken
    time_taken = end_time - start_time
    print(f"Time taken for solve_nonlinear_fem: {time_taken} seconds")

    delta = np.max(np.abs(ldFEA.deformation))
    # Store results
    nDOF = ldFEA.mesh.num_nodes*3
    print("-----------------------------")
    print(f'Problem: BeamBending')    
    print("Total num nodes: ", nDOF)
    print("FEA time: ", time_taken)
    #print("Total load: ", totalLoad)
    print("Max deformation: ", delta)
    print(f'Strain Energy: {ldFEA.strain_energy}')
    print("-----------------------------")
    
    #plots.plotMesh(ldFEA.mesh, bc=ldFEA.boundary_conditions, u=ldFEA.sol, title = 'Large Deformation Beam')