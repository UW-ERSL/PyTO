import numpy as np
from scipy.sparse import csr_matrix

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

    def solve_nonlinear_fem(self,verbose = False, n_steps = 10,max_iter = 30,tol = 1e-9):
        """
        Solve nonlinear finite element problem using Newton-Raphson method.
        """
        success = False
        n_dof = 3*self.mesh.num_nodes
        self.sol = np.zeros(n_dof)
        for step in range(1, n_steps + 1):
            if verbose:
                print(f'Load Step {step}/{n_steps}')
            load_factor = step/n_steps
            
            iter = 0
            err = 1
            while (iter < max_iter) and (err > tol):
                iter += 1
                self.assemble_k()  # Assembles K, M matrices and F vector
                all_dof = np.arange(n_dof)
                self.delta_sol = np.zeros(n_dof)
                self.free_dof = np.setdiff1d(all_dof, self.fixed_dof)
                b = self.F - (load_factor * self.neuman_force)
                # Fix constrained nodes
                for i in self.fixed_dof:
                    self.K[i,:] = 0
                    self.K[i,i] = 1
                    b[i] = -load_factor * self.nodal_dirichlet[i] + self.sol[i]
                
                d_sol = -np.linalg.solve(self.K, b)
                self.delta_sol = d_sol[:n_dof]
                self.sol += self.delta_sol
                
                self.u = self.sol[::3]
                self.v = self.sol[1::3]
                self.w = self.sol[2::3]
                
                err = np.linalg.norm(self.delta_sol)/np.linalg.norm(self.sol)
                
                if verbose:
                    print(f'{iter}\t {err:E}\t {np.linalg.norm(b)/n_coord/self.num_nodes:E}')
        
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
        position_nodes = self.mesh.p[:3, nodes]
        _, grad_N = self.shape_function(xi)
        return grad_N @ position_nodes.T

    def assemble_k(self):
        """
        Assemble global stiffness matrix and force vector
        """
        self.dof_per_node = 3
        n_dof = self.dof_per_node*self.mesh.num_nodes
        
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
            
            dof = np.array([3*nodes-2, 3*nodes-1, 3*nodes]).flatten()
            dof = dof.reshape(1, self.dof_per_elem)
            
            row_index = np.repeat(dof, self.dof_per_elem)
            col_index = np.tile(dof, self.dof_per_elem)
            entries = k_elem.flatten()
            
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
        elem_nodes = self.mesh.elemArray[:,elem]
        sol = np.vstack((self.sol[3*elem_nodes-2], self.sol[3*elem_nodes-1], self.sol[3*elem_nodes]))

        for g in range(num_gq):
            grad_n_all = grad_n_cell[g]
            j_total = self.jacobian(elem, xi_gq[:,g])
            grad_ndx = np.linalg.solve(j_total, grad_n_all)
            F = np.eye(3) + sol.T @ grad_ndx.T
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
        self.strain_energy = 0.5 * self.sol @ (self.K[:self.num_dof, :self.num_dof] @ self.sol)
        return self
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
        
        # Initialize arrays
        n_total = num_gq * num_gq * num_gq
        xi_gq = np.zeros((3, n_total))
        wt_gq = np.zeros(n_total)
        
        # Build 3D points using tensor product
        idx = 0
        for i in range(num_gq):
            for j in range(num_gq):
                for k in range(num_gq):
                    xi_gq[0,idx] = x[i]
                    xi_gq[1,idx] = x[j] 
                    xi_gq[2,idx] = x[k]
                    wt_gq[idx] = w[i] * w[j] * w[k]
                    idx += 1
                    
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
        N = N-1
        N1 = N+1
        N2 = N+2
        
        xu = np.linspace(-1, 1, N1)
        y = np.cos((2*np.arange(N+1)+1)*np.pi/(2*N+2)) + (0.27/N1)*np.sin(np.pi*xu*N/N2)
        
        L = np.zeros((N1,N2))
        Lp = np.zeros((N1,N2))
        
        y0 = 2
        while np.max(np.abs(y-y0)) > np.finfo(float).eps:
            L[:,0] = 1
            Lp[:,0] = 0
            L[:,1] = y
            Lp[:,1] = 1
            for k in range(2,N1):
                L[:,k+1] = ((2*k-1)*y*L[:,k]-(k-1)*L[:,k-1])/k
                
            Lp = (N2)*(L[:,N1-1]-y*L[:,N2-1])/(1-y**2)
            
            y0 = y
            y = y0-L[:,N2-1]/Lp
            
        x = (a*(1-y)+b*(1+y))/2
        w = (b-a)/((1-y**2)*Lp**2)*(N2/N1)**2
        
        return x, w
    
if __name__ == "__main__":
    import examples_structural as examplesStructural
    import plots

    
    mesh, mat_prop, bc = examplesStructural.createBeamSurfaceLoadProblem(nDOFDesired=3000)	
    
    #plots.plotMesh(mesh, bc, title = 'Large Deformation Beam')
    # Create test instance
    ldFEA = largeDeformationFEA(mesh, mat_prop, bc)
    ldFEA.solve_nonlinear_fem(verbose = True)
    ldFEA.compute_strain_energy()
    print(f'Strain Energy: {ldFEA.strain_energy}')
    plots.plotDeformation(mesh, ldFEA, title = 'Large Deformation Beam')