import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import mat_lib
import bound_cond
import pyvista as pv # pip install pyvista

class StructFEATet:
  """Linear Structural Finite Element Analysis using 10-noded quadratic tet elements."""

  def __init__(self,
							 quadratic_tet_mesh,
							 mat_prop: mat_lib.StructuralMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    
    self.mesh = quadratic_tet_mesh
    self.mat_prop, self.bc =  mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
 
    self.createEdofMatStructural()

  def createEdofMatStructural(self):
        """
        Create the element degree of freedom matrix for structural analysis.
        This function creates a matrix that maps each element to its corresponding degrees of freedom.
        The matrix is structured such that each node in an element contributes three degrees of freedom (x, y, z).
        Attributes:
        self.edofMat (numpy.ndarray): The element degree of freedom matrix.
        """
        self.edofMat = np.zeros((self.mesh.num_elems, 30), dtype=int)
        for i in range(self.mesh.num_elems):
            for j in range(10):
                self.edofMat[i, j*3:j*3+3] = [3*self.mesh.elems[i, j], 3*self.mesh.elems[i, j]+1, 3*self.mesh.elems[i, j]+2]

  def tet10_stiffness_matrix_structural(self, elem_nodes):
        """Calculate element stiffness matrix for 10-noded quadratic tetrahedral element."""
        # Gauss quadrature points and weights for tetrahedral elements
        gp = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 0.1666666667, 0.1666666667],
            [0.1666666667, 0.5, 0.1666666667],
            [0.1666666667, 0.1666666667, 0.5],
            [0.1666666667, 0.1666666667, 0.1666666667]
        ])
        w = np.array([0.0714285714, 0.0476190476, 0.0476190476, 0.0476190476, 0.0476190476])

        # Initialize element stiffness matrix
        k_elem = np.zeros((30, 30))  # 10 nodes * 3 DOF per node

        # Material properties
        E = self.mat_prop.E
        nu = self.mat_prop.nu
        D = (E/(1+nu)/(1-2*nu)) * np.array([
            [1-nu, nu, nu, 0, 0, 0],
            [nu, 1-nu, nu, 0, 0, 0],
            [nu, nu, 1-nu, 0, 0, 0],
            [0, 0, 0, (1-2*nu)/2, 0, 0],
            [0, 0, 0, 0, (1-2*nu)/2, 0],
            [0, 0, 0, 0, 0, (1-2*nu)/2]
        ])

        # Integration over Gauss points
        for i in range(len(w)):
            xi, eta, zeta = gp[i]
            # Shape functions derivatives
            dN = self._shape_deriv(xi, eta, zeta)
            J = dN @ elem_nodes
            dN_xyz = np.linalg.solve(J.T, dN.T).T

            # B matrix assembly
            B = np.zeros((6, 30))
            for n in range(10):
                B[0, n*3] = dN_xyz[n,0]
                B[1, n*3+1] = dN_xyz[n,1]
                B[2, n*3+2] = dN_xyz[n,2]
                B[3, n*3:n*3+2] = [dN_xyz[n,1], dN_xyz[n,0]]
                B[4, n*3+1:n*3+3] = [dN_xyz[n,2], dN_xyz[n,1]]
                B[5, [n*3, n*3+2]] = [dN_xyz[n,2], dN_xyz[n,0]]

            k_elem += B.T @ D @ B * w[i] * np.linalg.det(J)

        return k_elem

  def assemble_global_stiffness_matrix(self):
    """Assemble the global stiffness matrix for the structural analysis."""
    num_dof = self.mesh.num_nodes * 3
    rows, cols, data = [], [], []
    
    for elem_idx in range(self.mesh.num_elems):
        # Get nodal coordinates for this element
        elem_nodes = self.mesh.nodes[self.mesh.elems[elem_idx]]
        # Calculate element stiffness matrix
        k_elem = self.tet10_stiffness_matrix_structural(elem_nodes)
        
        # Get global DOF indices for this element
        dof = np.array([[3*n, 3*n+1, 3*n+2] for n in self.mesh.elems[elem_idx]]).flatten()
        
        # Add to global matrix using COO format
        for i in range(30):
            for j in range(30):
                rows.append(dof[i])
                cols.append(dof[j])
                data.append(k_elem[i,j])
    
    # Create sparse matrix
    K_global = jax_sprs.BCOO((data, (rows, cols)), shape=(num_dof, num_dof))
    return K_global
  

  def _shape_deriv(self, xi, eta, zeta):
        """Calculate shape function derivatives for 10-noded tet."""
        L1 = 1 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

        dN = np.array([
            [4*L1 - 1,   4*L2 - 1,   4*L3 - 1,   4*L4 - 1,
            4*(L1 + L2), 4*(L2 + L3), 4*(L3 + L1),
            4*(L1 + L4), 4*(L2 + L4), 4*(L3 + L4)]
        ])
        return dN

  def solve(self) -> jnp.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
  
    self.sol =  lin_sol.solve(self.K,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return self.sol

  def plot(self, title = 'Quadratic Tet Mesh'):
        # For visualization, we'll convert back to linear tets by only using corner nodes
        linear_elems = self.mesh.elems[:, :4]  # Extract only corner nodes
        plotter = pv.UnstructuredGrid({pv.CellType.TETRA: linear_elems}, self.mesh.nodes)
        plotter.plot(show_edges=True, show_scalar_bar=False, show_grid=True)
        print(f"Number of nodes: {self.mesh.num_nodes}, Number of elements: {self.mesh.num_elems}")


if __name__ == "__main__":
    from tet_mesher import TetMesher
    import jax # import jax to enable 64 bit precision
    import time	
    jax.config.update("jax_enable_x64", True)

    
    

    