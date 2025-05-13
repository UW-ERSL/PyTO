import time
import numpy as np
import linear_solvers 
import deflation
import mat_lib
import bound_cond
import pyvista as pv # pip install pyvista
from numba import njit
import scipy.sparse as sp

@njit(cache=True)
def tet10_stiffness_matrix_structural(E,nu, elem_nodes):
        """Calculate element stiffness matrix for 10-noded quadratic tetrahedral element."""
        # Gauss quadrature points and weights for tetrahedral elements
        a = 0.1381966011250105
        b = 1-3*a 
        gp = [
                    (a, a, a),
                    (b, a, a),
                    (a, b, a),
                    (a, a, b)
                ]
        w = [1/24] * 4

        # Initialize element stiffness matrix
        k_elem = np.zeros((30, 30))  # 10 nodes * 3 DOF per node

        # Material properties
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
            L4 = 1 - xi - eta - zeta
            dN = -np.array([
                    [
                            4*xi - 1,
                            0,
                            0,
                            -4*L4 + 1,
                            4*eta,
                            0,
                            4*zeta,
                            4*(1 - 2*xi - eta - zeta),
                            -4*eta,
                            -4*zeta
                    ],
                    [
                            0,
                            4*eta - 1,
                            0,
                            -4*L4 + 1,
                            4*xi,
                            4*zeta,
                            0,
                            -4*xi,
                            4*(1 - xi - 2*eta - zeta),
                            -4*zeta
                    ],
                    [
                            0,
                            0,
                            4*zeta - 1,
                            -4*L4 + 1,
                            0,
                            4*eta,
                            4*xi,
                            -4*xi,
                            -4*eta,
                            4*(1 - xi - eta - 2*zeta)
                    ]
            ])
            J = dN @ elem_nodes

            # Jacobian determinant and transformation of derivatives
            dN_xyz = np.linalg.solve(J, dN).T

            # B matrix assembly
            B = np.zeros((6, 30))
            cols = np.arange(0, 30, 3)
            B[0, cols]     = dN_xyz[:, 0]
            B[1, cols+1]   = dN_xyz[:, 1]
            B[2, cols+2]   = dN_xyz[:, 2]
            B[3, cols]     = dN_xyz[:, 1]
            B[3, cols+1]   = dN_xyz[:, 0]
            B[4, cols+1]   = dN_xyz[:, 2]
            B[4, cols+2]   = dN_xyz[:, 1]
            B[5, cols]     = dN_xyz[:, 2]
            B[5, cols+2]   = dN_xyz[:, 0]

            detJ = np.linalg.det(J)
            k_elem += B.T @ D @ B * w[i] * detJ
        return k_elem
class TetStructuralFEA:
  """Linear Structural Finite Element Analysis using 10-noded quadratic tet elements."""

  def __init__(self,
							 quadratic_tet_mesh,
							 mat_prop: mat_lib.Material,
							 bc: bound_cond.BC,
							 solver: linear_solvers.Solvers,
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
        node_indices = np.arange(10)
        dof_indices = np.arange(3)
        for i in range(self.mesh.num_elems):
            # Use broadcasting to create all indices at once
            self.edofMat[i] = (3 * self.mesh.elems[i, node_indices, None] + dof_indices).flatten()


  def assemble_global_stiffness_matrix(self):
        """Assemble the global stiffness matrix for the structural analysis."""
        data = []
        
        for elem_idx in range(self.mesh.num_elems):
            # Get nodal coordinates for this element     
            elem_nodes = self.mesh.node_xyz[self.mesh.elems[elem_idx]]
            # Calculate element stiffness matrix
            k_elem = tet10_stiffness_matrix_structural(self.mat_prop.youngs_modulus, self.mat_prop.poissons_ratio,elem_nodes)
            data.append(k_elem.flatten())

        self.node_idx = np.stack((
                np.kron(self.edofMat, np.ones((30, 1))).flatten(),
                np.kron(self.edofMat, np.ones((1, 30))).flatten())
                ).T.astype(int)
    
        ke_stacked = np.concatenate(data)
        
        self.K = sp.coo_matrix((ke_stacked,(self.node_idx[:, 0], self.node_idx[:, 1])),
                    shape=(self.bc.num_dofs, self.bc.num_dofs))

        return 


  def solve(self) -> np.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
  
    self.sol =  linear_solvers.solve(self.K,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    self.deformation = np.sqrt(self.sol[0::3]**2 + self.sol[1::3]**2 + self.sol[2::3]**2)
    self.max_deformation = np.max(self.deformation)
    
    return 

  def plot_deformation(self):
    """Plot the deformed shape of the mesh."""
    # Create a PyVista plotter
    plotter = pv.Plotter()
    # Create a PyVista mesh from the tetrahedral mesh
    pv_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: self.mesh.elems[:,0:4]}, self.mesh.node_xyz)
    
    sol = self.sol.copy()
    sol = sol.reshape((-1, 3))
   

    deltaMax = self.max_deformation
    L = np.max([np.max(self.mesh.node_xyz[:,i]) - np.min(self.mesh.node_xyz[:,i]) for i in range(3)])
    scale = float(0.2*L/deltaMax)
    deformed_mesh = pv_mesh.copy()
    deformed_mesh.points += scale*sol[:,0:4]
    # Add both original and deformed mesh to the plotter
    plotter.add_mesh(pv_mesh, show_edges=True, color='red', opacity=0.2, label='Original')
    plotter.add_mesh(deformed_mesh, show_edges=True, color='lightblue', opacity=1, label='Deformed')
    
    # Add legend
    plotter.add_legend()
    # Add axes widget
    plotter.add_axes()
    # Show the plot
    plotter.show()  

if __name__ == "__main__":
    import time	
    from tet_structural_examples import TetStructuralExamples, getTetStructuralProblem
   
    problem = TetStructuralExamples.BeamBending 
    quadratic_tet_mesh, mat_prop, bc, elem_body_force  = getTetStructuralProblem(problem,nDOFDesired = 10000)
    
    solver = linear_solvers.Solvers.PARDISO # typically DPCG or PARDISO

    dsolver = deflation.DeflationSolver()
    
    if (solver == linear_solvers.Solvers.DPCG):
        nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*quadratic_tet_mesh.num_nodes/dsolver.dofPerGroup)))
        print('Number of deflation groups: ', nGroups)
        dsolver.create_deflation_groups(quadratic_tet_mesh, nGroups)
        #dsolver.plot_deflation_groups(quadratic_tet_mesh)
        dsolver.create_deflation_matrix(quadratic_tet_mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]
  
    fe_solver = TetStructuralFEA(quadratic_tet_mesh,
                mat_prop=mat_prop,
                bc=bc,
                solver=solver,
                dsolver=dsolver,
                rtol = 1e-8)

    startTime = time.time()
    fe_solver.assemble_global_stiffness_matrix()
    print("Time for assembly: ", time.time() - startTime)
    startTime = time.time()
    fe_solver.solve()
    delta = np.max(np.abs(fe_solver.deformation))
    
    # Store results
    nDOF = 3*fe_solver.mesh.num_nodes
    
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max deformation: ', delta)
    print('-----------------------------')
    fe_solver.plot_deformation()

    