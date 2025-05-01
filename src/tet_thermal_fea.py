import time
import numpy as np
import linear_solvers as lin_sol
import mat_lib
import bound_cond
import pyvista as pv # pip install pyvista
from numba import njit
import scipy.sparse as sp

@njit(cache=True)
def tet4_stiffness_matrix_thermal(thermal_conductivity,xyz_nodes):
  """Computes the element stiffness matrix of a tetrahedral element in 3D.
  

  Returns: The element stiffness matrix of size (4, 4)
  """
  xyz_nodes = np.asarray(xyz_nodes, dtype=np.float64)
  # Define the shape function gradients in the reference element
  dN_dxi = np.array([[-1.0, 1.0, 0.0, 0.0],
                      [-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0, 1.0]], dtype=np.float64)

  # Compute the Jacobian matrix
  jac = dN_dxi @ xyz_nodes

  det_jac = np.linalg.det(jac)
  # Compute the shape function gradients in the physical element using solve
  dN_dxyz = np.linalg.solve(jac, dN_dxi)

  # Compute the element stiffness matrix
  ke = thermal_conductivity * (dN_dxyz.T @ dN_dxyz) * (det_jac) / 6.0
  return ke

@njit(cache=True)
def tet4_specific_heat_matrix(specific_heat: float,
      mass_density: float,
        xyz_nodes: np.ndarray,
      ) -> np.ndarray:
  """Computes the element specific heat matrix of a tetrahedral element in 3D.
  The specific heat matrix is simplified as:

        C = (1/20) * C*V*rho * predefined_matrix

  Args:
  mat_prop: The thermal material properties of the element.
  xyz_nodes: Array of (4, 3) containing the x, y, z coordinates of the nodes.

  Returns: The element specific heat matrix of size (4, 4)
  """
  xyz_nodes = np.asarray(xyz_nodes, dtype=np.float64)
  # Compute the Jacobian matrix
  dN_dxi = np.array([[-1.0, 1.0, 0.0, 0.0],
                      [-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0, 1.0]], dtype=np.float64)

  jac = dN_dxi @ xyz_nodes

  det_jac = np.linalg.det(jac)

  # Define the predefined matrix of 1s and 2s
  predefined_matrix = np.array([
      [2.0, 1.0, 1.0, 1.0],
      [1.0, 2.0, 1.0, 1.0],
      [1.0, 1.0, 2.0, 1.0],
      [1.0, 1.0, 1.0, 2.0]
  ], dtype=np.float64)


  # Compute the specific heat matrix
  ce = (1 / 120) * det_jac * predefined_matrix*specific_heat*mass_density

  return ce


class TetThermalFEA:
  """Linear Thermal Finite Element Analysis using linear tet elements."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.ThermalMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.createEdofMatThermal()
   
  def createEdofMatThermal(self):
        self.edofMat = np.array(self.mesh.elems[:, :4], dtype=int)

  def assemble_global_stiffness_matrix(self):
    """Assemble the global stiffness matrix."""
    # Initialize the global stiffness matrix in COO format
    data = []
    
    startTime = time.time()
    K = self.mat_prop.thermal_conductivity
    for i in range(self.mesh.num_elems):
      elem_nodes = self.mesh.nodes[self.mesh.elems[i]]
      ke = tet4_stiffness_matrix_thermal(K, elem_nodes)
      data.append(ke.flatten())
    
    rows = np.repeat(self.edofMat, 4, axis=1)
    cols = np.tile(self.edofMat, 4)
    self.node_idx = np.array(np.vstack((rows.flatten(), cols.flatten())).T)
  
    ke_stacked = np.array(np.concatenate(data))
    # Create the sparse global stiffness matrix
    self.K = sp.coo_matrix((ke_stacked, (self.node_idx[:, 0], self.node_idx[:, 1])),
                  shape=(self.bc.num_dofs, self.bc.num_dofs))
  
    
  def solve(self) -> np.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
  
    u =  lin_sol.solve(self.K,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    
    self.sol = u.copy()
    return u
  
  def plotTemperature(self, show_edges =  True, show_scalar_bar = True, show_grid = False):
    plotter = pv.UnstructuredGrid({pv.CellType.TETRA: self.mesh.elems}, self.mesh.nodes)
    plotter.point_data["field"] = self.sol
    # Some common alternatives:
    plotter.plot(show_edges=show_edges, show_scalar_bar=show_scalar_bar, show_grid=show_grid, cmap="jet",
                    scalar_bar_args={ 
                  'title': '',
                  'vertical': True,
                  'position_x': 0.8,
                  'position_y': 0.3,
                  'width': 0.1
                  })      # Classic rainbow colormap
        

if __name__ == "__main__":
    import time	
    from tet_thermal_examples import TetThermalExamples, getTetThermalProblem
  
    
    nDOFDesired = 10000
    problem = TetThermalExamples.AnnularPlate
    tetmesh, mat_prop, bc = getTetThermalProblem(problem, nDOFDesired=nDOFDesired)
  
    solver = lin_sol.Solvers.PARDISO
    fe_solver = TetThermalFEA(mesh=tetmesh,
                mat_prop=mat_prop,
                bc=bc,
                solver=solver)

    startTime = time.time()
    fe_solver.assemble_global_stiffness_matrix()

    u = np.asarray(fe_solver.solve())
    uMax = np.max(np.abs(u))
    
    # Store results
    nDOF = fe_solver.mesh.num_nodes

    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max u: ', uMax)
    print('-----------------------------')

  
    fe_solver.plotTemperature(show_edges=False) # plot the solution field
