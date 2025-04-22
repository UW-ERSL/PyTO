import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import mat_lib
import bound_cond

class ThermalFEATet:
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

  def tet4_stiffness_matrix_thermal(self,
          mat_prop: mat_lib.ThermalMaterial,
          xyz_nodes: jnp.ndarray,
        ) -> jnp.ndarray:
    """Computes the element stiffness matrix of a tetrahedral element in 3D.
    The stiffness matrix for linear thermal is derived as:

          K = sum_gauss(B'D B |J| w )
      Where,

        B is the strain displacement matrix.
        D is the constitutive matrix.
        J is the Jacobian.
        w is the gauss weight.
    Args:
    mat_prop: The thermal material properties of the element.
    xyz_nodes: Array of (4, 3) containing the x, y, z coordinates of the nodes.

    Returns: The element stiffness matrix of size (4, 4)
    """
    K = mat_prop.thermal_conductivity
    # Initialize the element stiffness matrix.
    ke = np.zeros((4, 4))

    # Define the shape function gradients in the reference element
    dN_dxi = np.array([[-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1]])

    # Compute the Jacobian matrix
    jac = dN_dxi @ xyz_nodes

    det_jac = np.linalg.det(jac)
    inv_jac = np.linalg.inv(jac)

    # Compute the shape function gradients in the physical element
    dN_dxyz = inv_jac @ dN_dxi

    # Compute the element stiffness matrix
    ke = K * (dN_dxyz.T @ dN_dxyz) * det_jac / 6.0

    return ke

  
  def tet4_specific_heat_matrix(self,
          mat_prop: mat_lib.ThermalMaterial,
          xyz_nodes: jnp.ndarray,
        ) -> jnp.ndarray:
    """Computes the element specific heat matrix of a tetrahedral element in 3D.
    The specific heat matrix is simplified as:

          C = (1/20) * C*V*rho * predefined_matrix

    Args:
    mat_prop: The thermal material properties of the element.
    xyz_nodes: Array of (4, 3) containing the x, y, z coordinates of the nodes.

    Returns: The element specific heat matrix of size (4, 4)
    """
    # Compute the Jacobian matrix
    dN_dxi = np.array([[-1, 1, 0, 0],
                       [-1, 0, 1, 0],
                       [-1, 0, 0, 1]])
    jac = dN_dxi @ xyz_nodes
    det_jac = np.linalg.det(jac)

    # Define the predefined matrix of 1s and 2s
    predefined_matrix = np.array([
        [2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2]
    ])
    C=mat_prop.specific_heat
    rho=mat_prop.mass_density
    # Compute the specific heat matrix
    ce = (1 / 120) * det_jac * predefined_matrix*C*rho
    # print(f"Specific heat matrix (ce):\n{ce}")
    # t=1
    return ce

  def assemble_global_stiffness_matrix(self):
    """Assemble the global stiffness matrix."""
    # Initialize the global stiffness matrix in COO format
    data = []
    for i in range(self.mesh.num_elems):
      ke = self.tet4_stiffness_matrix_thermal(self.mat_prop, self.mesh.nodes[self.edofMat[i, :]])
      data.append(ke.flatten())

    self.node_idx = jnp.stack((
              np.kron(self.edofMat, np.ones((4, 1))).flatten(),
              np.kron(self.edofMat, np.ones((1, 4))).flatten())
              ).T.astype(int)
  
    ke_stacked = jnp.concatenate(data)
    # Create the sparse global stiffness matrix
    self.K = jax_sprs.BCOO((ke_stacked, self.node_idx),
                  shape=(self.bc.num_dofs, self.bc.num_dofs))
    
    

    
  def solve(self) -> jnp.ndarray:
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

if __name__ == "__main__":
    import jax # import jax to enable 64 bit precision
    import time	
    from tet_examples_thermal import *
    jax.config.update("jax_enable_x64", True)

    
    # Create lists to store DOFs and corresponding uMax values
    dof_range = [100,500, 1000,2000]  # Range of desired DOFs
    dofs = []
    u_maxs = []
    timing = []
    example = 3
    for nDOFDesired in dof_range:
      if example == 1:
        tetmesh, mat_prop, bc = createThickPlateThermalProblemTet(nDOFDesired=nDOFDesired)
      elif example == 2:
        tetmesh, mat_prop, bc = createLBracketThermalProblemTet(nDOFDesired=nDOFDesired)
      elif example == 3:
        tetmesh, mat_prop, bc = createAnnularPlateThermalProblemTet(nDOFDesired=nDOFDesired)
    
      solver = lin_sol.Solvers.PARDISO
      
      fe_solver = ThermalFEATet(mesh=tetmesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)

      startTime = time.time()
      fe_solver.assemble_global_stiffness_matrix()
      u = np.asarray(fe_solver.solve())
      uMax = np.max(np.abs(u))
      
      # Store results
      nDOF = fe_solver.mesh.num_nodes
      dofs.append(nDOF)
      u_maxs.append(uMax)
      timing.append(time.time() - startTime)
      print('-----------------------------')
      print("nDof: ", nDOF)
      print('Solver: ', fe_solver.solver.name)
      print("FEA time: ", time.time() - startTime)
      print('Max u: ', uMax)
      print('-----------------------------')
    
    # Plot DOF vs uMax
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(dofs, np.array(u_maxs), 'bo-')
    plt.xlabel('Number of DOFs')
    plt.ylabel('Maximum Temperature')
    plt.ylim(0.9*min(u_maxs), 1.1*max(u_maxs))
    plt.grid(True)
    plt.title(f'Convergence Study - TetMesh')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(dofs, np.array(timing), 'bo-')
    plt.xlabel('Number of DOFs')
    plt.ylabel('Timing (secs)')
    plt.grid(True)
    plt.title(f'Timing - TetMesh')
    plt.show()
   
   
	
    tetmesh.plotField(u,show_edges=True) # plot the solution field
