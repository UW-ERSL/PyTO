"""Thermal Finite Element Analysis."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import element_stiffness as elem_stiff
import mat_lib
import bound_cond
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

class ThermalFEA:
  """Linear Thermal Finite Element Analysis."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.ThermalMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_thermal(mat_prop, mesh.elem_size))

    self.node_idx = jnp.stack((
                      np.kron(self.mesh.edofMat, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 8))).flatten())
                      ).T.astype(int)


  def solve(self, x: jnp.ndarray = None,) -> jnp.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = jnp.ones((self.mesh.num_elems,))
    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff, x).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

    u =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return u

if __name__ == "__main__":
    import plots
    import thermal_fea as fea
    import linear_solvers as lin_solv
    import jax # import jax to enable 64 bit precision
    import time	
    from examples_thermal import *
    jax.config.update("jax_enable_x64", True)

    problem = ThermalExamples.Moran
    nDOFDesired = 50000
    mesh, mat_prop, bc = getThermalProblem(problem,nDOFDesired = nDOFDesired)
    solver = lin_solv.Solvers.PARDISO # typically DPCG or PARDISO
  
    fe_solver = fea.ThermalFEA(mesh = mesh,
                          mat_prop = mat_prop,
                          bc = bc,
                          solver = solver)

    startTime = time.time()
    u = np.asarray(fe_solver.solve())
    
    uMax = np.max(np.abs(u))
    nDOF = fe_solver.mesh.num_nodes
   
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max u: ', uMax)
    print('-----------------------------')
	
    plots.plotMesh(fe_solver.mesh, None, u,title=f'Dof = {nDOF}, max u: {uMax:.3e}',show_edges=False,)
