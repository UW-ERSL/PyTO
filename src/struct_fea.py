"""Structural Finite Element Analysis."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import element_stiffness as elem_stiff
import mat_lib
import bound_cond
import plots
import struct_fea as fea
import linear_solvers as lin_solv
import mat_lib
import os


script_dir = os.path.dirname(os.path.abspath(__file__))


class StructFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.StructuralMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_structural(mat_prop, mesh.elem_size))

    self.node_idx = jnp.stack((
                      np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
                      ).T.astype(int)


  def solve(self, elem_youngs_modulus: jnp.ndarray) -> jnp.ndarray:
    """Solve the structural finite element problem.

    Args:
      elem_youngs_modulus: Array of (num_elems,) of the young's modulus of each
        element.

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """

    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff,
									               elem_youngs_modulus).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

    
    u =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return u

if __name__ == "__main__":    
  jax.config.update("jax_enable_x64", True)

  from examples_structural import *

  example = 5
  if example == 1:
    mesh, mat_prop, bc = createCantileverProblem(nDOFDesired=10000)
  elif example == 2:
    mesh, mat_prop, bc = createMBBProblem(nDOFDesired=10000)   
  elif example == 3:
    mesh, mat_prop, bc = createDistributedLoadProblem(nDOFDesired=10000)    
  elif example == 4:
    mesh, mat_prop, bc = createMultiloadProblem(nDOFDesired=10000)
  elif example == 5:
    mesh, mat_prop, bc = createLBracketProblem(nDOFDesired=10000)    
  elif example == 6:
    mesh, mat_prop, bc = createCompliantMechanismProblem(nDOFDesired=10000)
  elif example == 7:
    mesh, mat_prop, bc = createBeamSurfaceLoadProblem(nDOFDesired=10000)
  elif example == 8:
    mesh, mat_prop, bc = createFilletedBeamProblem(nDOFDesired=100000)

  fe_solver = fea.StructFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = lin_solv.Solvers.SPSOLVE)

  youngs_modulus = np.ones((fe_solver.mesh.num_elems,))
  startTime = time.time()
  u = np.asarray(fe_solver.solve(elem_youngs_modulus= youngs_modulus))
  delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
  deltaMax = np.max(delta)
  nDOF = 3*fe_solver.mesh.num_nodes

  print('-----------------------------')
  print("nDof: ", nDOF)
  print('Solver: ', fe_solver.solver.name)
  print("FEA time: ", time.time() - startTime)
  print('Max displacement: ', deltaMax)
  print('-----------------------------')

  plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
      title=f'dof = {nDOF}, Max deformation: {deltaMax:.3e}')