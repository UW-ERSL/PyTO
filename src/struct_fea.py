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
               elem_body_force: jnp.ndarray = None,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_structural(mat_prop, mesh.elem_size))

    self.node_idx = jnp.stack((
                      np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
                      ).T.astype(int)
    self.elem_body_force = elem_body_force


  def solve(self, elem_material_scaling: jnp.ndarray = None) -> jnp.ndarray:
    """Solve the structural finite element problem.

    Args:
      elem_material_scaling: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if elem_material_scaling is None:
      elem_material_scaling = jnp.ones((self.mesh.num_elems,))

    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff,
									               elem_material_scaling).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
  

    if self.elem_body_force is not None:
      elem_force = self.elem_body_force.copy()
      for i in range(3):
        elem_force[i::3]  *= elem_material_scaling[:]
      nNodes = self.mesh.num_nodes
      tStart  = time.time()
      node_forces = np.zeros((nNodes * 3,))
      node_forces[0::3] = self.mesh.elem_to_node_field_mapping* elem_force[0::3] 
      node_forces[1::3] = self.mesh.elem_to_node_field_mapping* elem_force[1::3] 
      node_forces[2::3] = self.mesh.elem_to_node_field_mapping* elem_force[2::3] 
      
      self.bc.force += node_forces


    u =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return u

if __name__ == "__main__":    
  jax.config.update("jax_enable_x64", True)

  from examples_structural import *


  example = 1
  elem_body_force = None # by default no body force


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
  elif example == 9:
    mesh, mat_prop, bc, elem_body_force  = createCircularPlateProblem(nDOFDesired=100000)
  elif example == 10:
    mesh, mat_prop, bc = createArrowHeadProblem(nDOFDesired=200000)

  fe_solver = fea.StructFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = lin_solv.Solvers.PARDISO,
        elem_body_force = elem_body_force)


  startTime = time.time()
  u = np.asarray(fe_solver.solve())
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