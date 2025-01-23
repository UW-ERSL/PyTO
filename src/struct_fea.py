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
                    elem_stiff.hex8_stiffness_matrix(mat_prop, mesh.elem_size))

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