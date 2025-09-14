"""Boundary conditions module."""

import dataclasses
import numpy as np
import scipy.sparse as spy_sprs


@dataclasses.dataclass
class BC:
  """Class for handling boundary conditions.

  Attributes:
    force: Array of size (num_dofs,) that contain the imposed load on each dof.
    fixed_dofs: Array of size (num_fixed_dofs,) that contain all the dof numbers
      that are fixed.
    dirichlet_values: Array of size (num_fixed_dofs,) that contain the values of
      the fixed dirichlet BC.
  """
  force: np.ndarray
  fixed_dofs: np.ndarray
  dirichlet_values: np.ndarray
  constraint_matrix: np.ndarray = None # for example, sliding bc
  constraint_rhs: np.ndarray = None # for example, zeros, for sliding bc

  @property
  def num_dofs(self)->int:
    return self.force.shape[0]


  @property
  def free_dofs(self)-> np.ndarray:
    return np.setdiff1d(np.arange(self.num_dofs), self.fixed_dofs)

  def set_force(self, force: np.ndarray):
    """Set the force array.
    Useful for transient problems.
    """
    self.force = force
    
def impose_dirichlet_bc(A: spy_sprs.csr_matrix,
                        b: np.ndarray,
                        bc: BC,
                        ) -> tuple[spy_sprs.csr_matrix, np.ndarray]:
  """Imposes Dirichlet boundary conditions on the system of equations.

  Modifies the right hand side to account for the Dirichlet boundary
  conditions.

  Args:
    A: The stiffness matrix of the system of shape (n,n).
    b: The right hand side of the system of shape (n,).
    bc: The boundary conditions.

  Returns: The modified stiffness matrix and right hand side.
  """
  if (bc.constraint_matrix is  None):
    A_modified = A.copy()
    b_modified = b.copy()

    # Modify the right hand side.
    b_modified -= A[:, bc.fixed_dofs] * bc.dirichlet_values

    # Remove rows and columns corresponding to the Dirichlet boundary conditions.
    A_modified = (
            A_modified[bc.free_dofs, :][:, bc.free_dofs]
            )
    b_modified = b_modified[bc.free_dofs]

    return A_modified, b_modified
  else:
    raise NotImplementedError("impose_dirichlet_bc not implemented for constraint_matrix BCs; see linear_solvers.solver function")