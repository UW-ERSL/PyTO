"""Boundary conditions module."""

import dataclasses
import numpy as np
import scipy.sparse as spy_sprs
import torch


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
  
def apply_dirichlet_bc_torch(K, f, bc: BC):
  """Imposes Dirichlet boundary conditions on a torch-based linear system.

  Modifies the right-hand side to account for prescribed displacements and
  enforces them in the stiffness matrix using a penalty-free replacement
  (rows/cols zeroed, diagonal set to 1).

  Args:
    K: Stiffness matrix of the system, as a torch sparse COO tensor of shape (n, n).
    f: Right hand side vector of the system, as a dense torch tensor of shape (n,).
    bc: Boundary conditions object of type BC, with fields:
        - fixed_dofs: numpy array of fixed dof indices.
        - dirichlet_values: numpy array of prescribed displacement values.

  Returns:
    A tuple (K_mod, f_mod) where:
      K_mod: Modified stiffness matrix as a torch sparse COO tensor of shape (n, n).
      f_mod: Modified right hand side as a dense torch tensor of shape (n,).
  """

  device, dtype = K.device, K.dtype
  ndof = K.shape[0]

  fixed = torch.from_numpy(bc.fixed_dofs).to(device=device, dtype=torch.long)
  dir_vals = torch.from_numpy(bc.dirichlet_values).to(device=device, dtype=dtype)

  # prescribed displacement vector
  u_pres = torch.zeros(ndof, device=device, dtype=dtype)
  u_pres[fixed] = dir_vals

  # coalesce once
  Kc = K.coalesce()
  idx = Kc.indices()
  val = Kc.values()

  # modify RHS: f_mod = f - K * u_pres, then enforce dirichlet values at fixed dofs
  reaction = torch.sparse.mm(Kc, u_pres.unsqueeze(1)).squeeze(1)
  f_mod = f - reaction
  f_mod[fixed] = dir_vals

  # zero rows and cols corresponding to fixed dofs
  rows, cols = idx[0], idx[1]
  mask = torch.isin(rows, fixed) | torch.isin(cols, fixed)
  val = torch.where(mask, torch.tensor(0.0, device=device, dtype=dtype), val)

  K_mod = torch.sparse_coo_tensor(idx, val, Kc.shape, device=device, dtype=dtype)

  # add identity on fixed dofs (K_ii = 1)
  diag_idx = torch.stack([fixed, fixed], dim=0)
  diag_val = torch.ones_like(fixed, dtype=dtype)
  K_diag = torch.sparse_coo_tensor(diag_idx, diag_val, Kc.shape, device=device, dtype=dtype)

  return (K_mod + K_diag).coalesce(), f_mod