"""Linear solvers for finite element analysis."""

import enum
import numpy as np
import scipy.sparse as spy_sprs
import scipy.sparse.linalg as spy_linalg
from scipy.sparse.linalg import spilu, LinearOperator
import pyamg # pip install pyamg
# Mac does not support pypardiso, so we skip 
try:
  import pypardiso # pip install pypardiso
except ImportError:
  pypardiso = None



import bound_cond

class Preconditioners(enum.Enum):
  JACOBI = enum.auto()
  ILU = enum.auto()

DEFAULT_TOL = 1.e-6 # Default tolerance for iterative solvers

def _jacobi_preconditioner(A: spy_sprs.coo_matrix,
                          eps_tol: float = 1.e-12,
                          ) -> spy_sprs.coo_matrix:
  """Compute the Jacobi preconditioner for a sparse matrix A.

  Args:
    A: The sparse stiffness matrix in COO format.
    eps_tol: The tolerance for the diagonal entries of A. If the absolute value
      of the diagonal entry is less than this value, it is set to 1.0 to avoid
      division by zero.

  Returns: The Jacobi preconditioner in COO format.
  """
  diag_data = A.diagonal()
  diag_data[np.abs(diag_data) < eps_tol] = 1.0  # Avoid division by zero
  diag_idxs = np.arange(A.shape[0])
  return spy_sprs.coo_matrix((1.0 / diag_data, (diag_idxs, diag_idxs)),
                                shape=A.shape)


def _ilu_preconditioner(A: spy_sprs.coo_matrix,
                        drop_tol: float = 1.e-1,
                        ) -> spy_linalg.LinearOperator:
  """Compute the ILU preconditioner for a sparse matrix A.
  
  Args:
    A: The sparse stiffness matrix in COO format.
    drop_tol: The tolerance for dropping small entries in the ILU factorization.
  """
  ilu = spy_linalg.spilu(A, drop_tol = drop_tol)
  return spy_linalg.LinearOperator(A.shape, matvec= ilu.solve)


def get_preconditioner(A: spy_sprs.coo_matrix,
                      preconditioner: Preconditioners,
                      **kwargs,
                      ) -> spy_linalg.LinearOperator:
  """Get the preconditioner for the linear solver.

  Args:
    A: The sparse stiffness matrix in COO format.
    preconditioner: The preconditioner from `Preconditioners` to use.

  Returns: The preconditioner as a linear operator.
  """
  if preconditioner == Preconditioners.JACOBI:
    return _jacobi_preconditioner(A, **kwargs)

  elif preconditioner == Preconditioners.ILU:
    return _ilu_preconditioner(A,  drop_tol=1e-2, fill_factor=5, permc_spec='COLAMD',**kwargs)

  else:
    return spy_sprs.eye(A.shape[0])


class Solvers(enum.Enum):
	SPSOLVE = enum.auto()
	PCG = enum.auto()
	PYAMG = enum.auto()
	DPCG = enum.auto()
	PARDISO = enum.auto()


def solve(A: spy_sprs.coo_matrix, 
					b: np.ndarray,
		      solver: Solvers,
          bc: bound_cond.BC,
					**kwargs,
					)-> np.ndarray:
  """Solve a linear system of equations.

      Solve for x in Ax = b for x using the specified solver.

  Args:
    A: The stiffness matrix of the system of shape (n,n).
    b: The right hand side of the system of shape (n,).
    solver: The solver from `Solvers` to use.

  Returns: The solution x of the system of shape (n,).
  """

  def solver_wrapper(A0, b0):
 
    A, b = bound_cond.impose_dirichlet_bc(A0.tocsr(), b0, bc)

 
    if solver == Solvers.SPSOLVE:
      x = spy_linalg.spsolve(A, b) # very slow for large problems

    elif solver == Solvers.PCG:
      rtol = kwargs.get('rtol', DEFAULT_TOL) # for iterative solvers
      M = _jacobi_preconditioner(A)
      #M = _ilu_preconditioner(A) # ILU preconditioner takes too long
      x, _ = spy_linalg.cg(A, b, M = M, rtol = rtol)

    elif solver == Solvers.PYAMG:
      # Smoothed Aggregation solver gives the wrong result
      #ml = pyamg.smoothed_aggregation_solver(A, B=b, smooth='energy')
      #x = ml.solve(b, tol=kwargs['rtol'])
      rtol = kwargs.get('rtol', DEFAULT_TOL) # for iterative solvers
      x = pyamg.solve(A, b,tol= rtol, verb = False)

    elif solver == Solvers.DPCG:
      dsolver = kwargs['dsolver']
      rtol = kwargs.get('rtol', DEFAULT_TOL) # for iterative solvers
      M = _jacobi_preconditioner(A)
      x = dsolver.deflatedPCG(A,
                              b,
                              W = dsolver.W,
                              M = M,
                              rtol = rtol)

    elif solver == Solvers.PARDISO:
      x = pypardiso.spsolve(A, np.array(b))
      pypardiso.ps.free_memory()
    else:
      raise ValueError('Unknown solver type')

    u = np.zeros(b0.shape)
    u[bc.fixed_dofs] = bc.dirichlet_values
    u[bc.free_dofs] = x
    return u
  return solver_wrapper(A,b)
