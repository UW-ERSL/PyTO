"""Linear solvers for finite element analysis."""

import enum
import numpy as np

import jax
import jax.numpy as jnp

import scipy.sparse as spy_sprs
import scipy.sparse.linalg as spy_linalg

import pyamg # pip install pyamg
import pypardiso # pip install pypardiso

import bound_cond

#import ilupp # pip install ilupp

class Preconditioners(enum.Enum):
  JACOBI = enum.auto()
  ILU = enum.auto()


def _jacobi_preconditioner(A: spy_sprs.coo_matrix,
                          eps_tol: float = 1.e-18,
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
    return _ilu_preconditioner(A, **kwargs)

  else:
    return spy_sprs.eye(A.shape[0])


class Solvers(enum.Enum):
	SPSOLVE = enum.auto()
	CG = enum.auto()
	PYAMG = enum.auto()
	DPCG = enum.auto()
	PARDISO = enum.auto()
	#ILUPP = enum.auto()


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
  def mv(u):
    return A @ u

  def solver_wrapper(A0, b0):
    A_sp = spy_sprs.coo_matrix((np.asarray(A0.data),
                                (A0.indices[:,0], A0.indices[:,1])),
                                shape=A0.shape)
    A, b = bound_cond.impose_dirichlet_bc(A_sp.tocsr(), b0, bc)

    if solver == Solvers.SPSOLVE:
      x = spy_linalg.spsolve(A, b)

    elif solver == Solvers.CG:
      M = _jacobi_preconditioner(A)
      #M = _ilu_preconditioner(A) # ILU preconditioner takes too long
      x, _ = spy_linalg.cg(A, b, M = M, rtol = kwargs['rtol'])

    elif solver == Solvers.PYAMG:
      # Smoothed Aggregation solver gives the wrong result
      #ml = pyamg.smoothed_aggregation_solver(A, B=b, smooth='energy')
      #x = ml.solve(b, tol=kwargs['rtol'])
      x = pyamg.solve(A, b,tol= kwargs['rtol'], verb = kwargs['verbose'])

    elif solver == Solvers.DPCG:
      dsolver = kwargs['dsolver']
      M = _jacobi_preconditioner(A)
      x = dsolver.deflatedPCG(A,
                              b,
                              W = dsolver.W,
                              M = M,
                              rtol = kwargs['rtol'])

    elif solver == Solvers.PARDISO:
      x = pypardiso.spsolve(A, np.array(b))
      pypardiso.ps.free_memory()
    # elif solver == Solvers.ILUPP:
    #   print("Does not seem to work ...")
    #   iChol = ilupp.icholt(A, add_fill_in=0, threshold=0.1)
    #   x, _ = spy_linalg.cg(A, b, M = iChol, rtol = kwargs['rtol'])
    else:
      raise ValueError('Unknown solver type')

    u = jnp.zeros(b0.shape)
    u = u.at[bc.free_dofs].set(x)
    return u

  result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
  cust_solver = lambda mv, b: jax.pure_callback(solver_wrapper, result_shape, A, b)
  sol = jax.lax.custom_linear_solve(mv, b, cust_solver, symmetric=True)
  return sol.reshape(-1)