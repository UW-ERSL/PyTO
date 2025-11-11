import torch
import enum
import numpy as np
import scipy.sparse as spy_sprs
from scipy.sparse.linalg import spsolve
from typing import Dict

# Try to use PARDISO if available
try:
    import pypardiso
except ImportError:
    pypardiso = None

# Try to use PETSc if available
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None


class Solvers(enum.Enum):
    SPSOLVE = enum.auto()
    PCG = enum.auto()
    PYAMG = enum.auto()
    PETSC = enum.auto()
    PARDISO = enum.auto()

def _petsc_solve(A_csr: spy_sprs.csr_matrix, rhs, solver_options: dict = None):
    """
    Solve A x = rhs using PETSc via petsc4py, using options
    from the provided dictionary.
    
    This function expects 'solver_options' to be the dictionary that
    might contain a 'petsc_solver' sub-dictionary, matching the JAX logic.
    """
    if PETSc is None:
        raise ImportError("petsc4py is not available; cannot use PETSC solver.")

    if solver_options is None:
        solver_options = {}

    # This logic correctly parses the nested dictionary, just like the JAX version.
    petsc_opts = solver_options.get("petsc_solver", {})
    # --- CHANGED LINES ---
    # Set the DEFAULT ksp_type to "preonly"
    ksp_type = petsc_opts.get("ksp_type", "preonly")
    # Set the DEFAULT pc_type to "cholesky"
    pc_type = petsc_opts.get("pc_type", "cholesky")
    # --- END CHANGED LINES ---

    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(
            A_csr.indptr.astype(PETSc.IntType, copy=False),
            A_csr.indices.astype(PETSc.IntType, copy=False),
            A_csr.data,
        ),
    )

    rhs_vec = PETSc.Vec().createSeq(len(rhs))
    rhs_vec.setValues(range(len(rhs)), np.asarray(rhs))

    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()  # Allow command-line overrides

    # Set types from options dict (or defaults)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)

    # Add the special case from JAX version
    if ksp_type == "tfqmr":
        ksp.getPC().setFactorSolverType("mumps")

    x_vec = PETSc.Vec().createSeq(len(rhs))
    ksp.solve(rhs_vec, x_vec)

    return x_vec.getArray()


def _direct_solve(
    A: spy_sprs.spmatrix, rhs, solver_kind: str, solver_options: dict = None
):
    """Solve A x = rhs using the requested backend."""
    A_csr = A.tocsr()

    if solver_kind == "petsc" and PETSc is not None:
        # Pass the entire options dictionary
        x = _petsc_solve(A_csr, rhs, solver_options)
        print("PETSc used for sparse solve.")
    elif solver_kind == "pardiso" and pypardiso is not None:
        x = pypardiso.spsolve(A_csr, rhs)
        print("pypardiso used for sparse solve.")
        pypardiso.ps.free_memory()
    else:
        x = spsolve(A_csr, rhs)
        print("spsolve used for sparse solve.")
    return x


class SparseLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mtrx: torch.Tensor,
        b: torch.Tensor,
        solver_kind: str,
        solver_params: dict, # Renamed for clarity
    ):
        """
        Solve A x = b for x, where A is a sparse torch tensor.
        """
        # Remember which solver to use in backward
        ctx.solver_kind = solver_kind
        # Save the entire params dictionary for backward pass
        ctx.solver_params = solver_params

        mtrx = mtrx.coalesce()
        mtrx_indices = mtrx.indices().detach().cpu().numpy()
        mtrx_values = mtrx.values().detach().cpu().numpy()
        mtrx_shape = mtrx.shape
        b_numpy = b.detach().cpu().numpy()

        mtrx_scipy = spy_sprs.coo_matrix(
            (mtrx_values, (mtrx_indices[0], mtrx_indices[1])),
            shape=mtrx_shape,
        ).tocsc()

        # Solve A x = b
        x_numpy = _direct_solve(mtrx_scipy, b_numpy, solver_kind, solver_params)

        x_torch = torch.tensor(x_numpy, dtype=b.dtype, device=b.device)

        ctx.save_for_backward(mtrx.indices(), mtrx.values(), x_torch)
        ctx.mtrx_shape = mtrx_shape
        ctx.b_dtype = b.dtype
        ctx.b_device = b.device

        return x_torch

    @staticmethod
    def backward(ctx, grad_x):
        """
        Given grad_x = ∂L/∂x, compute:
          - grad_b = ∂L/∂b  = y  where Aᵀ y = grad_x
          - grad_A = ∂L/∂A  with entries -(y_i * x_j)
        """
        mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
        mtrx_shape = ctx.mtrx_shape
        solver_kind = ctx.solver_kind
        # Retrieve the entire params dictionary
        solver_params = ctx.solver_params

        grad_A_sparse = None
        grad_b = None

        vals = mtrx_values.detach().cpu().numpy()
        idxs = (mtrx_indices[0].cpu().numpy(), mtrx_indices[1].cpu().numpy())
        mtrx_scipy_T = (
            spy_sprs.coo_matrix((vals, idxs), shape=mtrx_shape)
            .transpose()
            .tocsc()
        )

        # Solve Aᵀ y = grad_x for y
        grad_x_numpy = grad_x.cpu().numpy()
        grad_b_numpy = _direct_solve(
            mtrx_scipy_T, grad_x_numpy, solver_kind, solver_params
        )
        grad_b_torch = torch.tensor(
            grad_b_numpy, dtype=ctx.b_dtype, device=ctx.b_device
        )

        if ctx.needs_input_grad[1]:
            grad_b = grad_b_torch.clone() # FIX: Was grad_b_

        if ctx.needs_input_grad[0]:
            rows, cols = mtrx_indices[0], mtrx_indices[1]
            y_at_rows = grad_b_torch.index_select(0, rows)
            x_at_cols = x_torch.index_select(0, cols)
            grad_vals = -(y_at_rows * x_at_cols)

            grad_A_sparse = torch.sparse_coo_tensor(
                mtrx_indices, grad_vals, mtrx_shape
            )

        # None for solver_kind, None for solver_params (non-tensor inputs)
        return grad_A_sparse, grad_b, None, None


def solve(
    mtrx: torch.Tensor, b: torch.Tensor, solver=None, solver_options: dict = None
) -> torch.Tensor:
    """
    Wrapper for SparseLinearSolve.apply with a linear-solver choice.

    Args:
        mtrx: Sparse torch tensor (ndof, ndof).
        b: Dense torch tensor (ndof,).
        solver: Optional solver enum (e.g., Solvers.PETSC /
                PARDISO / SPSOLVE).
        solver_options: Optional dictionary of solver-specific settings,
                        mirroring the JAX implementation's `params`.
                        E.g.: {"petsc_solver": {"ksp_type": "gmres", "pc_type": "lu"}}

    Returns:
        x: Solution tensor (ndof,).
    """
    # We don't import linear_solvers here; we just look at .name if present.
    solver_name = getattr(solver, "name", None)
    if solver_name == "PETSC":
        solver_kind = "petsc"
    elif solver_name == "PARDISO":
        solver_kind = "pardiso"
    else:
        solver_kind = "spsolve"

    # Pass an empty dict if None, to be handled gracefully downstream
    if solver_options is None:
        solver_options = {}

    return SparseLinearSolve.apply(mtrx, b, solver_kind, solver_options)



