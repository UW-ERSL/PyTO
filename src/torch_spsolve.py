import torch
import enum
import numpy as np
import scipy.sparse as spy_sprs
from scipy.sparse.linalg import spsolve
from typing import Dict
from scipy.sparse.linalg import splu

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
    elif solver_kind == "pardiso" and pypardiso is not None:
        x = pypardiso.spsolve(A_csr, rhs)
        pypardiso.ps.free_memory()
    else:
        x = spsolve(A_csr, rhs)
    return x


class SparseLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtrx, b, solver_kind, solver_params):
        mtrx = mtrx.coalesce()
        idx = mtrx.indices().detach().cpu().numpy()
        vals = mtrx.values().detach().cpu().numpy()
        shape = mtrx.shape
        b_np = b.detach().cpu().numpy()

        A_csc = spy_sprs.coo_matrix(
            (vals, (idx[0], idx[1])), shape=shape
        ).tocsc()

        ctx.solver_kind = solver_kind
        ctx.solver_params = solver_params
        ctx.mtrx_shape = shape
        ctx.b_dtype = b.dtype
        ctx.b_device = b.device

        if solver_kind == "spsolve":
            # Factor once
            lu = splu(A_csc)    # LU factorization
            x_np = lu.solve(b_np)
            ctx.lu = lu
            ctx.A_T = None
        else:
            # fall back to existing _direct_solve for petsc/pardiso
            x_np = _direct_solve(A_csc, b_np, solver_kind, solver_params)
            ctx.lu = None
            ctx.A_T = A_csc.T.tocsc()  # store for re-use in backward if wanted

        # Save minimal tensors for backward
        ctx.save_for_backward(mtrx.indices(), mtrx.values(), torch.from_numpy(x_np))

        return torch.tensor(x_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_x):
        mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
        shape = ctx.mtrx_shape
        solver_kind = ctx.solver_kind
        solver_params = ctx.solver_params

        grad_A_sparse = None
        grad_b = None

        grad_x_np = grad_x.cpu().numpy()

        if solver_kind == "spsolve" and ctx.lu is not None:
            # Use same LU factors: solve A^T y = grad_x
            y_np = ctx.lu.solve(grad_x_np, 'T')
        else:
            vals = mtrx_values.detach().cpu().numpy()
            idxs = (mtrx_indices[0].cpu().numpy(), mtrx_indices[1].cpu().numpy())
            A_T = spy_sprs.coo_matrix((vals, idxs), shape=shape).transpose().tocsc()
            y_np = _direct_solve(A_T, grad_x_np, solver_kind, solver_params)

        y_t = torch.tensor(y_np, dtype=ctx.b_dtype, device=ctx.b_device)

        if ctx.needs_input_grad[1]:
            grad_b = y_t

        if ctx.needs_input_grad[0]:
            rows, cols = mtrx_indices[0], mtrx_indices[1]
            y_rows = y_t.index_select(0, rows)
            x_cols = x_torch.index_select(0, cols)
            grad_vals = -(y_rows * x_cols)
            grad_A_sparse = torch.sparse_coo_tensor(mtrx_indices, grad_vals, shape)

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



