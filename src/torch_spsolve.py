# torch_spsolve.py
import torch
import enum
import numpy as np
import scipy.sparse as spy_sprs
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu
from typing import Optional

# --- PARDISO availability ---
try:
    import pypardiso
    HAVE_PARDISO = True
    # Prefer the handle-based solver for factorization reuse
    try:
        from pypardiso import PyPardisoSolver
        HAVE_PYPARDISO_SOLVER = True
    except Exception:
        HAVE_PYPARDISO_SOLVER = False
except Exception:
    pypardiso = None
    HAVE_PARDISO = False
    HAVE_PYPARDISO_SOLVER = False

# --- PETSc availability ---
try:
    import petsc4py.PETSc as PETSc
except Exception:
    PETSc = None


class Solvers(enum.Enum):
    SPSOLVE = enum.auto()
    PCG = enum.auto()
    PYAMG = enum.auto()
    PETSC = enum.auto()
    PARDISO = enum.auto()       # general (possibly unsymmetric)
    PARDISO_SPD = enum.auto()   # symmetric positive definite (mtype=2)


def _petsc_solve(A_csr: spy_sprs.csr_matrix, rhs, solver_options: dict = None):
    """Solve A x = rhs using PETSc via petsc4py."""
    if PETSc is None:
        raise ImportError("petsc4py is not available; cannot use PETSC solver.")
    if solver_options is None:
        solver_options = {}

    petsc_opts = solver_options.get("petsc_solver", {})
    ksp_type = petsc_opts.get("ksp_type", "preonly")
    pc_type  = petsc_opts.get("pc_type",  "cholesky")

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
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    if ksp_type == "tfqmr":
        ksp.getPC().setFactorSolverType("mumps")

    x_vec = PETSc.Vec().createSeq(len(rhs))
    ksp.solve(rhs_vec, x_vec)
    return x_vec.getArray()


def _direct_solve(A: spy_sprs.spmatrix, rhs, solver_kind: str, solver_options: dict = None):
    """Generic fallback solve (no factorization reuse across fwd/bwd)."""
    A_csr = A.tocsr()
    if solver_kind == "petsc" and PETSc is not None:
        return _petsc_solve(A_csr, rhs, solver_options)
    elif solver_kind.startswith("pardiso") and HAVE_PARDISO:
        # Simple one-shot; no reuse (we prefer handle-based path elsewhere)
        x = pypardiso.spsolve(A_csr, rhs)
        # DO NOT call pypardiso.ps.free_memory() here; it can hurt reuse in-session.
        return x
    else:
        return spsolve(A_csr, rhs)


class SparseLinearSolve(torch.autograd.Function):
    """
    Supports:
      • SciPy LU path with transpose solve reuse
      • PETSc
      • PARDISO (generic) one-shot
      • PARDISO_SPD with factorization reuse (PyPardisoSolver mtype=2)

    Call signature (unchanged for you):
        forward(K_coo_sparse, b, solver_kind, solver_params)
    """

    @staticmethod
    def forward(ctx, mtrx, b, solver_kind, solver_params: Optional[dict]):
        if solver_params is None:
            solver_params = {}

        # Always coalesce (safe; avoids duplicates)
        mtrx = mtrx.coalesce()
        idx  = mtrx.indices().detach().cpu().numpy()
        vals = mtrx.values().detach().cpu().numpy()
        shape = mtrx.shape
        b_np  = b.detach().cpu().numpy()

        A_csc = spy_sprs.coo_matrix((vals, (idx[0], idx[1])), shape=shape).tocsc()

        ctx.solver_kind  = solver_kind
        ctx.solver_params = solver_params
        ctx.mtrx_shape   = shape
        ctx.b_dtype      = b.dtype
        ctx.b_device     = b.device

        # Defaults
        ctx.lu = None
        ctx.pardiso_solver = None
        ctx.is_spd = (solver_kind == "pardiso_spd")

        if solver_kind == "spsolve":
            # SciPy LU — store factors to reuse for A^T in backward
            lu = splu(A_csc)
            x_np = lu.solve(b_np)
            ctx.lu = lu

        elif solver_kind == "pardiso_spd":
            # --- SPD path with factorization reuse ---
            if not HAVE_PYPARDISO_SOLVER:
                # Fallback to one-shot if handle-based solver not available
                x_np = _direct_solve(A_csc, b_np, "pardiso", solver_params)
            else:
                # mtype=2 => real symmetric positive definite
                solver = PyPardisoSolver(mtype=2)
                # Factorize once (analysis+factorization). Reused in backward.
                solver.factorize(A_csc)
                x_np = solver.solve(b_np)
                ctx.pardiso_solver = solver  # keep handle alive for backward

        elif solver_kind == "pardiso":
            # General PARDISO (no SPD guarantees). One-shot by default.
            if not HAVE_PARDISO:
                x_np = _direct_solve(A_csc, b_np, "spsolve", solver_params)
            else:
                # If handle-based solver is available you could also enable reuse here
                # with mtype=11 (real unsymmetric). For now keep simple:
                x_np = pypardiso.spsolve(A_csc, b_np)

        elif solver_kind == "petsc":
            x_np = _direct_solve(A_csc, b_np, "petsc", solver_params)

        else:
            # Fallback: SciPy
            x_np = _direct_solve(A_csc, b_np, "spsolve", solver_params)

        # Save minimal tensors for backward
        ctx.save_for_backward(mtrx.indices(), mtrx.values(), torch.from_numpy(x_np))
        return torch.tensor(x_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_x):
        mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
        shape        = ctx.mtrx_shape
        solver_kind  = ctx.solver_kind
        solver_params = ctx.solver_params

        grad_b = None
        grad_A_sparse = None

        grad_x_np = grad_x.detach().cpu().numpy()

        if ctx.lu is not None:
            # SciPy LU path: reuse factors to solve A^T y = grad_x
            y_np = ctx.lu.solve(grad_x_np, 'T')

        elif solver_kind == "pardiso_spd" and (ctx.pardiso_solver is not None):
            # SPD ⇒ A is symmetric; we can solve A y = grad_x with SAME factors.
            try:
                y_np = ctx.pardiso_solver.solve(grad_x_np)
            finally:
                # Optional: free PARDISO memory now that both fwd & bwd are done
                try:
                    ctx.pardiso_solver.free_memory()
                except Exception:
                    pass
                ctx.pardiso_solver = None

        elif solver_kind == "pardiso" and HAVE_PYPARDISO_SOLVER:
            # If you later flip to handle-based unsymmetric reuse:
            # solver = PyPardisoSolver(mtype=11)
            # solver.factorize(A_csc) stored in ctx; here do solver.solve(grad_x_np, transpose=True)
            # For now, rebuild A^T and one-shot:
            rows = mtrx_indices[0].cpu().numpy()
            cols = mtrx_indices[1].cpu().numpy()
            vals = mtrx_values.detach().cpu().numpy()
            A_T = spy_sprs.coo_matrix((vals, (rows, cols)), shape=shape).transpose().tocsc()
            y_np = _direct_solve(A_T, grad_x_np, "pardiso", solver_params)

        else:
            # Generic fallback: build A^T and solve with chosen backend
            rows = mtrx_indices[0].cpu().numpy()
            cols = mtrx_indices[1].cpu().numpy()
            vals = mtrx_values.detach().cpu().numpy()
            A_T = spy_sprs.coo_matrix((vals, (rows, cols)), shape=shape).transpose().tocsc()
            y_np = _direct_solve(A_T, grad_x_np, solver_kind if isinstance(solver_kind, str) else "spsolve", solver_params)

        y_t = torch.tensor(y_np, dtype=ctx.b_dtype, device=ctx.b_device)
        if ctx.needs_input_grad[1]:
            grad_b = y_t

        if ctx.needs_input_grad[0]:
            # Compute ∂L/∂A entries: -(y_rows * x_cols)
            rows_t, cols_t = mtrx_indices[0], mtrx_indices[1]
            y_rows = y_t.index_select(0, rows_t)
            x_cols = x_torch.index_select(0, cols_t)
            grad_vals = -(y_rows * x_cols)
            grad_A_sparse = torch.sparse_coo_tensor(mtrx_indices, grad_vals, shape)

        return grad_A_sparse, grad_b, None, None


def solve(mtrx: torch.Tensor,
          b: torch.Tensor,
          solver=None,
          solver_options: dict = None) -> torch.Tensor:
    """
    Wrapper for SparseLinearSolve.apply.

    Use:
        Solvers.SPSOLVE
        Solvers.PETSC
        Solvers.PARDISO
        Solvers.PARDISO_SPD   <-- SPD with factorization reuse
    """
    solver_name = getattr(solver, "name", None)
    if solver_name == "PETSC":
        solver_kind = "petsc"
    elif solver_name == "PARDISO_SPD":
        solver_kind = "pardiso_spd"
    elif solver_name == "PARDISO":
        solver_kind = "pardiso"
    else:
        solver_kind = "spsolve"

    if solver_options is None:
        solver_options = {}

    return SparseLinearSolve.apply(mtrx, b, solver_kind, solver_options)
