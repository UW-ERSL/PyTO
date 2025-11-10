# torch_spsolve.py (for example)

import torch
import scipy.sparse as spy_sprs
from scipy.sparse.linalg import spsolve


class SparseLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtrx: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mtrx = mtrx.coalesce()

        mtrx_indices = mtrx.indices().detach().cpu().numpy()
        mtrx_values  = mtrx.values().detach().cpu().numpy()
        mtrx_shape   = mtrx.shape
        b_numpy      = b.detach().cpu().numpy()

        mtrx_scipy = spy_sprs.coo_matrix(
            (mtrx_values, (mtrx_indices[0], mtrx_indices[1])),
            shape=mtrx_shape,
        ).tocsc()

        x_numpy = spsolve(mtrx_scipy, b_numpy)
        x_torch = torch.tensor(x_numpy, dtype=b.dtype, device=b.device)

        ctx.save_for_backward(mtrx.indices(), mtrx.values(), x_torch)
        ctx.mtrx_shape = mtrx_shape
        ctx.b_dtype    = b.dtype
        ctx.b_device   = b.device

        return x_torch


    @staticmethod
    def backward(ctx, grad_x):
        mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
        mtrx_shape = ctx.mtrx_shape

        grad_A_sparse = None
        grad_b        = None

        vals = mtrx_values.detach().cpu().numpy()
        idxs = (mtrx_indices[0].cpu().numpy(), mtrx_indices[1].cpu().numpy())
        mtrx_scipy_T = (
            spy_sprs.coo_matrix((vals, idxs), shape=mtrx_shape)
            .transpose()
            .tocsc()
        )

        grad_x_numpy = grad_x.cpu().numpy()
        grad_b_numpy = spsolve(mtrx_scipy_T, grad_x_numpy)
        grad_b_torch = torch.tensor(
            grad_b_numpy, dtype=ctx.b_dtype, device=ctx.b_device
        )

        if ctx.needs_input_grad[1]:
            grad_b = grad_b_torch

        if ctx.needs_input_grad[0]:
            rows, cols = mtrx_indices[0], mtrx_indices[1]
            y_at_rows  = grad_b_torch.index_select(0, rows)
            x_at_cols  = x_torch.index_select(0, cols)
            grad_vals  = -(y_at_rows * x_at_cols)

            grad_A_sparse = torch.sparse_coo_tensor(
                mtrx_indices, grad_vals, mtrx_shape
            )

        return grad_A_sparse, grad_b


solve = SparseLinearSolve.apply
