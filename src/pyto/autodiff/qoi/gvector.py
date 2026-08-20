"""Torch-differentiable g'*u ("GVECTOR") objective.

New in Phase 4: previously computed with plain NumPy (np.dot(sol, g)),
which crashed whenever sol carried a grad-tracked torch tensor from the
autodiff optimization chain in topopt_mma.py -- see plan Phase 4.
"""
import torch


def compute_gvector_torch(sol: torch.Tensor, g) -> torch.Tensor:
    """Compute the objective g'*sol using Torch.

    Args:
        sol: Array of shape (n_dof,) containing the global displacement/solution vector.
        g: Array of shape (n_dof,) -- the fixed vector to dot with the solution.

    Returns:
        The dot product as a scalar Torch tensor.
    """
    g_t = torch.as_tensor(g, dtype=sol.dtype, device=sol.device)
    return torch.dot(sol, g_t)
