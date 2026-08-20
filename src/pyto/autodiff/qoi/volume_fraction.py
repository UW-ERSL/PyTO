"""Torch-differentiable volume-fraction constraint."""
import torch


def compute_volume_constraint_torch(
    x: torch.Tensor,
    volfrac_upper: float,
) -> torch.Tensor:
    """Compute the volume-fraction constraint using Torch.

    The constraint is defined as:
        c(x) = mean(x) / volfrac_upper - 1,
    so c(x) <= 0 corresponds to satisfying the volume-fraction upper bound.

    Args:
        x: Array of shape (num_elems,) containing the element densities (or filtered densities).
        volfrac_upper: The target upper bound on the volume fraction.

    Returns:
        The volume-fraction constraint value as a scalar Torch tensor.
    """
    x = x.flatten()
    return x.mean() / volfrac_upper - 1.0
