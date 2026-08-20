"""Torch-differentiable mass objective/constraint.

New in Phase 4: previously computed with plain NumPy (np.sum(...)), which
crashed whenever x carried a grad-tracked torch tensor from the autodiff
optimization chain in topopt_mma.py -- see plan Phase 4. Single-material
only, matching the pre-existing NumPy version's assumption
(fe_solver.mat_prop.mass_density).
"""
import torch


def compute_mass_torch(x: torch.Tensor, fe_solver) -> torch.Tensor:
    """Compute the total mass using Torch.

    Args:
        x: Array of shape (num_elems,) containing the element densities.
        fe_solver: The FEA solver object, providing mesh.elem_size and mat_prop.mass_density.

    Returns:
        The total mass as a scalar Torch tensor.
    """
    elem_volume = (
        fe_solver.mesh.elem_size[0]
        * fe_solver.mesh.elem_size[1]
        * fe_solver.mesh.elem_size[2]
    )
    return (x.flatten() * elem_volume * fe_solver.mat_prop.mass_density).sum()
