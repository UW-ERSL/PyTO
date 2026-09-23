"""Torch-differentiable compliance objective/constraint."""
import torch


def compute_compliance_torch(
    sol: torch.Tensor,
    x: torch.Tensor,
    fe_solver,
    KE: torch.Tensor,
    material_model=None,
) -> torch.Tensor:
    """Compute the compliance objective using Torch given an existing solution and density field.

    Args:
        sol: Array of shape (n_dof,) containing the global displacement/solution vector.
        x: Array of shape (num_elems,) containing the element densities (or pseudo-densities).
        fe_solver: The structural FEA solver object, providing mesh and connectivity (mesh.edofMat).
        KE: Element stiffness matrix of shape (nRows, nRows) used for all elements.
        material_model: Material model object used to compute the element-wise material scaling.

    Returns:
        The compliance value as a scalar Torch tensor.
    """
    from pyto.autodiff.material_model import (
        get_structural_material_model_scaling_torch,
        get_thermal_material_model_scaling_torch,
    )
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.physics.thermal.tet_thermal_fea import TetThermalFEA

    # The objective must use the same material interpolation as the solver's
    # own state equation (thermal uses its own penalty/void constants), or its
    # gradient no longer matches the physics being solved.
    if isinstance(fe_solver, (HexThermalFEA, TetThermalFEA)):
        scaling_fn = get_thermal_material_model_scaling_torch
    else:
        scaling_fn = get_structural_material_model_scaling_torch

    x = x.flatten()  # (E,)
    dofMat = torch.as_tensor(
        fe_solver.mesh.edofMat,
        dtype=torch.long,
        device=sol.device,
    )  # (E, nRows)

    # Element DOFs: (E, nRows)
    u_e = sol[dofMat]

    # Element energies: ce_e = u_eᵀ KE u_e
    # u_e: (E, nRows), KE: (nRows, nRows) → ce: (E,)
    ce = torch.einsum("ei,ij,ej->e", u_e, KE, u_e)
    material_scaling = scaling_fn(x, material_model)  # (E,)

    return (material_scaling * ce).sum()
