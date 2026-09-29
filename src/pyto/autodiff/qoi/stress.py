"""Torch-differentiable p-norm / max von Mises stress.

compute_pnorm_stress_autograd returns both the p-norm stress (used for the
PNORM_STRESS QOI) and the max von Mises stress (used for the
MAX_VONMISES_STRESS QOI) from the same forward pass -- no separate function
needed for MAX_VONMISES_STRESS, the dispatcher just picks the other
element of the returned pair.
"""
import numpy as np
import torch


def compute_pnorm_stress_autograd(
    sol_t: torch.Tensor,          # (ndof,) torch, requires_grad=True
    x_filtered_t: torch.Tensor,   # (nelems,) torch
    fe_solver,                    # HexStructuralFEA with .mesh and .mat_prop
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute p-norm von Mises stress and max von Mises stress using PyTorch autograd.
    Args:
        sol_t: Torch tensor of shape (ndof,) containing the global displacement/solution vector.
        x_filtered_t: Torch tensor of shape (nelems,) containing the filtered element densities.
        fe_solver: The structural FEA solver object, providing mesh and connectivity
					  (mesh.edofMat) and material properties (mat_prop).
    Returns:
        pnorm_stress_t: Torch scalar tensor containing the p-norm von Mises stress.
        max_vm_t: Torch scalar tensor containing the maximum von Mises stress.
    """
    from pyto.autodiff.material_model import _EVOID_RELATIVE, _PNORM_EXPONENT

    device = sol_t.device
    dtype = sol_t.dtype

    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    # ---------- 1) gradN at element center (same as postprocess, but torch) ----------
    gradN_np = (1.0 / 8.0) * np.array([
        [-1,  1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1, -1, -1,  1,  1],
        [-1, -1, -1, -1,  1,  1,  1,  1],
    ], dtype=np.float64)

    gradN = torch.tensor(gradN_np, dtype=dtype, device=device)  # (3, 8)

    elem_size = torch.as_tensor(
        mesh.elem_size, dtype=dtype, device=device
    )  # (3,)
    gradN = gradN * (2.0 / elem_size.view(3, 1))  # (3,8)

    # ---------- 2) Gather DOFs per element ----------
    edof_np = mesh.edofMat  # (nelems, 24) int
    edof = torch.as_tensor(edof_np, dtype=torch.long, device=device)

    # x, y, z components per node:
    u_e = sol_t[edof[:, 0::3]]  # (nelems, 8)
    v_e = sol_t[edof[:, 1::3]]  # (nelems, 8)
    w_e = sol_t[edof[:, 2::3]]  # (nelems, 8)

    gradN_T = gradN.t()  # (8, 3)

    # ---------- 3) Displacement gradients ----------
    uGrad = u_e @ gradN_T  # (nelems, 3)
    vGrad = v_e @ gradN_T
    wGrad = w_e @ gradN_T

    # ---------- 4) Engineering strains ----------
    eps_xx = uGrad[:, 0]
    eps_yy = vGrad[:, 1]
    eps_zz = wGrad[:, 2]
    gamma_yz = uGrad[:, 1] + vGrad[:, 0]
    gamma_xz = uGrad[:, 2] + wGrad[:, 0]
    gamma_xy = vGrad[:, 2] + wGrad[:, 1]

    strain = torch.stack(
        [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy],
        dim=1,
    )  # (nelems, 6)

    # Thermo-structural solver: stress comes from the ELASTIC strain, total minus the free thermal strain
    # alpha * (T_mean - T_ref) on the normal components (as HexStructuralFEA.postprocess does). T is part of the
    # autograd graph, so the stress gradient also flows through the thermal solve.
    T = getattr(fe_solver, "temperature", None)
    if T is not None and hasattr(fe_solver, "thermal_fea"):
        elem_nodes = torch.as_tensor(mesh.elemArray[:, :8], dtype=torch.long, device=device)
        alpha = float(fe_solver.mat_prop.thermal_expansion_coefficient)
        eps_th = alpha * (T[elem_nodes].mean(dim=1) - fe_solver.thermal_fea.thermoElasticReferenceTemperature)
        strain = strain - torch.cat([eps_th.unsqueeze(1).expand(-1, 3), torch.zeros_like(strain[:, 3:])], dim=1)

    # ---------- 5) Constitutive matrix D (single material) ----------
    mat = fe_solver.mat_prop
    E = float(mat.youngs_modulus)
    nu = float(mat.poissons_ratio)

    D_np = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu,   nu,     nu,     0,                        0,                        0],
        [nu,       1 - nu, nu,     0,                        0,                        0],
        [nu,       nu,     1 - nu, 0,                        0,                        0],
        [0,        0,      0,     (1 - 2 * nu) / 2,          0,                        0],
        [0,        0,      0,      0,                       (1 - 2 * nu) / 2,         0],
        [0,        0,      0,      0,                        0,                       (1 - 2 * nu) / 2],
    ], dtype=np.float64)

    D = torch.tensor(D_np, dtype=dtype, device=device)  # (6,6)

    # ---------- 6) Cauchy stress per element ----------
    stress = strain @ D.t()  # (nelems, 6)

    # ---------- 7) STRESS_RELAXATION correction ----------
    q = 0.5  # same q as in postprocess
    x = x_filtered_t.view(nelems)
    x = torch.clamp(x, min=1e-12)
    correction = _EVOID_RELATIVE + (1.0 - _EVOID_RELATIVE) * x.pow(q)  # (nelems,)
    correction = correction.unsqueeze(1)  # (nelems, 1)

    eStress = correction * stress  # (nelems, 6)

    # ---------- 8) Von Mises per element ----------
    sxx = eStress[:, 0]
    syy = eStress[:, 1]
    szz = eStress[:, 2]
    syz = eStress[:, 3]
    sxz = eStress[:, 4]
    sxy = eStress[:, 5]

    vm_sq = 0.5 * (
        (sxx - syy) ** 2 +
        (syy - szz) ** 2 +
        (szz - sxx) ** 2
    ) + 3.0 * (syz ** 2 + sxz ** 2 + sxy ** 2)

    vm = torch.sqrt(vm_sq + 1e-16)  # (nelems,)

    # max von Mises (also used directly as the MAX_VONMISES_STRESS QOI)
    max_vm_t = vm.max()  # scalar

    # ---------- 9) p-norm of von Mises ----------
    p = float(_PNORM_EXPONENT)
    pnorm_stress_t = vm.pow(p).sum().pow(1.0 / p)  # scalar

    return pnorm_stress_t, max_vm_t
