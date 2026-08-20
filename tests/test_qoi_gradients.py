"""Finite-difference checks for the torch-native QOI functions used by the
autodiff optimization pipeline (topopt_mma.py), on a small real structural
mesh (not just the isolated linear-solve primitive).

Only the QOIs that are actually wired into topopt_mma's torch.autograd.grad
chain today are checked here: compliance, volume fraction, and p-norm
stress. MASS/GVECTOR/MAX_VONMISES_STRESS are not yet torch-native (see
plan Phase 4) and are intentionally not covered here.
"""
import torch

from topopt_material_model import MaterialModel
from topopt_obj_cons_sensitivities import (
    compute_compliance_torch,
    compute_volume_constraint_torch,
    compute_pnorm_stress_autograd,
)

EPS = 1e-6
REL_TOL = 1e-4


def _central_diff_grad(f_of_x, x0, eps=EPS):
    grad = torch.zeros_like(x0)
    for e in range(x0.numel()):
        xp = x0.clone(); xp[e] += eps
        xm = x0.clone(); xm[e] -= eps
        grad[e] = (f_of_x(xp) - f_of_x(xm)) / (2 * eps)
    return grad


def _sample_elements(n, k=6):
    step = max(1, n // k)
    return list(range(0, n, step))[:k]


def test_compliance_gradient_matches_finite_difference(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64)

    def compliance(x):
        x = x.clone().requires_grad_(True)
        sol = fe.solve(x, MaterialModel.SIMP)
        return compute_compliance_torch(sol, x, fe, structural_KE, MaterialModel.SIMP)

    x0.requires_grad_(True)
    c = compliance(x0)
    (analytic,) = torch.autograd.grad(c, x0)

    for e in _sample_elements(n):
        def f(x):
            with torch.no_grad():
                sol = fe.solve(x, MaterialModel.SIMP)
            return compute_compliance_torch(sol, x, fe, structural_KE, MaterialModel.SIMP).item()
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_volume_fraction_gradient_matches_finite_difference():
    n = 20
    x0 = torch.rand(n, dtype=torch.float64, requires_grad=True)
    volfrac_upper = 0.4

    c = compute_volume_constraint_torch(x0, volfrac_upper)
    (analytic,) = torch.autograd.grad(c, x0)

    def f(x):
        return compute_volume_constraint_torch(x, volfrac_upper).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL


def test_pnorm_stress_gradient_matches_finite_difference(structural_fe_solver):
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    pnorm, _max_vm = compute_pnorm_stress_autograd(sol, x0, fe)
    (analytic,) = torch.autograd.grad(pnorm, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        pnorm, _ = compute_pnorm_stress_autograd(sol, x, fe)
        return pnorm.item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"
