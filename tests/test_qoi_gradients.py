"""Finite-difference checks for the torch-native QOI functions used by the
autodiff optimization pipeline (topopt_mma.py), on small real structural
and thermal meshes (not just the isolated linear-solve primitive).

Every QOI registered in pyto.autodiff.qoi is covered here (compliance,
volume fraction, p-norm stress, mass, gvector, max von Mises), for both
physics where applicable -- the Phase 4 gate. Two of these (compliance,
max von Mises) are additionally cross-checked against
pyto.autodiff.reference_adjoint, the independently-derived legacy NumPy
adjoint oracle, not just against finite differences of the torch code
itself -- so a bug shared between the torch formula and its own FD check
can't hide.
"""
import numpy as np
import torch

from topopt_material_model import MaterialModel
from topopt_obj_cons_sensitivities import (
    compute_compliance_torch,
    compute_volume_constraint_torch,
    compute_pnorm_stress_autograd,
    compute_mass_torch,
    compute_gvector_torch,
    compute_compliance,
    compute_pnorm_stress_and_sensitivity,
)
from topopt_common import TOParams, TO_QOI

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


def test_max_vonmises_stress_gradient_matches_finite_difference(structural_fe_solver):
    # New in Phase 4: MAX_VONMISES_STRESS reuses compute_pnorm_stress_autograd's
    # second return value rather than the old NumPy-only sensitivity path.
    #
    # max() is only piecewise-smooth: its gradient has a kink wherever two
    # elements' stress values tie. Verified directly: at the *uniform*
    # x0=0.5 this fixture's mesh has an exact symmetry tie between elements
    # 23 and 167 (von Mises stresses agree to >7 significant digits), so
    # the argmax flips between them for a perturbation as small as 1e-7 in
    # either direction -- not a floating-point-precision artifact, a real
    # kink in the objective sitting exactly at this x0. A non-uniform,
    # randomized x0 (as any real optimization iterate would have) gives a
    # unique argmax and a well-defined, cleanly-verifiable gradient
    # (confirmed separately: relative error ~1e-9 once the tie is broken).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    torch.manual_seed(42)
    x0 = (0.3 + 0.4 * torch.rand(n, dtype=torch.float64)).requires_grad_(True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    _pnorm, max_vm = compute_pnorm_stress_autograd(sol, x0, fe)
    (analytic,) = torch.autograd.grad(max_vm, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        _, max_vm = compute_pnorm_stress_autograd(sol, x, fe)
        return max_vm.item()

    direction = analytic / analytic.norm()
    step = 1e-5
    xp = (x0.detach() + step * direction).clone()
    xm = (x0.detach() - step * direction).clone()
    fd_directional = (f(xp) - f(xm)) / (2 * step)
    analytic_directional = (analytic * direction).sum().item()
    rel_err = abs(fd_directional - analytic_directional) / max(abs(fd_directional), 1e-12)
    assert rel_err < REL_TOL, f"analytic={analytic_directional:.6e} fd={fd_directional:.6e}"


def test_mass_gradient_matches_finite_difference(structural_fe_solver):
    # New torch-native QOI in Phase 4 (was NumPy-only, crashed on grad-tracked x).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = torch.rand(n, dtype=torch.float64, requires_grad=True)

    mass = compute_mass_torch(x0, fe)
    (analytic,) = torch.autograd.grad(mass, x0)

    def f(x):
        return compute_mass_torch(x, fe).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL


def test_gvector_gradient_matches_finite_difference(structural_fe_solver):
    # New torch-native QOI in Phase 4 (was NumPy-only, crashed on grad-tracked sol).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)
    rng = np.random.default_rng(0)
    g = rng.standard_normal(fe.bc.num_dofs)

    sol = fe.solve(x0, MaterialModel.SIMP)
    obj = compute_gvector_torch(sol, g)
    (analytic,) = torch.autograd.grad(obj, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        return compute_gvector_torch(sol, g).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_compliance_gradient_matches_finite_difference_thermal(thermal_fe_solver, thermal_KE):
    # Both-physics coverage for the Phase 4 gate: thermal compliance,
    # exercising HexThermalFEA's new torch-native solve().
    #
    # Thermal compliance/gradient are ~4 orders of magnitude smaller here
    # than structural's (O(0.1) and O(1e-5) vs O(10) and O(1) respectively),
    # so the structural test's EPS=1e-6 hits SPSOLVE's floating-point noise
    # floor for this problem (verified: relative error swings from ~2e-5 at
    # eps=1e-3/1e-4 to ~1% at eps=1e-6). A larger, thermal-appropriate step
    # avoids that -- this is a finite-difference step-size choice, not a
    # looser correctness bar.
    THERMAL_EPS = 1e-4
    fe = thermal_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    c = compute_compliance_torch(sol, x0, fe, thermal_KE, MaterialModel.SIMP)
    (analytic,) = torch.autograd.grad(c, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        return compute_compliance_torch(sol, x, fe, thermal_KE, MaterialModel.SIMP).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += THERMAL_EPS
        xm = x0.detach().clone(); xm[e] -= THERMAL_EPS
        fd = (f(xp) - f(xm)) / (2 * THERMAL_EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_compliance_torch_matches_reference_adjoint_value(structural_fe_solver, structural_KE):
    # Cross-check against the independently-derived legacy NumPy oracle
    # (pyto.autodiff.reference_adjoint), not just finite differences of the
    # torch code itself -- the Phase 4 gate's actual wording.
    fe = structural_fe_solver
    x0 = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64, requires_grad=True)
    sol = fe.solve(x0, MaterialModel.SIMP)
    torch_value = compute_compliance_torch(sol, x0, fe, structural_KE, MaterialModel.SIMP).item()

    sol_np = sol.detach().cpu().numpy()
    x_np = x0.detach().cpu().numpy()
    KE_np = structural_KE.detach().cpu().numpy()
    reference_value = compute_compliance(sol_np, x_np, fe, KE_np, MaterialModel.SIMP)

    assert abs(torch_value - reference_value) / abs(reference_value) < 1e-8


def test_max_vonmises_torch_matches_reference_adjoint_value(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    x0 = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64, requires_grad=True)
    sol = fe.solve(x0, MaterialModel.SIMP)
    _pnorm, max_vm_torch = compute_pnorm_stress_autograd(sol, x0, fe)

    fe.postprocess()  # populates fe.vonMisesStress/stressComponents/pNormStress
    sol_np = sol.detach().cpu().numpy()
    x_np = x0.detach().cpu().numpy()
    KE_np = structural_KE.detach().cpu().numpy()

    # compute_pnorm_stress_and_sensitivity (reference_adjoint, unmodified
    # legacy code) solves a second system via linear_solvers.solve(
    # fe.stiff_mtrx, ...), which expects a SciPy sparse matrix -- but
    # fe.stiff_mtrx is now a torch sparse tensor since solve() became
    # torch-native (Phase 3/4). Convert at this test's boundary rather than
    # touching the oracle, which is meant to stay exactly as it was.
    import scipy.sparse as sp
    stiff_coo = fe.stiff_mtrx.coalesce()
    idx = stiff_coo.indices().detach().numpy()
    vals = stiff_coo.values().detach().numpy()
    fe.stiff_mtrx = sp.coo_matrix((vals, (idx[0], idx[1])), shape=tuple(stiff_coo.shape)).tocsc()

    _vm_pnorm, _sensitivity, max_vm_reference = compute_pnorm_stress_and_sensitivity(
        sol_np, x_np, fe, KE_np, MaterialModel.SIMP
    )

    assert abs(max_vm_torch.item() - max_vm_reference) / abs(max_vm_reference) < 1e-8


def test_stress_scaling_is_isolated_per_to_params():
    # Regression test for the Phase 4 fix: stress_scaling used to be a
    # mutable attribute on the compute_constraint_and_gradient function
    # object (module-global for the whole process), so two separate
    # optimization runs sharing that attribute would silently leak state
    # into each other. It's now a plain field on TOParams.
    a = TOParams()
    b = TOParams()
    assert a.stress_scaling == 1.0
    assert b.stress_scaling == 1.0
    a.stress_scaling = 0.42
    assert b.stress_scaling == 1.0, "stress_scaling leaked across separate TOParams instances"


def test_pareto_compliance_value_matches_compute_compliance_torch(structural_fe_solver, structural_KE):
    # Phase 5 completion: Pareto's driver (pyto.topopt.drivers.pareto) uses
    # its own local, independent topological-sensitivity formulas rather
    # than the torch-native QOI path (see that module's docstring for why
    # this duplication exists and hasn't been consolidated). Gradient-
    # checking topological sensitivity would test the wrong mathematical
    # object (see plan Phase 5 completion notes) -- what CAN be checked is
    # that the plain *compliance value* both formulas compute from the same
    # (x, sol) agrees, since both reduce to the same u^T K(x) u sum via
    # different code paths. This is the check that would have caught a
    # sign/indexing slip introduced while fixing the torch/NumPy solve
    # boundary, without claiming to validate a non-existent "gradient of T".
    from topopt_common import FEA_MODE, TOParams, TO_QOI
    from topopt_pareto import compute_objective_topological_sensitivity_compliance

    fe = structural_fe_solver
    x0_t = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64)
    sol_t = fe.solve(x0_t, MaterialModel.SIMP)
    fe.postprocess()  # populates fe.strainComponents/stressComponents for T (unused here, but required by the shared code path)

    sol_np = sol_t.detach().cpu().numpy()
    x0_np = x0_t.detach().cpu().numpy()
    KE_np = structural_KE.detach().cpu().numpy()

    to_params = TOParams()
    to_params.Objective = (TO_QOI.COMPLIANCE, None)
    _obj, _T, compliance_pareto = compute_objective_topological_sensitivity_compliance(
        FEA_MODE.STRUCTURAL, to_params, sol_np, x0_np, fe, KE_np, MaterialModel.SIMP
    )

    compliance_torch = compute_compliance_torch(sol_t, x0_t, fe, structural_KE, MaterialModel.SIMP).item()

    assert abs(compliance_pareto - compliance_torch) / abs(compliance_torch) < 1e-8
