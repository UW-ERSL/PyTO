"""Thermo-mechanical coupling: differentiable thermal force, coupled solve, gradients."""
import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel


def _solved_thermal(thermal_fe_solver, seed=0):
    fe = thermal_fe_solver
    rng = np.random.default_rng(seed)
    x = torch.tensor(0.3 + 0.6 * rng.random(fe.mesh.num_elems), dtype=torch.float64)
    T = fe.solve(x, MaterialModel.SIMP)
    return fe, x, T


def test_torch_thermoelastic_force_matches_numpy_reference(thermal_fe_solver):
    # The torch force must reproduce the existing NumPy get_thermoelastic_force to round-off.
    fe, x, T = _solved_thermal(thermal_fe_solver)
    f_np = fe.get_thermoelastic_force(x.numpy(), MaterialModel.SIMP)
    f_t = fe.get_thermoelastic_force_torch(T, x, MaterialModel.SIMP).detach().numpy()
    assert np.max(np.abs(f_t)) > 0
    assert np.allclose(f_t, f_np, rtol=1e-10, atol=1e-10 * np.max(np.abs(f_np)))


def test_torch_thermoelastic_force_gradients_wrt_temperature_and_density(thermal_fe_solver):
    # Autograd must flow through both T and x (checked against central finite differences).
    fe, x, T = _solved_thermal(thermal_fe_solver)
    T = T.detach().clone()
    w = torch.tensor(np.random.default_rng(1).standard_normal(3 * fe.mesh.num_nodes), dtype=torch.float64)

    def scalar(T_, x_):
        return (w * fe.get_thermoelastic_force_torch(T_, x_, MaterialModel.SIMP)).sum()

    T_req, x_req = T.clone().requires_grad_(True), x.clone().requires_grad_(True)
    gT, gx = torch.autograd.grad(scalar(T_req, x_req), (T_req, x_req))
    rng = np.random.default_rng(2)
    for name, base, g in (("T", T, gT), ("x", x, gx)):
        for i in rng.choice(base.numel(), size=3, replace=False):
            step = 1e-3 * max(1.0, abs(float(base[i])))
            up, dn = base.clone(), base.clone()
            up[i] += step; dn[i] -= step
            fd = (scalar(up, x) - scalar(dn, x)).item() / (2 * step) if name == "T" else \
                 (scalar(T, up) - scalar(T, dn)).item() / (2 * step)
            assert abs(fd - float(g[i])) <= 1e-5 * max(abs(fd), 1e-12) + 1e-12, (name, i, fd, float(g[i]))


# ---------------------------------------------------------------- B2: structural solve with a thermal load
def _uniform_temperature_problem(supports, n=(4, 2, 2), dT=40.0, T0=23.0):
    """Small brick, uniform temperature T0 + dT (thermal Dirichlet everywhere), structural supports chosen by
    `supports(mesh) -> list of dof indices`."""
    import pyto.core.bc as bound_cond
    import pyto.core.mat_lib as mat_lib
    import pyto.core.hex_mesher as hex_mesher
    import pyto.solve.numpy_backend as lin
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA

    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=n, elem_size=(0.5, 0.5, 0.5))
    mat = mat_lib.get_material("Steel")
    all_nodes = np.arange(mesh.num_nodes)
    tbc = bound_cond.BC(force=np.zeros(mesh.num_nodes), fixed_dofs=all_nodes, dirichlet_values=np.full(mesh.num_nodes, T0 + dT))
    thermal = HexThermalFEA(mesh=mesh, mat_prop=mat, bc=tbc, solver=lin.Solvers.SPSOLVE, thermoElasticReferenceTemperature=T0)
    fixed = np.array(sorted(set(supports(mesh))))
    sbc = bound_cond.BC(force=np.zeros(3 * mesh.num_nodes), fixed_dofs=fixed, dirichlet_values=np.zeros(len(fixed)))
    structural = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=sbc, solver=lin.Solvers.SPSOLVE)  # sets mesh.edofMat structural
    x = torch.ones(mesh.num_elems, dtype=torch.float64)
    T = thermal.solve(x, MaterialModel.SIMP)
    f_th = thermal.get_thermoelastic_force_torch(T, x, MaterialModel.SIMP)
    u = structural.solve(x, MaterialModel.SIMP, extra_force=f_th, elem_thermal_strain=thermal.get_element_thermal_strain(T))
    return mesh, mat, structural, u.detach().numpy(), dT


def test_free_thermal_expansion_gives_alpha_dT_times_position():
    # Minimal supports (6 rigid-body modes removed, nothing else): a uniformly heated body expands freely,
    # u = alpha * dT * (X - X_origin), and carries no stress.
    def supports(mesh):
        xyz = mesh.node_xyz
        origin = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        on_x = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[:, 0].max(), xyz[origin, 1], xyz[origin, 2]]), axis=1)))
        on_y = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[origin, 0], xyz[:, 1].max(), xyz[origin, 2]]), axis=1)))
        return [3 * origin, 3 * origin + 1, 3 * origin + 2, 3 * on_x + 1, 3 * on_x + 2, 3 * on_y + 2]
    mesh, mat, structural, u, dT = _uniform_temperature_problem(supports)
    expected = mat.thermal_expansion_coefficient * dT * (mesh.node_xyz - mesh.node_xyz.min(0))
    assert np.allclose(u.reshape(-1, 3), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())
    structural.postprocess()
    assert np.max(np.abs(structural.vonMisesStress)) < 1e-6 * mat.youngs_modulus * mat.thermal_expansion_coefficient * dT


def test_bar_clamped_at_both_ends_carries_minus_E_alpha_dT():
    # ux fixed on both end faces, lateral directions free: uniaxial stress sigma_xx = -E * alpha * dT.
    def supports(mesh):
        xyz = mesh.node_xyz
        ends = np.where((xyz[:, 0] == xyz[:, 0].min()) | (xyz[:, 0] == xyz[:, 0].max()))[0]
        origin = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        corner = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[origin, 0], xyz[:, 1].max(), xyz[origin, 2]]), axis=1)))
        return [3 * n for n in ends] + [3 * origin + 1, 3 * origin + 2, 3 * corner + 2]
    mesh, mat, structural, u, dT = _uniform_temperature_problem(supports)
    structural.postprocess()
    expected = mat.youngs_modulus * mat.thermal_expansion_coefficient * dT
    sxx = np.asarray(structural.stressComponents)[:, 0]
    assert np.allclose(sxx, -expected, rtol=1e-5), (sxx.min(), sxx.max(), -expected)
