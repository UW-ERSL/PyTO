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
