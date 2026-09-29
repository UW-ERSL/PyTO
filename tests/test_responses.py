"""Response library (plan S1): values against independent quantities, gradients against finite differences."""
import numpy as np
import pytest
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import responses as R


def _x0(fe, seed=0):
    return torch.tensor(0.3 + 0.6 * np.random.default_rng(seed).random(fe.mesh.num_elems), dtype=torch.float64)


def _check_gradient(fe, name, n_check=3, eps=1e-6, **params):
    x0 = _x0(fe)

    def value(x):
        return R.evaluate(name, fe.solve(x, MaterialModel.SIMP), x, fe, **params)

    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(value(x), x)
    for e in np.random.default_rng(1).choice(fe.mesh.num_elems, size=n_check, replace=False):
        up, dn = x0.clone(), x0.clone()
        up[e] += eps
        dn[e] -= eps
        fd = (value(up) - value(dn)).item() / (2 * eps)
        assert abs(fd - g[e].item()) <= 1e-4 * abs(fd) + 1e-9 * (abs(value(x0).item()) + 1e-30), (name, params, e, fd, g[e].item())


def _loaded_and_fixed_nodes(fe):
    loaded = np.unique(np.nonzero(fe.bc.force)[0] // 3)
    fixed = np.unique(np.asarray(fe.bc.fixed_dofs) // 3)
    return loaded, fixed


def test_structural_response_values(structural_fe_solver):
    fe = structural_fe_solver
    loaded, fixed = _loaded_and_fixed_nodes(fe)
    x = _x0(fe)
    u = fe.solve(x, MaterialModel.SIMP)
    n = int(loaded[0])
    assert R.evaluate("Displacement", u, x, fe, nodes=[n], component="y").item() == pytest.approx(u[3 * n + 1].item())
    mag = R.evaluate("Displacement", u, x, fe, nodes=[n], component="magnitude").item()
    assert mag == pytest.approx(torch.linalg.norm(u[3 * n:3 * n + 3]).item())
    # pnorm is an upper bound of the maximum and close to it for a large p
    uy = u.reshape(-1, 3)[:, 1].abs()
    pn = R.evaluate("Displacement", u, x, fe, component="y", aggregate_by="pnorm", p=16).item()
    assert uy.max().item() <= pn <= uy.max().item() * len(uy) ** (1 / 16) * 1.0001
    # equilibrium: the support reactions balance the applied load in every direction
    for c, comp in enumerate("xyz"):
        applied = float(np.asarray(fe.bc.force).reshape(-1, 3)[:, c].sum())
        react = R.evaluate("ReactionForce", u, x, fe, nodes=fixed, component=comp).item()
        assert react == pytest.approx(-applied, abs=1e-6 * max(1.0, np.abs(fe.bc.force).sum()))
    # strain energy of the whole model is half the compliance
    assert R.evaluate("StrainEnergy", u, x, fe).item() == pytest.approx(0.5 * R.evaluate("Compliance", u, x, fe).item())
    assert R.evaluate("VolumeFraction", u, x, fe).item() == pytest.approx(x.mean().item())
    assert R.evaluate("VolumeFraction", u, x, fe, elements=[0, 1]).item() == pytest.approx(x[:2].mean().item())


@pytest.mark.parametrize("name, params", [
    ("Compliance", {}),
    ("Displacement", {"component": "y", "aggregate_by": "mean"}),
    ("Displacement", {"component": "magnitude", "aggregate_by": "pnorm"}),
    ("ReactionForce", {"component": "y"}),
    ("StrainEnergy", {}),
    ("StressPNorm", {}),
    ("StressFailureFactor", {}),
    ("Mass", {}),
])
def test_structural_response_gradients(structural_fe_solver, name, params):
    fe = structural_fe_solver
    loaded, fixed = _loaded_and_fixed_nodes(fe)
    if name == "Displacement":
        params = {**params, "nodes": loaded}
    if name == "ReactionForce":
        # half of the supports: the sum over ALL supports balances the load exactly, so it does not depend on x
        params = {**params, "nodes": fixed[: len(fixed) // 2]}
    _check_gradient(fe, name, **params)


def test_thermal_responses(thermal_fe_solver):
    fe = thermal_fe_solver
    x = _x0(fe)
    T = fe.solve(x, MaterialModel.SIMP)
    assert R.evaluate("Temperature", T, x, fe).item() == pytest.approx(T.mean().item())
    assert R.evaluate("Temperature", T, x, fe, nodes=[3], aggregate_by="sum").item() == pytest.approx(T[3].item())
    # step 1e-4: prescribed temperatures make the values cancel heavily, and smaller steps drown in round-off
    # (checked: FD converges to the autograd value from eps 1e-3 down to 1e-4, then diverges)
    _check_gradient(fe, "Temperature", eps=1e-4, aggregate_by="pnorm")
    _check_gradient(fe, "Compliance", eps=1e-4)


def test_coupled_responses():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_thermoelastic import _coupled_problem
    fe = _coupled_problem()
    x = _x0(fe)
    u = fe.solve(x, MaterialModel.SIMP)
    assert R.physics_of(fe) == R.COUPLED
    assert R.evaluate("Temperature", u, x, fe).item() == pytest.approx(fe.temperature.mean().item())
    _check_gradient(fe, "Temperature", aggregate_by="mean")          # through the thermal solve only
    _check_gradient(fe, "Displacement", component="magnitude", aggregate_by="pnorm")   # through both solves


def test_physics_availability_and_errors(structural_fe_solver, thermal_fe_solver):
    assert "Temperature" not in R.available_responses(R.STRUCTURAL)
    assert "StressPNorm" not in R.available_responses(R.COUPLED)       # stress ignores thermal strain in the torch path
    assert set(R.available_responses(R.THERMAL)) == {"Compliance", "VolumeFraction", "Temperature"}
    fe = thermal_fe_solver
    x = _x0(fe)
    with pytest.raises(ValueError, match="not available for thermal"):
        R.evaluate("Displacement", fe.solve(x, MaterialModel.SIMP), x, fe)
    with pytest.raises(KeyError, match="Unknown response"):
        R.evaluate("Nope", None, x, fe)
