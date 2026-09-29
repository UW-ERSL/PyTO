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
    assert "StressPNorm" in R.available_responses(R.COUPLED)            # thermal strain subtracted in the torch path
    assert set(R.available_responses(R.THERMAL)) == {"Compliance", "VolumeFraction", "Temperature", "ThermalCompliance"}
    fe = thermal_fe_solver
    x = _x0(fe)
    with pytest.raises(ValueError, match="not available for thermal"):
        R.evaluate("Displacement", fe.solve(x, MaterialModel.SIMP), x, fe)
    with pytest.raises(KeyError, match="Unknown response"):
        R.evaluate("Nope", None, x, fe)


# ---------------------------------------------------------------- thermo-mechanical responses
def _heated_bar(clamped: bool, dT=40.0, T0=23.0):
    """Brick at uniform T0 + dT (thermal Dirichlet on every node), ThermoStructuralFEA. clamped: ux = 0 on both end
    faces (uniaxial thermal stress -E alpha dT); otherwise 6 minimal supports (free expansion, no stress)."""
    import pyto.core.bc as bound_cond
    import pyto.core.hex_mesher as hex_mesher
    import pyto.core.mat_lib as mat_lib
    import pyto.solve.numpy_backend as lin
    from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(4, 2, 2), elem_size=(0.5, 0.5, 0.5))
    mat = mat_lib.get_material("Steel")
    xyz = mesh.node_xyz
    n = np.arange(mesh.num_nodes)
    tbc = bound_cond.BC(force=np.zeros(mesh.num_nodes), fixed_dofs=n, dirichlet_values=np.full(mesh.num_nodes, T0 + dT))
    if clamped:
        ends = n[(xyz[:, 0] < 1e-9) | (xyz[:, 0] > xyz[:, 0].max() - 1e-9)]
        o = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        oy = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[o, 0], xyz[:, 1].max(), xyz[o, 2]]), axis=1)))
        fixed = sorted(set((3 * ends).tolist() + [3 * o + 1, 3 * o + 2, 3 * oy + 2]))
    else:
        o = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        ox = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[:, 0].max(), xyz[o, 1], xyz[o, 2]]), axis=1)))
        oy = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[o, 0], xyz[:, 1].max(), xyz[o, 2]]), axis=1)))
        fixed = [3 * o, 3 * o + 1, 3 * o + 2, 3 * ox + 1, 3 * ox + 2, 3 * oy + 2]
    sbc = bound_cond.BC(force=np.zeros(3 * mesh.num_nodes), fixed_dofs=np.array(fixed),
                        dirichlet_values=np.zeros(len(fixed)))
    fe = ThermoStructuralFEA(mesh=mesh, mat_prop=mat, structural_bc=sbc, thermal_bc=tbc, solver=lin.Solvers.SPSOLVE,
                             thermoElasticReferenceTemperature=T0)
    return fe, mat, dT


def test_coupled_stress_uses_elastic_strain():
    from pyto.autodiff.qoi.stress import compute_pnorm_stress_autograd
    fe, mat, dT = _heated_bar(clamped=True)
    x = torch.ones(fe.mesh.num_elems, dtype=torch.float64)
    u = fe.solve(x, MaterialModel.SIMP)
    _, max_vm = compute_pnorm_stress_autograd(u, x, fe)
    expected = mat.youngs_modulus * mat.thermal_expansion_coefficient * dT
    assert max_vm.item() == pytest.approx(expected, rel=1e-6)                 # uniaxial -E alpha dT
    fe.postprocess()                                                          # NumPy path (subtracts thermal strain)
    assert max_vm.item() == pytest.approx(float(np.max(fe.vonMisesStress)), rel=1e-6)
    free, _, _ = _heated_bar(clamped=False)
    uf = free.solve(x, MaterialModel.SIMP)
    assert compute_pnorm_stress_autograd(uf, x, free)[1].item() < 1e-6 * expected   # free expansion: no stress


def test_mechanical_and_thermal_compliance(structural_fe_solver, thermal_fe_solver):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_thermoelastic import _coupled_problem
    fs = structural_fe_solver
    x = _x0(fs)
    u = fs.solve(x, MaterialModel.SIMP)
    assert R.evaluate("MechanicalCompliance", u, x, fs).item() == pytest.approx(R.evaluate("Compliance", u, x, fs).item())
    ft = thermal_fe_solver
    T = ft.solve(_x0(ft), MaterialModel.SIMP)
    assert R.evaluate("ThermalCompliance", T, _x0(ft), ft).item() == pytest.approx(R.evaluate("Compliance", T, _x0(ft), ft).item())
    fc = _coupled_problem()
    xc = _x0(fc)
    uc = fc.solve(xc, MaterialModel.SIMP)
    mech = R.evaluate("MechanicalCompliance", uc, xc, fc).item()
    assert mech == pytest.approx(float(np.dot(fc.bc.force, uc.detach().numpy())))       # mechanical load only
    assert mech != pytest.approx(R.evaluate("Compliance", uc, xc, fc).item(), rel=1e-3)  # thermal load excluded
    # thermal compliance of the coupled thermal field = that of the stand-alone thermal solve (same law)
    th = fc.thermal_fea
    Tc = fc.temperature
    edof = torch.as_tensor(fc.mesh.edofMatThermal, dtype=torch.long)
    from pyto.autodiff.material_model import get_thermal_material_model_scaling_torch as ks
    direct = (ks(xc, MaterialModel.SIMP, th.conductivity_penalty, th.conductivity_void_ratio)
              * torch.einsum("ei,ij,ej->e", Tc[edof], th.elem_stiff_torch[0], Tc[edof])).sum()
    assert R.evaluate("ThermalCompliance", uc, xc, fc).item() == pytest.approx(direct.item())
    assert np.array_equal(fc.mesh.edofMat, fc.mesh.edofMatStructural)       # shared connectivity restored


@pytest.mark.parametrize("name", ["StressPNorm", "MechanicalCompliance", "ThermalCompliance"])
def test_coupled_thermo_mechanical_gradients(name):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_thermoelastic import _coupled_problem
    _check_gradient(_coupled_problem(), name, eps=1e-5)
