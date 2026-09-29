"""Expression compiler (plan S2): values, gradients, safety, error messages."""
import numpy as np
import pytest
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import responses as R
from pyto.topopt.expressions import ExpressionError, compile_expression


@pytest.fixture
def solved(structural_fe_solver):
    fe = structural_fe_solver
    x = torch.tensor(0.3 + 0.6 * np.random.default_rng(0).random(fe.mesh.num_elems), dtype=torch.float64)
    loaded = np.unique(np.nonzero(fe.bc.force)[0] // 3)
    sel = {"Tip": {"nodes": loaded}, "Left": {"elements": np.arange(10)}}
    return fe, x, sel


def test_values_match_direct_responses(solved):
    fe, x, sel = solved
    u = fe.solve(x, MaterialModel.SIMP)
    tip = sel["Tip"]["nodes"]
    direct = R.evaluate("Displacement", u, x, fe, nodes=tip, component="y") / 1e-3 + 0.1 * x.mean()
    e = compile_expression("Displacement(Tip, y, mean) / 1e-3 + 0.1 * VolumeFraction()", sel)
    assert e(u, x, fe).item() == pytest.approx(direct.item())
    assert e.responses == {"Displacement", "VolumeFraction"} and e.depends_on_solution
    # keyword form, aliases, All, material constants and functions
    kw = compile_expression("Displacement(selection=Tip, component=y, aggregate=mean)", sel)
    assert kw(u, x, fe).item() == pytest.approx(R.evaluate("Displacement", u, x, fe, nodes=tip, component="y").item())
    assert compile_expression("VolumeFraction(Left)", sel)(u, x, fe).item() == pytest.approx(x[:10].mean().item())
    assert compile_expression("VolumeFraction(All)")(u, x, fe).item() == pytest.approx(x.mean().item())
    s = compile_expression("StressPNorm() / YieldStrength")(u, x, fe).item()
    assert s == pytest.approx(R.evaluate("StressFailureFactor", u, x, fe).item())
    c = R.evaluate("Compliance", u, x, fe)
    assert compile_expression("sqrt(Compliance()) ** 2 - -1")(u, x, fe).item() == pytest.approx(c.item() + 1)
    sm = compile_expression("smooth_max(Compliance(), 2 * Compliance(), p=32)")(u, x, fe).item()
    assert 2 * c.item() <= sm <= 2 * c.item() * 2 ** (1 / 32) * 1.0001
    assert compile_expression("Compliance()").is_plain_compliance
    assert not compile_expression("VolumeFraction()").depends_on_solution


def test_gradient_equals_direct_response_gradient(solved):
    fe, x0, sel = solved
    expr = compile_expression("Displacement(Tip, magnitude, pnorm, p=6) * 1e4 + StrainEnergy(Left)", sel)
    tip, left = sel["Tip"]["nodes"], sel["Left"]["elements"]

    def grad(f):
        x = x0.clone().requires_grad_(True)
        return torch.autograd.grad(f(fe.solve(x, MaterialModel.SIMP), x), x)[0]

    g_expr = grad(lambda u, x: expr(u, x, fe))
    g_direct = grad(lambda u, x: R.displacement(u, x, fe, nodes=tip, component="magnitude", aggregate_by="pnorm", p=6.0)
                    * 1e4 + R.strain_energy(u, x, fe, elements=left))
    assert torch.allclose(g_expr, g_direct, rtol=1e-12, atol=0)


@pytest.mark.parametrize("text, message", [
    ("__import__('os').system('echo hi')", "Only plain function names|not allowed"),
    ("Compliance().__class__", "not allowed"),
    ("(lambda: 1)()", "Only plain function names"),
    ("[Compliance() for i in range(3)]", "not allowed"),
    ("open('f')", "Unknown function open"),
    ("'text'", "Only numbers"),
    ("Compliance()[0]", "not allowed"),
    ("abs(Compliance())", "not differentiable"),
    ("max(Compliance(), 1)", "smooth_max"),
    ("Compliance", "is a response: call it"),
    ("Displacement(Nowhere, y)", "Unknown selection 'Nowhere'"),
    ("Displacement(Tip, w)", "component must be one of"),
    ("Displacement(Tip, y, median)", "aggregate_by must be one of"),
    ("VolumeFraction(Tip)", "has no elements"),
    ("Displacement(Tip, y, pnorm, 0.5)", "p must be a number > 1"),
    ("Compliance(1)", "at most 0 positional"),
    ("Displacement(Tip, color=y)", "no parameter 'color'"),
    ("Foo * 2", "Unknown name 'Foo'"),
    ("1 +", "Syntax error"),
    ("", "empty"),
])
def test_rejected_expressions(solved, text, message):
    _, _, sel = solved
    with pytest.raises(ExpressionError, match=message):
        compile_expression(text, sel)


def test_physics_checked_at_evaluation(thermal_fe_solver):
    fe = thermal_fe_solver
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64)
    T = fe.solve(x, MaterialModel.SIMP)
    assert compile_expression("Temperature(All, mean)")(T, x, fe).item() == pytest.approx(T.mean().item())
    e = compile_expression("Displacement(All, y)")
    assert e.physics == {"structural", "thermo-structural"}
    with pytest.raises(ExpressionError, match="Displacement not available for thermal"):
        e(T, x, fe)
