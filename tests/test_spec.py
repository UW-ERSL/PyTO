"""OptimizationSpec (plan S3): serialization, validation, compilation, benchmark round trip, maximize."""
import io
import contextlib

import numpy as np
import pytest
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.topopt.common import TOParams, TO_QOI
from pyto.topopt.spec import (ConstraintSpec, ManufacturingSpec, MethodSpec, ObjectiveSpec, OptimizationSpec,
                              RegularizationSpec, compile_spec, spec_from_to_params, validate)


def _spec(**kw):
    return OptimizationSpec(**kw)


def test_json_round_trip():
    s = _spec(objective=ObjectiveSpec("Displacement(Tip, y, mean) / 1e-3", "maximize"),
              constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3),
                           ConstraintSpec("StressFailureFactor()", "<=", 0.5, enabled=False)],
              method=MethodSpec("MMA", 80, 0.1), regularization=RegularizationSpec(2.0, True),
              manufacturing=ManufacturingSpec("Z", ["X"], 4, "Keep"))
    assert OptimizationSpec.from_json(s.to_json()) == s
    with pytest.raises(ValueError, match="newer"):
        OptimizationSpec.from_dict({**s.to_dict(), "version": 99})


@pytest.mark.parametrize("spec, physics, message", [
    (_spec(objective=ObjectiveSpec("Displacement(Tip, y)")), "thermal", "Displacement not available for thermal"),
    (_spec(objective=ObjectiveSpec("Foo()")), "structural", "Unknown function Foo"),
    (_spec(objective=ObjectiveSpec("MaxStress()")), "structural", "can only be used as a constraint"),
    (_spec(constraints=[ConstraintSpec("MaxStress() * 2", "<=", 1.0)]), "structural", "on its own as a constraint"),
    (_spec(constraints=[ConstraintSpec("Compliance()", ">=", -1.0)]), "structural", "a >= bound must be > 0"),
    (_spec(constraints=[ConstraintSpec("Compliance()", "==", 1.0)]), "structural", "operator must be"),
    (_spec(objective=ObjectiveSpec("VolumeFraction()"), constraints=[]), "structural", "empty design"),
    (_spec(objective=ObjectiveSpec("3 * 2")), "structural", "does not depend on the design"),
    (_spec(method=MethodSpec("NEWTON")), "structural", "Unknown method"),
    (_spec(method=MethodSpec(move_limit=0)), "structural", "Move limit"),
    (_spec(manufacturing=ManufacturingSpec(keep_selection="Nope")), "structural", "not defined"),
])
def test_validation_errors(spec, physics, message):
    issues = validate(spec, physics, {"Tip": {"nodes": [1, 2]}})
    assert any(i.level == "error" and message in i.message for i in issues), issues


def test_validation_warnings():
    issues = validate(_spec(constraints=[]), "structural")
    assert [i.level for i in issues] == ["warning"] and "full design" in issues[0].message
    assert validate(_spec(), "structural") == []                        # default: compliance, vf <= 0.5
    ok = _spec(constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.4), ConstraintSpec("VolumeFraction()", "<=", 0.4)])
    assert any("more than once" in i.message for i in validate(ok, "structural"))


def test_compile_uses_native_types_when_exact_and_gfunction_otherwise(structural_fe_solver):
    fe = structural_fe_solver
    tip = np.unique(np.nonzero(fe.bc.force)[0] // 3)
    sel = {"Tip": {"nodes": tip}, "Keep": {"elements": np.array([0, 1, 2])}}
    spec = _spec(constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.4),
                              ConstraintSpec("VolumeFraction()", ">=", 0.1),
                              ConstraintSpec("MaxStress()", "<=", 2e8),
                              ConstraintSpec("Displacement(Tip, y, pnorm)", "<=", 1e-3),
                              ConstraintSpec("Displacement(Tip, magnitude)", ">=", 1e-6),
                              ConstraintSpec("Compliance()", "<=", 5.0, enabled=False)],
                 manufacturing=ManufacturingSpec("Z", ["X", "Y"], 3, "Keep"),
                 regularization=RegularizationSpec(2.5, True), method=MethodSpec("MMA", 42))
    p = compile_spec(spec, sel)
    assert p.Objective == (TO_QOI.COMPLIANCE, None)
    kinds = [c[0] for c in p.Constraints]
    assert kinds == [TO_QOI.VOLUME_FRACTION, TO_QOI.VOLUME_FRACTION_MIN, TO_QOI.MAX_VONMISES_STRESS,
                     TO_QOI.GFUNCTION, TO_QOI.GFUNCTION]
    assert p.Constraints[3][2] == 1e-3 and p.Constraints[4][2] is None          # >= compiled to 1 - f/b <= 0
    assert (p.ExtrudeZ, p.XSymmetry, p.YSymmetry, p.ZSymmetry, p.ZAxisAngularSymmetry) == (True, True, True, False, 3)
    assert p.MaxIterations == 42 and p.RelativeFilterRadius == 2.5 and p.HeavisideProjection
    assert list(p.ElemsToKeep) == [0, 1, 2]
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64)
    u = fe.solve(x, MaterialModel.SIMP)
    mag = torch.linalg.norm(u.reshape(-1, 3)[tip], dim=1).mean()
    assert p.Constraints[4][1](u, x, fe).item() == pytest.approx(1 - mag.item() / 1e-6)
    m = compile_spec(_spec(objective=ObjectiveSpec("Compliance()", "maximize")), sel)
    assert m.Objective[0] == TO_QOI.GFUNCTION                                     # maximize is never native
    assert m.Objective[1](u, x, fe).item() == pytest.approx(-(u @ torch.as_tensor(fe.bc.force)).item(), rel=1e-9)


def _problem(enum_member, getter, ndof):
    with contextlib.redirect_stdout(io.StringIO()):
        return getter(enum_member, nDOFDesired=ndof)


@pytest.mark.parametrize("name", ["MBBBeam", "LBracketTopLoad_Stress_Vol", "LBracketTopLoad_Vol_Stress",
                                  "LBracketTopLoad_Mass_StressFF", "CantileverMidLoadVolumeCompliance",
                                  "GravityPlate", "Inverter", "CentrifugalPlate"])
def test_benchmark_formulations_round_trip(name):
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    mesh, mat, bc, ebf, p = _problem(getattr(StructuralTOExamples, name), getStructuralTOProblem, 2000)
    spec, sel = spec_from_to_params(p)
    assert not [i for i in validate(spec, "structural", sel) if i.level == "error"]
    q = compile_spec(OptimizationSpec.from_json(spec.to_json()), sel, base=p)
    same = lambda a, b: a[0] == b[0] and (a[0] == TO_QOI.GFUNCTION or a[1:] == b[1:])
    assert all(same(a, b) for a, b in zip(q.Constraints, p.Constraints)) and len(q.Constraints) == len(p.Constraints)
    for attr in ("MaxIterations", "RelativeFilterRadius", "ExtrudeX", "ExtrudeY", "ExtrudeZ", "XSymmetry",
                 "YSymmetry", "ZSymmetry", "ZAxisAngularSymmetry", "HeavisideProjection"):
        assert getattr(q, attr) == getattr(p, attr), attr
    if p.ElemsToKeep is not None:
        assert list(q.ElemsToKeep) == list(p.ElemsToKeep)
    if p.Objective[0] == TO_QOI.GVECTOR:                                          # Inverter: g.u as an expression
        from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
        import pyto.solve.numpy_backend as lin
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=lin.Solvers.SPSOLVE, elem_body_force=ebf)
        x = torch.full((mesh.num_elems,), 0.3, dtype=torch.float64)
        u = fe.solve(x, MaterialModel.SIMP)
        assert q.Objective[1](u, x, fe).item() == pytest.approx(float(np.dot(p.Objective[1], u.detach().numpy())))
    else:
        assert q.Objective == p.Objective


def test_maximize_tip_deflection_equals_compliance_run():
    # One tip load: compliance C = F * (-mean u_y at the tip). Maximizing mean u_y (negative) is the compliance problem;
    # MMA normalizes the objective by its first value, so both runs must take the same path.
    import pyto.solve.numpy_backend as lin
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.drivers.mma import topopt_mma

    def run(spec_or_none):
        mesh, mat, bc, ebf, p = _problem(StructuralTOExamples.CantileverTipLoad, getStructuralTOProblem, 1500)
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=lin.Solvers.SPSOLVE)
        if spec_or_none is not None:
            tip = np.unique(np.nonzero(bc.force)[0] // 3)
            p = compile_spec(spec_or_none, {"Tip": {"nodes": tip}}, base=p)
        with contextlib.redirect_stdout(io.StringIO()):
            h = topopt_mma(fe, to_params=p, maxMMAIterations=8, print_progress=False, binarize_topology=False)[1]
        return np.ravel(np.array(h["objective"], dtype=float)), np.array(h["volfrac"], dtype=float), float(np.abs(bc.force).sum())

    J_c, vf_c, F = run(None)
    J_u, vf_u, _ = run(_spec(objective=ObjectiveSpec("Displacement(Tip, y, mean)", "maximize"),
                             constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.5)]))
    # The two objectives and their normalized gradients agree to ~2e-12 (checked); MMA's subproblem solver amplifies
    # such round-off differences to ~1e-4 over a few iterations, hence the 1e-3 tolerance.
    assert np.allclose(vf_u, vf_c, rtol=1e-3)
    assert np.allclose(J_u * F, J_c, rtol=1e-3)           # history holds the minimized value -mean(u_y) = C / F


def test_progress_labels_and_ge_negative_bound():
    """compile_spec records what the user wrote (for the iteration print) and refuses '>= negative bound'."""
    spec = OptimizationSpec(objective=ObjectiveSpec("Compliance()", "minimize"),
                            constraints=[ConstraintSpec("VolumeFraction()", ">=", 0.2),
                                         ConstraintSpec("VolumeFraction()", "<=", 0.5)])
    p = compile_spec(spec)
    assert p.ObjectiveLabel == ("Compliance()", "minimize")
    assert p.ConstraintLabels == [("VolumeFraction()", ">=", 0.2), ("VolumeFraction()", "<=", 0.5)]
    spec.constraints = [ConstraintSpec("VolumeFraction()", ">=", -0.1)]
    with pytest.raises(ValueError, match="must be > 0"):
        compile_spec(spec)
