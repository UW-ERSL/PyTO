"""Hand-derived ("manual") gradients for the MMA and OC drivers, as an alternative to automatic differentiation.

to_params.Gradient = "manual" makes MMA/OC use these instead of torch.autograd. They exist only where a hand-derived
sensitivity is available and verified against autograd (tests/test_manual_sensitivities.py):

  objective    COMPLIANCE (structural, thermal, thermo-structural), VOLUME_FRACTION, MASS, GVECTOR (structural)
  constraints  VOLUME_FRACTION (<=), VOLUME_FRACTION_MIN (>=), MASS, COMPLIANCE

Not available (manual_support() says why): user expressions / Python functions (GFUNCTION), stress objectives and
constraints (the legacy stress sensitivity does not match the autograd gradient), design-dependent body forces,
per-problem conductivity laws, and compliance with varying prescribed values together with loads. Values are computed by the same functions as the autograd path, so both
modes optimize exactly the same problem; only the derivative differs in how it is obtained.
"""
import contextlib
import io
import types

import numpy as np
import scipy.sparse as sps
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.topopt.common import TO_QOI

MANUAL_OBJECTIVES = {TO_QOI.COMPLIANCE, TO_QOI.VOLUME_FRACTION, TO_QOI.MASS, TO_QOI.GVECTOR}
MANUAL_CONSTRAINTS = {TO_QOI.VOLUME_FRACTION, TO_QOI.VOLUME_FRACTION_MIN, TO_QOI.MASS, TO_QOI.COMPLIANCE}


def _physics(fe_solver):
    from pyto.autodiff.qoi.responses import physics_of
    return physics_of(fe_solver)


def formulation_support(to_params) -> tuple:
    """manual_support() without the solver: only which objective/constraint types have a hand-derived gradient."""
    obj = to_params.Objective[0]
    if obj not in MANUAL_OBJECTIVES:
        return False, (f"No hand-derived gradient for the objective {getattr(obj, 'name', obj)}"
                       + (" (user expressions and Python functions need automatic differentiation)"
                          if obj == TO_QOI.GFUNCTION else "") + ".")
    for c in to_params.Constraints:
        if c[0] not in MANUAL_CONSTRAINTS:
            return False, (f"No hand-derived gradient for the constraint {getattr(c[0], 'name', c[0])}"
                           + (" (stress sensitivities are available with automatic differentiation only)"
                              if c[0] in (TO_QOI.MAX_VONMISES_STRESS, TO_QOI.STRESS_FAILURE_FACTOR,
                                          TO_QOI.PNORM_STRESS) else "") + ".")
    return True, ""


def manual_support(to_params, fe_solver) -> tuple:
    """(True, "") if every objective/constraint has a hand-derived gradient for this solver, else (False, reason)."""
    ok, reason = formulation_support(to_params)
    if not ok:
        return ok, reason
    obj = to_params.Objective[0]
    uses_compliance = obj == TO_QOI.COMPLIANCE or any(c[0] == TO_QOI.COMPLIANCE for c in to_params.Constraints)
    physics = _physics(fe_solver)
    if uses_compliance and getattr(fe_solver, "elem_body_force", None) is not None:
        return False, "The hand-derived compliance gradient does not include design-dependent body forces."
    thermal = fe_solver.thermal_fea if physics == "thermo-structural" else fe_solver if physics == "thermal" else None
    if uses_compliance and thermal is not None and (getattr(thermal, "conductivity_penalty", None) is not None
                                                    or getattr(thermal, "conductivity_void_ratio", None) is not None):
        return False, "The hand-derived compliance gradient assumes the default conductivity law."
    if uses_compliance:
        bcs = [fe_solver.bc] + ([fe_solver.thermal_fea.bc] if physics == "thermo-structural" else [])
        for bc in bcs:
            if _varying_prescribed(bc) and np.any(bc.force):
                return False, ("The hand-derived compliance gradient covers load-driven or prescribed-value-driven "
                               "problems, not both at once (varying prescribed values together with loads).")
    if obj == TO_QOI.GVECTOR and physics != "structural":
        return False, "The hand-derived g.u gradient is for structural problems only."
    return True, ""


def _varying_prescribed(bc) -> bool:
    """Prescribed values that actually drive the field. A uniform value (the same on every fixed dof) is a constant
    temperature offset / rigid translation, which K ignores (K.1 = 0), so it counts as no prescribed value
    (e.g. HeatPlate: heat load + all fixed nodes at 23 C behaves as load-driven)."""
    v = np.asarray(bc.dirichlet_values, dtype=float)
    return v.size > 0 and np.ptp(v) > 1e-12 * max(1.0, np.abs(v).max())


def _scipy_stiffness(fe_solver):
    """The assembled stiffness of the last solve as SciPy CSR (the legacy adjoint solves need SciPy)."""
    K = fe_solver.stiff_mtrx.coalesce()
    i = K.indices().cpu().numpy()
    return sps.csr_matrix((K.values().detach().cpu().numpy(), (i[0], i[1])), shape=tuple(K.shape))


def _compliance_gradient(sol, x, fe_solver, material_model):
    """dC/dx for C = u^T K u at the last solve (sol = that solve's solution)."""
    from pyto.autodiff.reference_adjoint import compute_compliance_and_gradient
    physics = _physics(fe_solver)
    if physics == "thermo-structural":
        from pyto.autodiff.material_model import _SIMP_STRUCTURAL_PENALTY, _SIMP_THERMAL_PENALTY
        from pyto.physics.thermoelastic.topopt_thermostructural_sensitivity import ThermoElasticSensitivity
        with contextlib.redirect_stdout(io.StringIO()):               # the analyser prints diagnostics
            return ThermoElasticSensitivity(fe_solver.thermal_fea, fe_solver).compute_compliance_sensitivity(
                x, fe_solver.temperature.detach().cpu().numpy(), sol, p=_SIMP_STRUCTURAL_PENALTY,
                q=_SIMP_THERMAL_PENALTY, material_model=material_model, solver=fe_solver.solver)
    # Load-driven: dC/dx = -dk/dx u^T ke u. Driven only by prescribed values (no load, e.g. FourCornersThermal): the
    # energy rises with stiffness and the sign flips (checked against autograd).
    sign = -1.0 if (not np.any(fe_solver.bc.force) and _varying_prescribed(fe_solver.bc)) else 1.0
    KE = fe_solver.elem_stiff[0]
    mesh = fe_solver.mesh
    saved = mesh.edofMat          # shared connectivity: make it match this solver (thermal meshes carry both)
    mesh.edofMat = mesh.edofMatThermal if physics == "thermal" else mesh.edofMatStructural
    try:
        return sign * compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)[1]
    finally:
        mesh.edofMat = saved


def _gvector_gradient(sol, x, fe_solver, material_model, g):
    from pyto.autodiff.reference_adjoint import compute_solution_dotproduct_and_gradient
    proxy = types.SimpleNamespace(mesh=fe_solver.mesh, bc=fe_solver.bc, solver=fe_solver.solver,
                                  dsolver=fe_solver.dsolver, kwargs=fe_solver.kwargs,
                                  stiff_mtrx=_scipy_stiffness(fe_solver))
    return compute_solution_dotproduct_and_gradient(sol, x, proxy, fe_solver.elem_stiff[0], material_model,
                                                    np.asarray(g, dtype=float))[1]


def manual_gradients(to_params, sol, x, fe_solver, material_model=MaterialModel.SIMP):
    """Gradients w.r.t. the physical density x (numpy) of the objective and of every constraint, in the exact form
    compute_objective_and_gradient / compute_constraint_and_gradient evaluate them. sol and x belong to the last
    solve. Returns (dobj (n,), dcons (m, n))."""
    ok, reason = manual_support(to_params, fe_solver)
    if not ok:
        raise ValueError(f"Manual gradient not available: {reason}")
    sol = sol.detach().cpu().numpy() if isinstance(sol, torch.Tensor) else np.asarray(sol)
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    n = x.size
    cache = {}

    def compliance():
        if "C" not in cache:
            cache["C"] = _compliance_gradient(sol, x, fe_solver, material_model)
        return cache["C"]

    def mass():
        vol = float(np.prod(fe_solver.mesh.elem_size))
        return np.full(n, vol * fe_solver.mat_prop.mass_density)

    obj = to_params.Objective[0]
    if obj == TO_QOI.COMPLIANCE:
        d_obj = compliance()
    elif obj == TO_QOI.VOLUME_FRACTION:
        d_obj = np.full(n, 1.0 / n)
    elif obj == TO_QOI.MASS:
        d_obj = mass()
    else:                                                          # GVECTOR
        d_obj = _gvector_gradient(sol, x, fe_solver, material_model, to_params.Objective[1])

    rows = []
    for ctype, _param, limit in to_params.Constraints:
        if ctype == TO_QOI.VOLUME_FRACTION:                        # mean(x)/limit - 1
            rows.append(np.full(n, 1.0 / (n * limit)))
        elif ctype == TO_QOI.VOLUME_FRACTION_MIN:                  # 1 - mean(x)/limit
            rows.append(np.full(n, -1.0 / (n * limit)))
        elif ctype == TO_QOI.MASS:                                 # mass/limit - 1
            rows.append(mass() / limit)
        else:                                                      # COMPLIANCE: C/limit - 1
            rows.append(compliance() / limit)
    return np.asarray(d_obj, dtype=float), np.array(rows, dtype=float).reshape(len(rows), n)
