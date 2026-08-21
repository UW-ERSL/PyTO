"""Torch-native QOI (quantity of interest) functions and the
objective/constraint dispatchers that route to them.

Every function here is a pure function of (sol, x, ...) returning a plain
torch scalar tensor -- no NumPy escape hatches, no dependence on
fe_solver's mutable postprocess() side-effect state. This is the one
production sensitivity path; the hand-derived NumPy adjoint equivalents
live in pyto.autodiff.reference_adjoint, used only as a test oracle.
"""
import torch

from pyto.topopt.common import TO_QOI

from pyto.autodiff.qoi.compliance import compute_compliance_torch
from pyto.autodiff.qoi.volume_fraction import compute_volume_constraint_torch
from pyto.autodiff.qoi.mass import compute_mass_torch
from pyto.autodiff.qoi.gvector import compute_gvector_torch
from pyto.autodiff.qoi.stress import compute_pnorm_stress_autograd

__all__ = [
    "compute_compliance_torch",
    "compute_volume_constraint_torch",
    "compute_mass_torch",
    "compute_gvector_torch",
    "compute_pnorm_stress_autograd",
    "compute_objective_and_gradient",
    "compute_constraint_and_gradient",
]


def compute_objective_and_gradient(to_params, sol: torch.Tensor, x: torch.Tensor, fe_solver, KE,
                material_model=None) -> torch.Tensor:
    """Compute the objective as a single torch scalar.

    Always returns one differentiable torch tensor -- topopt_mma.py gets
    the gradient via torch.autograd.grad(outputs=obj, inputs=x_raw), never
    from a returned NumPy gradient array. (Before Phase 4, MASS/GVECTOR/
    MAX_VONMISES_STRESS returned (value, numpy_gradient) tuples here, which
    is not what any current caller of this function actually consumes, and
    which crashed outright for MASS/GVECTOR since their old NumPy
    implementations couldn't handle a grad-tracked x/sol.)
    """
    objectiveType = to_params.Objective[0]  # first entry is the type of objective
    optionalParam = to_params.Objective[1]  # second entry is an optional parameter
    if objectiveType == TO_QOI.COMPLIANCE:
        return compute_compliance_torch(sol, x, fe_solver, KE, material_model)
    elif objectiveType == TO_QOI.VOLUME_FRACTION:
        return torch.mean(x)
    elif objectiveType == TO_QOI.MASS:
        return compute_mass_torch(x, fe_solver)
    elif objectiveType == TO_QOI.PNORM_STRESS:
        stressObj, _max_von_mises = compute_pnorm_stress_autograd(sol, x, fe_solver)
        return stressObj
    elif objectiveType == TO_QOI.MAX_VONMISES_STRESS:
        _pnorm_stress, max_von_mises = compute_pnorm_stress_autograd(sol, x, fe_solver)
        return max_von_mises
    elif objectiveType == TO_QOI.GVECTOR:
        g = optionalParam
        return compute_gvector_torch(sol, g)
    else:
        raise NotImplementedError(f"Objective {objectiveType} is not implemented yet.")


def compute_constraint_and_gradient(to_params, sol: torch.Tensor, x: torch.Tensor, fe_solver, KE,
                material_model=None) -> tuple:
    """Compute all constraints as a (m,1) torch tensor of scalars.

    dc (the second return value) is NOT populated with a correct gradient
    for every branch -- topopt_mma.py discards it (`c_t, _ =
    compute_constraint_and_gradient(...)`) and recomputes every gradient
    itself via torch.autograd.grad on c_t, so dc is kept only for return
    signature compatibility with any future caller, not as a real
    sensitivity. Only c[m,0] is guaranteed correct.

    stress_scaling lives on to_params (per optimization run) rather than as
    a mutable function attribute -- see plan Phase 4 -- so two separate
    optimization runs in the same process don't share this state.
    """
    nConstraints = len(to_params.Constraints)
    c = torch.zeros((nConstraints, 1), dtype=x.dtype)
    dc = torch.zeros((nConstraints, x.flatten().shape[0]), dtype=x.dtype)

    for m in range(nConstraints):
        constraintType = to_params.Constraints[m][0]  # first entry is the type of constraint
        optionalParam = to_params.Constraints[m][1]  # second entry is an optional parameter
        constraintLimit = to_params.Constraints[m][2]  # third entry is the constraint value
        if constraintType == TO_QOI.COMPLIANCE:
            compliance_t = compute_compliance_torch(sol, x, fe_solver, KE, material_model)
            c[m, 0] = compliance_t / constraintLimit - 1.0
        elif constraintType == TO_QOI.MASS:
            mass_t = compute_mass_torch(x, fe_solver)
            c[m, 0] = mass_t / constraintLimit - 1.0
        elif constraintType == TO_QOI.VOLUME_FRACTION:
            c[m, 0] = compute_volume_constraint_torch(x, constraintLimit)
        elif constraintType == TO_QOI.PNORM_STRESS:
            pnorm_stress, _max_von_mises = compute_pnorm_stress_autograd(sol, x, fe_solver)
            c[m, 0] = pnorm_stress / constraintLimit - 1.0
        elif constraintType == TO_QOI.MAX_VONMISES_STRESS:
            # See De Leon, D.M., Alexandersen, J., O. Fonseca, J.S. and Sigmund, O., 2015.
            # Stress-constrained topology optimization for compliant mechanism design.
            # Structural and Multidisciplinary Optimization, 52(5), pp.929-943
            pnorm_stress, max_von_mises = compute_pnorm_stress_autograd(sol, x, fe_solver)
            normalized_pnorm = to_params.stress_scaling * pnorm_stress
            c[m, 0] = normalized_pnorm / constraintLimit - 1.0
            if max_von_mises > constraintLimit:
                to_params.stress_scaling = (
                    0.5 * (max_von_mises / pnorm_stress).item() + 0.5 * to_params.stress_scaling
                )
        elif constraintType == TO_QOI.STRESS_FAILURE_FACTOR:
            # NOTE: this branch referenced a nonexistent TO_QOI.STRESS_SAFETY_FACTOR
            # before Phase 4 (the real enum member is STRESS_FAILURE_FACTOR --
            # see topopt_common.py's TO_QOI and topopt_structural_benchmarks.py,
            # which actually constructs constraints with it, e.g.
            # LBracketTopLoad_Vol_StressFailureFactor).
            #
            # The `1.0 / constraintLimit` term here was flagged, not fixed, in
            # Phase 4 pending physics verification. Verified now: failure
            # factor is defined as stress/yieldStrength (opposite of safety
            # factor = yieldStrength/stress, per topopt_common.py's enum
            # comment and the benchmark's own "failure limit is the opposite
            # of safety factor" comment). The constraint failure_factor <=
            # constraintLimit therefore means stress <= yieldStrength *
            # constraintLimit, i.e. c = stress/(yieldStrength*constraintLimit)
            # - 1.0 <= 0 -- matching the value/limit - 1.0 <= 0 pattern every
            # other branch here uses. The old `1.0 / constraintLimit` term
            # instead allowed stress up to yieldStrength/constraintLimit --
            # with the benchmark's constraintLimit=0.5, that's stress <= 2x
            # yield strength, the opposite of the intended "use at most half
            # the material's strength" margin. Fixed to multiply by
            # constraintLimit instead of dividing by it (see
            # test_stress_failure_factor_constraint_direction in
            # tests/test_qoi_gradients.py for the regression check).
            pnorm_stress, max_von_mises = compute_pnorm_stress_autograd(sol, x, fe_solver)
            yieldStrength = fe_solver.mat_prop.yield_strength
            normalized_pnorm = to_params.stress_scaling * pnorm_stress
            allowed_stress = yieldStrength * constraintLimit
            c[m, 0] = normalized_pnorm / allowed_stress - 1.0
            if max_von_mises > allowed_stress:
                to_params.stress_scaling = (
                    0.5 * (max_von_mises / pnorm_stress).item() + 0.5 * to_params.stress_scaling
                )
        else:
            raise NotImplementedError(f"Constraint {constraintType} is not implemented yet.")
    return c, dc
