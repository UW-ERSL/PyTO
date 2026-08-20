"""Smoke tests for the three optimizer drivers: does a couple of iterations
run at all, on both physics, without exceptions?

The xfail cases here are not aspirational -- each one was verified to fail
with exactly the stated exception before being marked, and documents a real,
currently-broken path found during the file-restructuring plan's Phase 0.
They are the "did I fix it" signal for later phases (see
~/.claude/plans/cheerful-dazzling-seahorse.md, Phase 4/5): once a phase's
fix lands, its xfail should start showing as an unexpected pass (XPASS) and
be promoted to a normal passing test.
"""
import numpy as np
import pytest
import torch

import topopt_mma
import topopt_ocm
import topopt_levelset
import topopt_pareto
from topopt_common import FEA_MODE


def test_mma_structural_runs(structural_fe_solver, structural_problem):
    _, _, _, _, to_params = structural_problem
    sol, history, success, error_msg, n_feas = topopt_mma.topopt_mma(
        fe_solver=structural_fe_solver,
        to_params=to_params,
        maxMMAIterations=2,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 2
    assert np.isfinite(history["objective"][-1])


def test_mma_thermal_runs(thermal_fe_solver, thermal_problem):
    # Was xfail through Phase 3 (HexThermalFEA.solve() was pure NumPy/SciPy);
    # fixed in Phase 4 by adding get_thermal_material_model_scaling_torch and
    # rewriting HexThermalFEA.solve()/postprocess()/get_thermoelastic_force()
    # to be torch-native, mirroring HexStructuralFEA.
    _, _, _, _, to_params = thermal_problem
    sol, history, success, error_msg, n_feas = topopt_mma.topopt_mma(
        fe_solver=thermal_fe_solver,
        to_params=to_params,
        maxMMAIterations=2,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 2
    assert np.isfinite(history["objective"][-1])


def test_oc_structural_runs(structural_fe_solver, structural_problem):
    # Was xfail through Phase 4 (topopt_ocm.py passed a plain NumPy x into
    # fe_solver.solve(), which requires torch since the autodiff migration);
    # fixed in Phase 5 by rewriting the driver to build the same
    # x_raw (requires_grad) -> solve -> objective -> torch.autograd.grad
    # chain topopt_mma.py already used, and by fixing 4 setPseudoDensity()
    # call sites that also required a torch tensor (a second, previously
    # masked bug -- see plan Phase 5).
    _, _, _, _, to_params = structural_problem
    sol, history, success, error_msg, n_feas = topopt_ocm.topopt_optimality_criteria(
        fe_solver=structural_fe_solver,
        to_params=to_params,
        maxIterations=3,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 3
    assert np.isfinite(history["objective"][-1])
    assert history["objective"][-1] < history["objective"][0]


def test_levelset_structural_runs(structural_fe_solver, structural_problem):
    # Was broken since the torch-autodiff migration (Phase 0/4, never
    # explicitly xfailed since topopt_levelset.py predates this plan's
    # Phase 0 test suite): fe_solver.solve(rho, ...) and
    # mesh.setPseudoDensity(rho) both received a plain NumPy rho, and both
    # now require torch. Fixed in Phase 5 by wrapping rho in torch.tensor(...)
    # at each of the 4 call sites, and by converting fe_solver.sol back to
    # NumPy inside compute_compliance_and_sensitivity (which reads
    # fe_solver.sol directly, not as a parameter). No torch.autograd.grad
    # chain needed here -- unlike MMA/OC, Levelset's sensitivity
    # (compute_compliance_and_sensitivity) is a hand-derived NumPy shape
    # derivative, not a torch gradient, so rho never needs requires_grad=True.
    #
    # Convergence sanity, not exact numbers (see plan Phase 5 completion
    # notes on why gradient-checking a shape derivative would test the wrong
    # thing): volume fraction should trend down toward the target over a
    # modest number of iterations, not necessarily reach it.
    _, _, _, _, to_params = structural_problem
    sol, history, success, error_msg, n_feas = topopt_levelset.topopt_levelset(
        FEA_MODE.STRUCTURAL,
        fe_solver=structural_fe_solver,
        to_params=to_params,
        maxIterations=15,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 2
    assert np.all(np.isfinite(history["objective"]))
    assert history["volfrac"][-1] < history["volfrac"][0]


def test_pareto_structural_runs(structural_fe_solver, structural_problem):
    # Was broken since the torch-autodiff migration (Phase 0/4, never
    # explicitly xfailed for the same reason as Levelset): fe_solver.solve(x)/
    # mesh.setPseudoDensity(x) both received plain NumPy, both now require
    # torch. Fixed in Phase 5 the same way as Levelset (torch.tensor(...) at
    # each of the 2 solve + 7 setPseudoDensity call sites), plus one more
    # thing Levelset didn't need: sol itself has to be converted back to
    # NumPy right after solve() (`.detach().cpu().numpy()`), since Pareto's
    # local compute_objective_topological_sensitivity_compliance mixes it
    # into plain NumPy expressions (np.dot(...) * sol[...], which raises
    # TypeError when the right-hand side is still a torch tensor even if it
    # doesn't require grad).
    #
    # Pareto's own local topological-sensitivity formulas are deliberately
    # left untouched -- see this test file's neighboring note and
    # pareto.py's module docstring for the known (flagged, not fixed)
    # duplication with pyto.topopt.topological_sensitivity.
    #
    # Unlike Levelset, Pareto converges very quickly on this small fixture
    # (<1s), so this runs it to actual completion rather than truncating.
    _, _, _, _, to_params = structural_problem
    sol, history, success, error_msg, n_feas = topopt_pareto.topopt_pareto(
        FEA_MODE.STRUCTURAL,
        fe_solver=structural_fe_solver,
        to_params=to_params,
        print_progress=False,
        plot_progress=False,
    )
    assert success, error_msg
    assert np.all(np.isfinite(history["objective"]))
    target_volfrac = to_params.Constraints[0][2]
    assert abs(history["volfrac"][-1] - target_volfrac) < 1e-6
    assert history["volfrac"][0] > history["volfrac"][-1]
