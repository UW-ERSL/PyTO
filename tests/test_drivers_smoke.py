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


@pytest.mark.xfail(
    reason=(
        "HexThermalFEA.solve() is still pure NumPy/SciPy (no torch material-model "
        "twin exists in topopt_material_model.py for thermal, unlike "
        "get_structural_material_model_scaling_torch). topopt_mma.py always builds "
        "x as a requires_grad torch tensor, so the thermal path raises "
        "RuntimeError: Can't call numpy() on Tensor that requires grad. "
        "Fix planned in plan Phase 1/4."
    ),
    strict=True,
)
def test_mma_thermal_runs(thermal_fe_solver, thermal_problem):
    _, _, _, _, to_params = thermal_problem
    topopt_mma.topopt_mma(
        fe_solver=thermal_fe_solver,
        to_params=to_params,
        maxMMAIterations=2,
        print_progress=False,
        plot_progress=False,
    )


@pytest.mark.xfail(
    reason=(
        "topopt_ocm.py keeps the design variable as a plain NumPy array and calls "
        "fe_solver.solve(x, material_model) directly; HexStructuralFEA.solve() now "
        "requires a torch tensor (get_structural_material_model_scaling_torch does "
        "x**p with p as a torch tensor), so this raises "
        "TypeError: unsupported operand type(s) for ** ... 'numpy.ndarray' and 'Tensor'. "
        "Fix planned in plan Phase 5."
    ),
    strict=True,
)
def test_oc_structural_runs(structural_fe_solver, structural_problem):
    _, _, _, _, to_params = structural_problem
    topopt_ocm.topopt_optimality_criteria(
        fe_solver=structural_fe_solver,
        to_params=to_params,
        maxIterations=2,
        print_progress=False,
        plot_progress=False,
    )
