"""Shared fixtures for the PyTO test suite.

The ``src/`` package uses flat, non-package imports everywhere
(``import hex_mesher``, ``from topopt_common import *``), so every module
assumes ``src/`` is directly on ``sys.path``. This conftest adds it once so
existing modules can be imported unmodified from ``tests/``.
"""
import os
import sys

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import deflation
import hex_structural_fea
import hex_thermal_fea
import torch_spsolve
import linear_solvers
import hex_element_stiffness
from topopt_material_model import MaterialModel
from topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
from topopt_thermal_benchmarks import ThermalTOExamples, getThermalTOProblem


# Small fixture sizes chosen for fast test turnaround (~1-2s to mesh).
STRUCTURAL_NDOF = 600
THERMAL_NDOF = 300


@pytest.fixture(scope="session")
def structural_problem():
    """A small structural TO problem: mesh, mat_prop, bc, elem_body_force, to_params.

    NOTE: getStructuralTOProblem() does not populate mesh.edofMat (only
    mesh.edofMatStructural / mesh.edofMatThermal) -- HexStructuralFEA.solve()
    still expects the flat mesh.edofMat name. Today, only PyTOGUI.py patches
    this manually (see PyTOGUI.py:2951-2973) before constructing an FEA
    solver from a mesh. We do the same patch here since it reflects the
    real, current calling convention -- this gap itself is exercised
    directly by test_drivers_smoke.py's xfail cases.
    """
    mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(
        StructuralTOExamples.ShortCantileverTipLoad, nDOFDesired=STRUCTURAL_NDOF
    )
    mesh.createEdofMatStructural()
    mesh.edofMat = mesh.edofMatStructural
    return mesh, mat_prop, bc, elem_body_force, to_params


@pytest.fixture
def structural_fe_solver(structural_problem):
    """A fresh HexStructuralFEA instance per test (avoids cross-test state leakage)."""
    mesh, mat_prop, bc, elem_body_force, _ = structural_problem
    return hex_structural_fea.HexStructuralFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc,
        solver=torch_spsolve.Solvers.SPSOLVE,
        dsolver=deflation.DeflationSolver(use_gpu=False),
        elem_body_force=elem_body_force,
    )


@pytest.fixture(scope="session")
def structural_KE(structural_problem):
    import torch
    _, mat_prop, _, _, _ = structural_problem
    mesh = structural_problem[0]
    return torch.tensor(
        hex_element_stiffness.hex8_stiffness_matrix_structural(
            mat_prop.youngs_modulus, mat_prop.poissons_ratio, mesh.elem_size
        )
    )


@pytest.fixture(scope="session")
def thermal_problem():
    """A small thermal TO problem, same edofMat caveat as structural_problem."""
    mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(
        ThermalTOExamples.HeatPlate, nDOFDesired=THERMAL_NDOF
    )
    mesh.createEdofMatThermal()
    mesh.edofMat = mesh.edofMatThermal
    return mesh, mat_prop, bc, elem_body_force, to_params


@pytest.fixture
def thermal_fe_solver(thermal_problem):
    mesh, mat_prop, bc, elem_body_force, _ = thermal_problem
    return hex_thermal_fea.HexThermalFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc,
        solver=linear_solvers.Solvers.SPSOLVE,
        dsolver=deflation.DeflationSolver(use_gpu=False),
        elem_body_force=elem_body_force,
    )


@pytest.fixture(scope="session")
def thermal_KE(thermal_problem):
    import torch
    mesh, mat_prop, _, _, _ = thermal_problem
    return torch.tensor(
        hex_element_stiffness.hex8_stiffness_matrix_thermal(
            mat_prop.thermal_conductivity, mesh.elem_size
        )
    )
