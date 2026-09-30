"""Add a new built-in quantity ("response") and use it in expressions, without writing a gradient.

The new response here is Grayness(sel): the mean of 4 x (1 - x) over an element selection, 0 for a pure 0/1
design and 1 for x = 0.5 everywhere. As a constraint it pushes the optimizer toward a crisp design.

This file registers it at run time, which is handy for experiments. To make it permanent (and visible in the
GUI), add the same Response(...) to RESPONSES in src/pyto/autodiff/qoi/responses.py; see
docs/extending/add-a-response.md. Run from the repository root:

    python docs/examples/custom_response.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import torch

from pyto.autodiff.qoi import responses
from pyto.autodiff.qoi.responses import COUPLED, STRUCTURAL, THERMAL, Response
from pyto.topopt import expressions


def grayness(sol, x, fe_solver, elements=None):
    """Mean of 4 x (1 - x) over the selected elements (whole model if None). Torch operations only."""
    xs = x if elements is None else x[torch.as_tensor(elements, dtype=torch.long)]
    return (4.0 * xs * (1.0 - xs)).mean()


# 1. register the response (name, function, physics it applies to, parameters, unit, ...)
responses.RESPONSES["Grayness"] = Response(
    "Grayness", grayness, frozenset({STRUCTURAL, THERMAL, COUPLED}),
    params={"elements": "element selection"}, unit="-",
    depends_on_solution=False,                     # design-only: no adjoint solve needed
    description="Mean of 4x(1-x): 0 for a black-and-white design, 1 for all-grey.")
# 2. declare its positional parameters so Grayness(sel) parses (keywords use the same names)
expressions.POSITIONAL["Grayness"] = ("elements",)


if __name__ == "__main__":
    import pyto.autodiff.sparse_solve as sparse_solve
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec, validate

    mesh, mat, bc, ebf, base = getStructuralTOProblem(StructuralTOExamples.CantileverTipLoad, nDOFDesired=5000)
    spec = OptimizationSpec(objective=ObjectiveSpec("Compliance()", "minimize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3),
                                         ConstraintSpec("Grayness()", "<=", 0.22)])
    print(validate(spec, "structural") or "formulation OK")
    for label, s in (("without", OptimizationSpec(objective=spec.objective, constraints=spec.constraints[:1])),
                     ("with", spec)):
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
        _, h, ok, msg, n = topopt_mma(fe, to_params=compile_spec(s, base=base), maxMMAIterations=40,
                                      binarize_topology=False, print_progress=False)
        x = torch.as_tensor(fe.mesh.elemPseudoDensity)
        print(f"{label:7s} grayness constraint: compliance {h['objective'][-1]:.4g} J, "
              f"grayness {grayness(None, x, fe).item():.3f}")
