"""User-defined objective and constraint written as Python functions (torch), in two ways:

  A. directly in TOParams:     to_params.Objective = (TO_QOI.GFUNCTION, fn)
  B. through a spec reference: ObjectiveSpec("python:<file.py>::<function>")  (what the GUI stores)

Both are differentiated by autograd through the FE solve; no gradient is written by hand. check_gradient()
confirms the gradient against finite differences before the run. Run from the repository root:

    python docs/examples/user_defined_function.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import torch


def tip_deflection_squared(sol, x, fe_solver):
    """Mean squared displacement along the load direction at the loaded nodes (m^2). Torch operations only."""
    f = torch.as_tensor(fe_solver.bc.force)
    loaded = torch.nonzero(f).flatten()
    direction = torch.sign(f[loaded])
    return ((sol[loaded] * direction) ** 2).mean()


def material_in_left_half(sol, x, fe_solver):
    """Volume fraction of the left half of the model (depends on the design only)."""
    left = torch.as_tensor(fe_solver.mesh.elem_centers[:, 0] < np.median(fe_solver.mesh.elem_centers[:, 0]))
    return x[left].mean()


if __name__ == "__main__":
    import pyto.autodiff.sparse_solve as sparse_solve
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.capabilities import check_gradient
    from pyto.topopt.common import TO_QOI
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec

    mesh, mat, bc, ebf, to_params = getStructuralTOProblem(StructuralTOExamples.CantileverTipLoad, nDOFDesired=5000)
    fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)

    # ---- A. functions straight into TOParams
    to_params.Objective = (TO_QOI.GFUNCTION, tip_deflection_squared)
    to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3),
                             (TO_QOI.GFUNCTION, material_in_left_half, 0.45)]     # fn <= 0.45 (limit > 0)
    for r in check_gradient(to_params, fe):
        print(f"gradient check {r.name}: max relative error {r.max_rel_error:.1e} {'OK' if r.ok else 'MISMATCH'}")
    _, history, ok, msg, n = topopt_mma(fe, to_params=to_params, maxMMAIterations=30, print_progress=False)
    left = (history["constraint_2"][-1] + 1.0) * 0.45     # constraints are stored as value/limit - 1
    print(f"A: objective {history['objective'][-1]:.3e} m^2, left-half material {left:.3f} (limit 0.45)")

    # ---- B. the same objective referenced from a spec (the GUI's "Python function..." type)
    spec = OptimizationSpec(objective=ObjectiveSpec(f"python:{os.path.abspath(__file__)}::tip_deflection_squared"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3)])
    to_params_b = compile_spec(spec, allow_code=True, base=to_params)    # allow_code: loading runs the file
    fe_b = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
    _, history_b, ok, msg, n = topopt_mma(fe_b, to_params=to_params_b, maxMMAIterations=30, print_progress=False)
    print(f"B: objective {history_b['objective'][-1]:.3e} m^2")
