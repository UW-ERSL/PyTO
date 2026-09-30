"""A new optimization method written against PyTO's driver contract: projected steepest descent.

It solves  minimize f(x)  subject to  mean(x) <= V,  0 <= x <= 1  with any objective PyTO can differentiate
(built-in, expression or Python function): gradient step, move limit, then a bisection on a shift that projects
the step back onto the volume budget. It is deliberately simple; it shows the pieces every driver needs:

  * setup_driver_state()   -> density filter H, element stiffness KE
  * the density chain      x -> H x / Hs -> fe.solve -> objective, differentiated by autograd
  * the return contract    (solution, history, success, message, n_fea), history["objective"], ["volfrac"], ...

Run from the repository root:  python docs/examples/custom_driver.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import compute_objective_and_gradient
from pyto.topopt.common import TO_QOI
from pyto.topopt.drivers._shared import setup_driver_state, torch_sparse_filter


def topopt_projected_gradient(fe_solver, to_params, maxIterations: int = 60, move: float = 0.1,
                              print_progress: bool = True, iteration_callback=None):
    """Projected steepest descent. Needs exactly one constraint: (TO_QOI.VOLUME_FRACTION, None, V)."""
    (ctype, _, V), = to_params.Constraints
    assert ctype == TO_QOI.VOLUME_FRACTION, "this demo driver handles one volume-fraction constraint"
    mm = MaterialModel.SIMP
    state = setup_driver_state(fe_solver, to_params)          # filter, forced elements, KE, body force
    H, Hs = torch_sparse_filter(state["H"], state["Hs"])
    filtered_mean = lambda z: float(np.mean(state["H"] @ z / np.ravel(state["Hs"])))   # volume of the physical design
    KE = torch.tensor(state["KE"])

    x = np.full(fe_solver.mesh.num_elems, V)                   # start from a uniform design
    history = {"objective": [], "volfrac": [], "constraint_1": []}
    sol = None
    for it in range(maxIterations):
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        x_phys = torch.sparse.mm(H, x_t.unsqueeze(1)).squeeze(1) / Hs     # density filter
        fe_solver.mesh.setPseudoDensity(x_phys.detach())                    # for plotting
        sol = fe_solver.solve(x_phys, mm)
        f = compute_objective_and_gradient(to_params, sol, x_phys, fe_solver, KE, mm)
        (g,) = torch.autograd.grad(f, x_t)                                   # df/dx through filter and solve
        g = g.numpy() / (np.abs(g.numpy()).max() + 1e-300)

        # gradient step, then find the shift mu that puts the design back on the volume budget
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mu = 0.5 * (lo + hi)
            x_new = np.clip(x - move * g + mu, np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            lo, hi = (mu, hi) if filtered_mean(x_new) < V else (lo, mu)
        change = np.abs(x_new - x).max()
        x = x_new

        history["objective"].append(f.item())
        history["volfrac"].append(x_phys.mean().item())
        history["constraint_1"].append(x_phys.mean().item() / V - 1.0)
        if iteration_callback is not None:
            iteration_callback(history)
        if print_progress:
            print(f"it {it:3d}  objective {f.item():.5g}  volume {x_phys.mean().item():.3f}  change {change:.3f}")
        if change < 1e-3:
            break
    return sol.detach().numpy(), history, True, "No errors.", it + 1


if __name__ == "__main__":
    import pyto.autodiff.sparse_solve as sparse_solve
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA

    mesh, mat, bc, ebf, to_params = getStructuralTOProblem(StructuralTOExamples.CantileverTipLoad, nDOFDesired=5000)
    fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
    to_params.Objective = (TO_QOI.COMPLIANCE, None)
    to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
    u, history, ok, msg, n = topopt_projected_gradient(fe, to_params, maxIterations=60, print_progress=False)
    J = history["objective"]
    print(f"compliance {J[0]:.4g} -> {J[-1]:.4g} J in {n} iterations, volume {history['volfrac'][-1]:.3f}")
