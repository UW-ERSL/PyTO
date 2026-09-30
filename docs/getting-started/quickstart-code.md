# Quick start: code

This page runs a complete topology optimization from Python in about 30 lines, and explains each step. It is for anyone who will use PyTO as a library.

The problem is a cantilever beam: clamped on the left, loaded at the right tip. PyTO finds where to put 30 % of the material to make it as stiff as possible.

## The script
This is [`docs/examples/quickstart.py`](../examples/quickstart.py). Run it from the repository root with `python docs/examples/quickstart.py`.

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import pyvista
pyvista.OFF_SCREEN = True                      # save images instead of opening windows

import pyto.autodiff.sparse_solve as sparse_solve
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.topopt.common import TO_QOI
from pyto.topopt.drivers.mma import topopt_mma

# 1. A ready-made problem: mesh, material, supports/loads, default optimization settings
mesh, material, bc, body_force, to_params = getStructuralTOProblem(StructuralTOExamples.CantileverTipLoad,
                                                                   nDOFDesired=5000)

# 2. The FE solver (SPSOLVE works everywhere; PARDISO is faster where installed)
fe = HexStructuralFEA(mesh=mesh, mat_prop=material, bc=bc, solver=sparse_solve.Solvers.SPSOLVE,
                      elem_body_force=body_force)

# 3. What to optimize: minimum compliance (maximum stiffness) with at most 30 % material
to_params.Objective = (TO_QOI.COMPLIANCE, None)
to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]

# 4. Optimize with MMA (gradients by automatic differentiation)
u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params, maxMMAIterations=30,
                                                 print_progress=False)
print(f"compliance {history['objective'][-1]:.4g} J, volume fraction {history['volfrac'][-1]:.3f}, "
      f"{n_fea} FE solves, {message}")

# 5. Save a picture of the design (the density is stored on the mesh)
out = os.path.join(os.getcwd(), "quickstart_topology.png")
fe.plot_mesh(save_path=out, plot_bc=None, title=None, camera_position="xy")
print("saved", out)
```

Expected output (the numbers vary slightly between machines):
```
compliance 19 J, volume fraction 0.300, 30 FE solves, No errors.
```

![Optimized cantilever](../images/quickstart-cantilever.png)

## What each step does
1. **The problem.** `getStructuralTOProblem` builds one of PyTO's benchmark problems. It returns:
   - the **mesh** (voxels made from an STL);
   - the **material** (steel);
   - the **boundary conditions** `bc`: fixed dofs and the load vector;
   - an optional **body force** (`None` here);
   - default optimization settings (`to_params`).

   `nDOFDesired` sets the mesh size, roughly the number of unknowns. How to build all of this yourself: [A problem from scratch](../code/problem-from-scratch.md).
2. **The FE solver.** `HexStructuralFEA` assembles and solves linear elasticity on the mesh. `fe.solve(x)` returns the displacements for a density `x`, and PyTorch can differentiate through that solve. See [Concepts](../code/concepts.md).
3. **The formulation.** An objective and a list of constraints, as `(type, parameter[, limit])` tuples. Compliance is the work done by the load, uᵀKu; lower means stiffer. The same can be written as formulas; see [Formulations with specs](../code/formulation-with-spec.md) and [Expressions](../code/expressions.md).
4. **The optimizer.** `topopt_mma` runs the Method of Moving Asymptotes (MMA). It returns:
   - `u`: the displacement of the final design;
   - `history`: objective and volume fraction per iteration;
   - `success`, `message`: whether it finished cleanly;
   - `n_fea`: the number of FE solves.

   See [Running the optimizers](../code/running-optimizers.md).
5. **The result.** The final density is stored on the mesh (`mesh.elemPseudoDensity`). `plot_mesh` draws the solid elements. More options: [Fields and plots](../postprocessing/fields-and-plots.md).

## Try next
- Change `0.3` to `0.2` for a lighter design, or `StructuralTOExamples.CantileverTipLoad` to `StructuralTOExamples.MBBBeam`.
- Minimize the tip deflection instead of compliance: [Expressions](../code/expressions.md).
- Write your own objective in Python: [User-defined functions](../code/user-defined-functions.md).

---
Next: [Concepts](../code/concepts.md) · [Back to contents](../README.md)
