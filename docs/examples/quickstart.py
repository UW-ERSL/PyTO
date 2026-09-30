"""Quick start: optimize a benchmark cantilever for stiffness at 30 % volume and save the result.

Run from the repository root:  python docs/examples/quickstart.py
"""
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
