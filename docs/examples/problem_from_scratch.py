"""A topology optimization problem built from scratch in code: a clamped box beam with a tip load.

Every object is created explicitly (mesh, material, boundary conditions, FE solver, formulation), then MMA
minimizes the tip deflection with at most 40 % material. Run from the repository root:

    python docs/examples/problem_from_scratch.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import pyvista
pyvista.OFF_SCREEN = True

import pyto.autodiff.sparse_solve as sparse_solve
import pyto.core.bc as bound_cond
import pyto.core.mat_lib as mat_lib
from pyto.core.hex_mesher import HexMesher
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.topopt.drivers.mma import topopt_mma
from pyto.topopt.spec import ConstraintSpec, MethodSpec, ObjectiveSpec, OptimizationSpec, compile_spec, validate

# ---------------------------------------------------------------- 1. mesh: 2.0 x 0.5 x 0.1 m box of hexahedra
L = (2.0, 0.5, 0.1)
n = (48, 12, 2)                                   # elements per direction
mesh = HexMesher()
mesh.grid_mesh(num_elems=n, elem_size=tuple(L[i] / n[i] for i in range(3)))
mesh.createEdofMatStructural()                    # element -> dof connectivity (3 dofs per node)
xyz = mesh.node_xyz                               # (num_nodes, 3) node coordinates in m

# ---------------------------------------------------------------- 2. material
steel = mat_lib.get_material("Steel")             # E = 210 GPa, nu = 0.3, rho = 7850 kg/m^3
# custom = mat_lib.create_material_with_defaults("MyAlloy", youngs_modulus=70e9, yield_strength=250e6)

# ---------------------------------------------------------------- 3. supports and loads
# dof numbering: node i has dofs 3*i (x), 3*i+1 (y), 3*i+2 (z)
clamped = mesh.getNodesOnBoundingBoxPlane(0, True)                   # all nodes on the x = 0 face
fixed_dofs = np.concatenate([3 * clamped, 3 * clamped + 1, 3 * clamped + 2])

tip = np.where((np.isclose(xyz[:, 0], L[0])) & (np.abs(xyz[:, 1] - L[1] / 2) < 0.03))[0]   # mid-height, x = L
force = np.zeros(3 * mesh.num_nodes)
force[3 * tip + 1] = -1000.0 / len(tip)                                                  # 1 kN total, -y

bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=np.zeros(len(fixed_dofs)))
mesh.node_indices[clamped, 3] = 1                 # markers only used to draw supports (1) and loads (2)
mesh.node_indices[tip, 3] = 2

# ---------------------------------------------------------------- 4. FE solver
fe = HexStructuralFEA(mesh=mesh, mat_prop=steel, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)

# ---------------------------------------------------------------- 5. formulation (the same data the GUI edits)
selections = {"Tip": {"nodes": tip}}              # names usable inside expressions
spec = OptimizationSpec(
    objective=ObjectiveSpec("Displacement(Tip, magnitude, pnorm) / 1e-4", "minimize"),   # peak tip deflection
    constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.4)],
    method=MethodSpec("MMA", max_iterations=40))
for issue in validate(spec, "structural", selections):
    print(issue)                                  # errors stop compile_spec; warnings are advice
to_params = compile_spec(spec, selections)

# ---------------------------------------------------------------- 6. optimize and look at the result
u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params, maxMMAIterations=40,
                                                 print_progress=False)
tip_deflection = np.abs(u.reshape(-1, 3)[tip, 1]).max()
print(f"peak tip deflection {tip_deflection * 1e3:.3f} mm, volume fraction {history['volfrac'][-1]:.3f}")

out = os.path.join(os.getcwd(), "from_scratch_topology.png")
fe.plot_mesh(save_path=out, plot_bc=None, title=None, camera_position="xy")
print("saved", out)
