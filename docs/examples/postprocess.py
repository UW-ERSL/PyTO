"""Post-processing an optimized design: convergence plot, stress and deformation, an honest 0/1 evaluation,
VTU export for ParaView and a smooth STL of the result.

Run from the repository root (writes files into the current folder):

    python docs/examples/postprocess.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista
pyvista.OFF_SCREEN = True
import torch

import pyto.autodiff.sparse_solve as sparse_solve
from pyto.autodiff.material_model import MaterialModel
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
from pyto.gui.optimization_setup import final_design_summary
from pyto.io.topopt_stl_recovery import extract_isosurface_cnn, subtract_voids_from_stl
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.topopt.common import TO_QOI
from pyto.topopt.drivers.mma import topopt_mma

OUT = os.getcwd()
mesh, mat, bc, ebf, to_params = getStructuralTOProblem(StructuralTOExamples.CantileverTipLoad, nDOFDesired=5000)
fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
to_params.Objective = (TO_QOI.COMPLIANCE, None)
to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
u, history, ok, msg, n = topopt_mma(fe, to_params=to_params, maxMMAIterations=40, binarize_topology=False,
                                    print_progress=False)

# 1. convergence: history holds one value per iteration; constraints are stored as value/limit - 1 (<= 0 = ok)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(history["objective"], label="compliance (J)")
ax.set_xlabel("iteration"); ax.set_ylabel("objective"); ax.set_yscale("log")
ax2 = ax.twinx(); ax2.plot(history["volfrac"], "C1:", label="volume fraction"); ax2.set_ylabel("volume fraction")
fig.legend(loc="upper right"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "pp_convergence.png"), dpi=120)

# 2. fields of the final (grey) design: the solver keeps its last solve; postprocess() computes strain/stress
fe.postprocess()
fe.plot_deformation(save_path=os.path.join(OUT, "pp_deformation.png"))
fe.plot_vonMisesStress(save_path=os.path.join(OUT, "pp_vonmises.png"))
fe.plot_pseudo_density(save_path=os.path.join(OUT, "pp_density.png"))
for line in final_design_summary(fe, "structural")["lines"]:      # stresses on the solid region only
    print(line)

# 3. honest numbers: threshold to a 0/1 design and solve it again
x01 = torch.as_tensor((np.asarray(mesh.elemPseudoDensity) > 0.5).astype(float))
with torch.no_grad():
    u01 = fe.solve(x01, MaterialModel.SIMP)
print(f"grey design compliance {history['objective'][-1]:.4g} J at volume {history['volfrac'][-1]:.3f}; "
      f"0/1 design {float(np.dot(bc.force, u01.numpy())):.4g} J at volume {x01.mean().item():.3f}")

# 4. VTU for ParaView (density as a field on the hex mesh)
vtu_file = os.path.join(OUT, "pp_design.vtu")
mesh.export_vtu_mesh(elem_field=mesh.elemPseudoDensity, density_field="density", file_name=vtu_file)

# 5. smooth STL: a small CNN smooths the voxel voids, which are then subtracted from the original STL
voids = extract_isosurface_cnn(pyvista.read(vtu_file), isovalue=0.5)
part = subtract_voids_from_stl(mesh.stlMesh.triangulate().compute_normals(), voids)
part.save(os.path.join(OUT, "pp_design.stl"))
print("wrote pp_convergence.png, pp_deformation.png, pp_vonmises.png, pp_density.png, pp_design.vtu, pp_design.stl")
