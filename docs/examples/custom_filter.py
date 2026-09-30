"""A new manufacturing filter: a design that repeats n times along x (periodic cells), plugged in at run time.

Filters in PyTO are sparse matrices multiplied into the density filter H (topopt/common.py, createFilters);
the drivers use x_phys = H x / Hs and autograd differentiates through it, so a new filter needs no gradient
code. This file builds the filter, wraps createFilters to include it, and optimizes a simply supported beam.
The permanent version is a TOParams field plus a branch in createFilters; see docs/extending/add-a-filter.md.

Run from the repository root:  python docs/examples/custom_filter.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
from scipy.sparse import coo_matrix


def createXPeriodicFilter(mesh, n_cells: int):
    """Average every element with its copies in the other n_cells - 1 cells along x."""
    x0, x1 = mesh.bbox.x.min, mesh.bbox.x.max
    period = (x1 - x0) / n_cells
    rows, cols, data = [], [], []
    for i, c in enumerate(mesh.elem_centers):
        k0 = int((c[0] - x0) // period)
        copies = [j for k in range(n_cells)
                  if (j := mesh.get_element_near_point(np.array([c[0] + (k - k0) * period, c[1], c[2]]))) != -1]
        for j in copies:
            rows.append(i)
            cols.append(j)
            data.append(1.0 / len(copies))
    return coo_matrix((data, (rows, cols)), shape=(mesh.num_elems, mesh.num_elems)).tocsc()


PERIODIC_CELLS = 4

# run-time plug-in: every driver builds its filter through pyto.topopt.drivers._shared.createFilters
import pyto.topopt.drivers._shared as shared
from pyto.topopt.common import createFilters


def createFiltersWithPeriodic(fe_solver, to_params):
    H, Hs = createFilters(fe_solver, to_params)
    if PERIODIC_CELLS > 1:
        H = H * createXPeriodicFilter(fe_solver.mesh, PERIODIC_CELLS)
        Hs = np.array(H.sum(1)).squeeze()
    return H, Hs


shared.createFilters = createFiltersWithPeriodic

if __name__ == "__main__":
    import pyvista
    pyvista.OFF_SCREEN = True
    import pyto.autodiff.sparse_solve as sparse_solve
    import pyto.core.bc as bound_cond
    import pyto.core.mat_lib as mat_lib
    from pyto.core.hex_mesher import HexMesher
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.common import TOParams, TO_QOI
    from pyto.topopt.drivers.mma import topopt_mma

    # 3 x 0.5 m beam, supported at the two bottom corners, uniform load on top, in-plane (z fixed)
    mesh = HexMesher()
    mesh.grid_mesh(num_elems=(60, 10, 1), elem_size=(0.05, 0.05, 0.05))
    mesh.createEdofMatStructural()
    xyz = mesh.node_xyz
    bottom = np.where(xyz[:, 1] < 1e-9)[0]
    ends = np.concatenate([mesh.getNodesOnBoundingBoxPlane(0, True), mesh.getNodesOnBoundingBoxPlane(0, False)])
    support = np.intersect1d(ends, bottom)
    fixed = np.unique(np.concatenate([3 * support, 3 * support + 1, 3 * support + 2,
                                      3 * np.arange(mesh.num_nodes) + 2]))
    top = mesh.getNodesOnBoundingBoxPlane(1, False)
    force = np.zeros(3 * mesh.num_nodes)
    force[3 * top + 1] = -1000.0 / len(top)
    bc = bound_cond.BC(force=force, fixed_dofs=fixed, dirichlet_values=np.zeros(len(fixed)))
    fe = HexStructuralFEA(mesh=mesh, mat_prop=mat_lib.get_material("Steel"), bc=bc,
                          solver=sparse_solve.Solvers.SPSOLVE)

    p = TOParams()
    p.Objective = (TO_QOI.COMPLIANCE, None)
    p.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
    _, history, ok, msg, n = topopt_mma(fe, to_params=p, maxMMAIterations=30, binarize_topology=False,
                                        print_progress=False)
    x = mesh.elemPseudoDensity.reshape(10, 60)                                  # (y, x) for this one-layer mesh
    cells = [x[:, 15 * k:15 * (k + 1)] for k in range(PERIODIC_CELLS)]
    print(f"compliance {float(history['objective'][-1]):.4g} J; largest density difference between cells "
          f"{max(np.abs(c - cells[0]).max() for c in cells):.3f}")
    fe.plot_mesh(save_path=os.path.join(os.getcwd(), "periodic_beam.png"), plot_bc=None, title=None,
                 camera_position="xy")
