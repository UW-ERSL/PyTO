import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hex_mesher
from topopt_benchmarks import StructuralTOExamples

from truss_3D_opt_examples import *
from trussopt_3D_pyvista import *
# run the trussopt.py script to generate the truss_output.csv file
# ToDo: integrate the trussopt.py directly in the TO. Eliminate the CSV file. The current code is being tested for the CantileverMidLoad example.
# May 17, 2025 Removed dependency on CSV file. The trussopt.py is now integrated with PyTO.
def get_3d_rho_from_truss_output(Nd, Cn, a, mesh, truss_width, truss_height, truss_depth, target_volfrac: float, use_binary_fill: bool = True, threshold=1e-4):
    """
    Rasterize optimized truss bars (Nd, Cn, a) into a 3D density grid using the mesh.

    Parameters:
    - Nd: (num_nodes, 3) array of node coordinates
    - Cn: (num_bars, 4) connectivity array (n1, n2, length, active_flag)
    - a:  (num_bars,) array of bar cross-sectional areas
    - mesh: HexMesher object with attributes .grid, .elem_size, .bbox
    - truss_dims: tuple of (truss_width, truss_height, truss_depth)
    - target_volfrac: desired volume fraction for output density
    - use_binary_fill: whether to use binary or smooth radial fill
    - threshold: minimum bar area to include

    Returns:
    - rho3D: (nz, ny, nx) NumPy array of density field
    """

    nx, ny, nz = mesh.grid
    dx, dy, dz = mesh.elem_size
    dx, dy, dz = float(dx), float(dy), float(dz)

    # Determine scale factors from truss domain to mesh domain
    scale_x = (mesh.bbox.x.max - mesh.bbox.x.min) / truss_width
    scale_y = (mesh.bbox.y.max - mesh.bbox.y.min) / truss_height
    scale_z = (mesh.bbox.z.max - mesh.bbox.z.min) / truss_depth
    scale_vol = scale_x * scale_y * scale_z

    # Target volume
    V_total = nx * ny * nz * dx * dy * dz
    V_target = target_volfrac * V_total

    bar_vol = np.sum([a[i] * Cn[i, 2] for i in range(len(a)) if a[i] >= threshold])
    bar_vol_scaled = bar_vol * scale_vol
    tk = V_target / (bar_vol_scaled + 1e-8)

    # Initialize density grid
    rho3D = np.zeros((nz, ny, nx))

    for i in range(len(Cn)):
        if a[i] < threshold:
            continue
        
        n1, n2 = int(Cn[i, 0]), int(Cn[i, 1])
        x1, y1, z1 = Nd[n1]
        x2, y2, z2 = Nd[n2]
        A = a[i]

        # Rescale to mesh domain
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        z1, z2 = z1 * scale_z, z2 * scale_z

        length = np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])
        N = max(10, int(5 * length / min(dx, dy, dz)))
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)
        zs = np.linspace(z1, z2, N)

        n_voxels = a[i] * Cn[i, 2] * tk  # volume estimate
        radius = int(round((3 * n_voxels / (4 * np.pi)) ** (1/3)))

        for x, y, z in zip(xs, ys, zs):
            ix = int(np.clip(np.floor(x / dx), 0, nx - 1))
            iy = int(np.clip(np.floor(y / dy), 0, ny - 1))
            iz = int(np.clip(np.floor(z / dz), 0, nz - 1))

            for dx_off in range(-radius, radius + 1):
                for dy_off in range(-radius, radius + 1):
                    for dz_off in range(-radius, radius + 1):
                        iix = ix + dx_off
                        iiy = iy + dy_off
                        iiz = iz + dz_off
                        if 0 <= iix < nx and 0 <= iiy < ny and 0 <= iiz < nz:
                            if use_binary_fill:
                                rho3D[iiz, iiy, iix] = 1
                            else:
                                dist = np.sqrt(dx_off**2 + dy_off**2 + dz_off**2)
                                weight = max(0, 1 - dist / (radius + 1e-8))
                                rho3D[iiz, iiy, iix] = max(rho3D[iiz, iiy, iix], A * weight)

    rho3D /= np.max(rho3D) + 1e-8  # normalize
    return rho3D


def normalize_rho_to_exact_volfrac(rho3D: np.ndarray, target_volfrac: float, max_iter=3) -> np.ndarray:
    """
    Normalize rho3D to match an exact target volume fraction within [0, 1].
    If clipping prevents exact match, apply a fallback mass spreading.
    """
    total_elems = rho3D.size
    target_mass = target_volfrac * total_elems

    for _ in range(max_iter):
        rho3D *= target_volfrac / (np.mean(rho3D) + 1e-8)
        rho3D = np.clip(rho3D, 0.0, 1.0)

        current_mass = np.sum(rho3D)
        excess = current_mass - target_mass

        if abs(excess) < 1e-4:
            break

        unclipped = (rho3D < 1.0)
        num_unclipped = np.sum(unclipped)

        if num_unclipped > 0:
            adjustment = excess / (num_unclipped + 1e-8)
            rho3D[unclipped] -= adjustment
            rho3D = np.clip(rho3D, 0.0, 1.0)

    # Final check — if still off, spread residual uniformly
    mass_error = np.sum(rho3D) - target_mass
    if abs(mass_error) > 1e-4:
        rho3D += (target_mass - np.sum(rho3D)) / total_elems
        rho3D = np.clip(rho3D, 0.0, 1.0)

    assert abs(np.mean(rho3D) - target_volfrac) < 1e-4, \
        f"Final volume fraction off: got {np.mean(rho3D)}, expected {target_volfrac}"

    return rho3D


def get_3D_rho_from_2D(to_problem: StructuralTOExamples, mesh: hex_mesher.HexMesher, target_volfrac, use_binary_fill: bool = False, b_plot: bool = False)-> np.ndarray:
    
    truss_width, truss_height, truss_depth  = get_truss_width_height(mesh)
    
    truss_example, poly, Nd, dof, f = get_trussopt_3D_example(to_problem, truss_width, truss_height, truss_depth)
    print('open edges', poly.n_open_edges)
    #plot_3d_domain_with_bconditions(poly, Nd, dof, f)

    Nd, Cn, a, q = trussopt3D(truss_example, poly, Nd, dof, f, st = 1, sc =1, jc = 0.1, b_plot=False)
    if b_plot:
        plot_truss_with_domain_and_bcs(poly, Nd, dof, f, Cn, a, q, threshold=1e-4, title="Truss and Domain", update=False)

    rho3D = get_3d_rho_from_truss_output(Nd, Cn, a, mesh, truss_width, truss_height, truss_depth, target_volfrac = target_volfrac, use_binary_fill = use_binary_fill, threshold = max(a) * 1e-3)
    print("rho3D stats: min =", rho3D.min(), "max =", rho3D.max(), "mean =", rho3D.mean())
    #if b_plot:
        #plot_rho3D(rho3D, mesh, 0.5)
    
    # Final rho for 3D problems to initialize TO
    rho_flat = normalize_rho_to_exact_volfrac(rho3D.reshape(-1), target_volfrac=target_volfrac)
    x = rho_flat.copy()

    print("Actual volume fraction 3D:", np.mean(x))
    return x

def plot_rho3D(rho3D, mesh, threshold=None):
    """
    Visualize a 3D density field using PyVista.

    Parameters:
    - rho3D: 3D numpy array shaped (nz, ny, nx)
    - mesh: object with grid dims, element size, and bounding box
    - threshold: if given, only plot elements with density >= threshold.
                 if None, render full density as a volume field.
    """

    nx, ny, nz = mesh.grid
    dx, dy, dz = mesh.elem_size
    ox, oy, oz = mesh.bbox.x.min, mesh.bbox.y.min, mesh.bbox.z.min

    # Convert rho3D to (nx, ny, nz) format
    rho3D_plot = rho3D.transpose(2, 1, 0)
    assert rho3D_plot.shape == (nx, ny, nz)

    # Create uniform grid
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
        origin=(ox, oy, oz)
    )

    # Flatten and assign to cell data
    flattened = rho3D_plot.flatten(order='F')
    assert flattened.size == grid.n_cells
    grid.cell_data['density'] = flattened

    # Initialize plotter
    p = pv.Plotter()

    if threshold is not None:
        # Threshold-based surface extraction
        thresholded = grid.threshold(value=threshold, scalars='density', preference='cell')
        p.add_mesh(thresholded, scalars='density', cmap='viridis', show_scalar_bar=True,
                   scalar_bar_args={'title': f'Density ≥ {threshold}'})
    else:
        point_grid = grid.cell_data_to_point_data()
        contours = point_grid.contour(isosurfaces=[0.2, 0.4, 0.6])
        p.add_mesh(contours, cmap='viridis', scalar_bar_args={'title': 'Density'})


    p.add_axes()
    p.show_grid()
    p.show()

if __name__ == "__main__":    
	from topopt_benchmarks import *
	import hex_structural_fea
	print("-" * 50)
	to_problem = StructuralTOExamples.TwoBar # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
    
	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
	#mesh.plot()
	rho_flat = get_3D_rho_from_2D(to_problem, mesh, target_volfrac=0.5, use_binary_fill = False, b_plot = True)  
	x = rho_flat
	mesh.setPseudoDensity(x)

	fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = None,
				dsolver = None,
				rtol = 1e-8,
        		elem_body_force = elem_body_force)
    
	fe_solver.plot_pseudo_density(auto_close = True, title = f"Initial Density")
