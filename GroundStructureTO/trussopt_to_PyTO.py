import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hex_mesher
from topopt_benchmarks import StructuralTOExamples
from trussopt import *
# run the trussopt.py script to generate the truss_output.csv file
# ToDo: integrate the trussopt.py directly in the TO. Eliminate the CSV file. The current code is being tested for the CantileverMidLoad example.
# May 17, 2025 Removed dependency on CSV file. The trussopt.py is now integrated with PyTO.
def get_2d_rho_from_truss_output(Nd, Cn, a, mesh: hex_mesher.HexMesher, truss_width, truss_height, target_volfrac: float, use_binary_fill: bool = True, threshold=1e-4):
    """
    Rasterize the optimized truss members (with given Nd, Cn, a) to a 2D density grid.
    
    Parameters:
    - Nd: Node coordinates - list of nodes (each row = [x,y])
    - Cn: Connectivity matrix - 
            [ start_node_index, end_node_index, length, is_active (0 or 1) ]
    - a: Member areas - optimal area for each row in Cn.
    - q: Member forces - optimal force for each row in Cn. q>0 means tension, q<0 means compression.
    - mesh: hex_mesher.HexMesher object
    - threshold: minimum area to include a bar in the rasterization

    Returns:
    - rho2D: (ny, nx) NumPy array representing rasterized density grid
    """

    nx, ny, nz = mesh.grid
    dx, dy, dz = mesh.elem_size
    elem_area = dx * dy
    # Determine physical size of mesh
    mesh_width = mesh.bbox.x.max - mesh.bbox.x.min
    mesh_height = mesh.bbox.y.max - mesh.bbox.y.min

    # Compute scaling factor to map truss domain into mesh domain
    scale_x = mesh_width / truss_width
    scale_y = mesh_height / truss_height

    print("Mesh domain size:", mesh.bbox.x, mesh.bbox.y)
    print(f"Truss width: {truss_width}, Mesh width: {mesh_width}, scale_x: {scale_x}")
    print(f"Truss height: {truss_height}, Mesh height: {mesh_height}, scale_y: {scale_y}")

    scale_area = scale_x * scale_y

    A_max = np.max(a)
    bar_area_max = A_max * scale_area
    
    # Total area in mesh
    V_total = nx * ny * nz * dx * dy * dz
    V_target = target_volfrac * V_total
    A_target = (V_target / nz) / dz  # effective target area for one 2D slice

    bar_total = np.sum([a[i] * Cn[i, 2] for i in range(len(a)) if a[i] >= threshold])
    bar_total_scaled = bar_total * scale_x * scale_y
    tk = A_target / (bar_total_scaled + 1e-8)

    print(f"Target area: {A_target:.2f}, Truss bar area scaled: {bar_total_scaled:.2f}, tk = {tk:.4f}")

    # Initialize 2D density grid
    rho2D = np.zeros((ny, nx))

    
    for i in range(len(Cn)):
        if a[i] < threshold:
            continue  # skip insignificant members
        
        n1, n2 = int(Cn[i, 0]), int(Cn[i, 1])
        x1, y1 = Nd[n1]
        x2, y2 = Nd[n2]
        A = a[i]

        # Rescale from truss domain to mesh domain
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        length = np.hypot(x2 - x1, y2 - y1)
        N = max(10, int(5 * length / min(dx, dy)))
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)

        n_cells = a[i] * Cn[i, 2] * tk  # area × length × tk
        radius = int(round(np.sqrt(n_cells / np.pi)))

        
        for x, y in zip(xs, ys):
            ix = int(np.clip(np.floor(x / dx), 0, nx - 1))
            iy = int(np.clip(np.floor(y / dy), 0, ny - 1))

            if use_binary_fill:
                for dx_off in range(-radius, radius + 1):
                    for dy_off in range(-radius, radius + 1):
                        iix = ix + dx_off
                        iiy = iy + dy_off
                        if 0 <= iix < nx and 0 <= iiy < ny:
                            rho2D[iiy, iix] = 1.0
            else:
                #radius = int(np.ceil(np.sqrt(A) / min(dx, dy)))
                for dx_off in range(-radius, radius + 1):
                    for dy_off in range(-radius, radius + 1):
                        iix = ix + dx_off
                        iiy = iy + dy_off
                        if 0 <= iix < nx and 0 <= iiy < ny:
                            dist = np.hypot(dx_off, dy_off)
                            weight = max(0, 1 - dist / (radius + 1e-8))
                            rho2D[iiy, iix] = max(rho2D[iiy, iix], A * weight)


    rho2D /= np.max(rho2D) + 1e-8  # normalize to [0, 1]
    #plt.plot([x1, x2], [y1, y2], color='r')  # Bar segment
    #plt.imshow(rho2D, origin='lower', cmap='gray')
    return rho2D

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
    print("Truss opt initialization")

    trussopt_problem = get_trussopt_problem(to_problem)
    truss_width, truss_height = get_truss_width_height(mesh)
    Nd, Cn, a, q = trussopt(trussopt_problem, truss_width, truss_height, b_plot=True) #much larger jc value if you want only a handful of members in the final design.
    rho2D = get_2d_rho_from_truss_output(Nd, Cn, a, mesh, truss_width, truss_height, target_volfrac = target_volfrac, use_binary_fill = use_binary_fill, threshold = max(a) * 1e-3)
    if b_plot:
        plotTruss(Nd, Cn, a, q, max(a) * 1e-3, "Finished", False)
        #plot_rho2D(rho2D, mesh)
    nx, ny, nz = mesh.grid

    rho_flat = np.zeros(mesh.num_elems)

    for i in range(mesh.num_elems):
        elem_center = mesh.elem_centers[i]
        x, y = elem_center[0], elem_center[1]
        # scale x, y to index in rho2D
        ix = int(np.clip(np.floor((x - mesh.bbox.x.min) / mesh.elem_size[0]), 0, rho2D.shape[1]-1))
        iy = int(np.clip(np.floor((y - mesh.bbox.y.min) / mesh.elem_size[1]), 0, rho2D.shape[0]-1))
        rho_flat[i] = rho2D[iy, ix]
    # shape: (6633,)
    assert rho_flat.shape[0] == mesh.num_elems  

    # Compare first layer
    # plt.imshow(rho3D[:, :, 0], cmap='viridis', origin='lower')
    # plt.title("rho3D[:, :, 0] — first Z slice")
    # plt.show()
    
    # Final rho for 3D problems to initialize TO
    rho_flat = normalize_rho_to_exact_volfrac(rho_flat, target_volfrac=target_volfrac)
    x = rho_flat.copy()

    print("Actual volume fraction 3D:", np.mean(x))
    return x

def get_truss_width_height(mesh: hex_mesher.HexMesher):
    """
    Get the width and height of the truss from the mesh.
    
    Parameters:
    - mesh: hex_mesher.HexMesher object
    
    Returns:
    minimum value of 10 for the smallest dimension so that the truss grid can have unit spacing.
    - truss_width: width of the truss
    - truss_height: height of the truss
    """
    # Determine physical size of mesh
    mesh_width = mesh.bbox.x.max - mesh.bbox.x.min
    mesh_height = mesh.bbox.y.max - mesh.bbox.y.min

    # Determine scale factor so that smallest dimension becomes 10
    scale_factor = 10.0 / min(mesh_width, mesh_height)

    # Compute desired truss domain size
    truss_width = int(np.ceil(mesh_width * scale_factor))
    truss_height = int(np.ceil(mesh_height * scale_factor))
    return truss_width, truss_height

def get_trussopt_problem(to_problem: StructuralTOExamples):
    if to_problem == StructuralTOExamples.CantileverMidLoad:
        trussopt_problem = TrussOptExamples.CantileverMidLoad
    elif to_problem == StructuralTOExamples.CantileverTipLoad:
        trussopt_problem = TrussOptExamples.CantileverTipLoad
    elif to_problem == StructuralTOExamples.Mitchell_1:
        trussopt_problem = TrussOptExamples.Mitchell_1
    elif to_problem == StructuralTOExamples.ShortCantileverTipLoad:
        trussopt_problem = TrussOptExamples.ShortCantileverTipLoad
    elif to_problem == StructuralTOExamples.ShortCantileverMidLoad:
        trussopt_problem = TrussOptExamples.ShortCantileverMidLoad
    elif to_problem == StructuralTOExamples.TwoBar:
        trussopt_problem = TrussOptExamples.TwoBar
    elif to_problem == StructuralTOExamples.MBBB:
        trussopt_problem = TrussOptExamples.MBBB
    elif to_problem == StructuralTOExamples.DistributedLoad:
        trussopt_problem = TrussOptExamples.DistributedLoad
    elif to_problem == StructuralTOExamples.Multiload:
        trussopt_problem = TrussOptExamples.Multiload
    elif to_problem == StructuralTOExamples.LBracketMidLoad:
        trussopt_problem = TrussOptExamples.LBracketMidLoad
    else:
        raise ValueError(f"Unknown example: {to_problem}")
    return trussopt_problem

def plot_rho2D(rho2D, mesh: hex_mesher.HexMesher):
    """
    Plot the 2D density grid with physical aspect ratio from mesh.
    
    Parameters:
    - rho2D: 2D numpy array (ny, nx)
    - mesh: provides nx, ny, dx, dy for setting extent
    """
    import matplotlib.pyplot as plt

    nx, ny, _ = mesh.grid
    dx, dy, _ = mesh.elem_size

    width = nx * dx
    height = ny * dy

    plt.figure(figsize=(10, 5))
    plt.imshow(
        rho2D, 
        cmap='viridis',
        origin='lower',
        extent=[0, width, 0, height],
        aspect='auto'
    )
    plt.colorbar(label='Density')
    plt.title("2D Density Grid from Ground Structure")
    plt.xlabel("X (physical units)")
    plt.ylabel("Y (physical units)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
	from topopt_benchmarks import *

	print("-" * 50)
	to_problem = StructuralTOExamples.CantileverMidLoad # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

    
	rho_flat = get_3D_rho_from_2D(mesh, use_binary_fill = True, b_plot = True)  
	x = rho_flat
	#mesh.setPseudoDensity(x)
	#fe_solver.plot_pseudo_density(auto_close = False, title = f"Initial Density")
