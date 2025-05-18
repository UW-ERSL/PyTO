import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hex_mesher

from trussopt import *
# run the trussopt.py script to generate the truss_output.csv file
# ToDo: integrate the trussopt.py directly in the TO. Eliminate the CSV file. The current code is being tested for the CantileverMidLoad example.
# May 17, 2025 Removed dependency on CSV file. The trussopt.py is now integrated with PyTO.
def get_2d_rho_from_structural_output(Nd, Cn, a, mesh: hex_mesher.HexMesher, use_binary_fill: bool, threshold=1e-4):
    """
    Rasterize the optimized truss members (with given Nd, Cn, a) to a 2D density grid.
    
    Parameters:
    - Nd: (n_nodes, 2) array of node coordinates
    - Cn: (n_members, 2 or 4) array of member connections and lengths
    - a:  (n_members,) array of cross-sectional areas
    - mesh: hex_mesher.HexMesher object
    - threshold: minimum area to include a bar in the rasterization

    Returns:
    - rho2D: (ny, nx) NumPy array representing rasterized density grid
    """
    import numpy as np

    nx, ny, nz = mesh.grid
    dx, dy, dz = mesh.elem_size
    elem_area = dx * dy
    
    scale_x = (nx * dx) / 20.0 # harde coded to 20.0 which is the domain of the truss, change needed
    scale_y = (ny * dy) / 10.0
    scale_area = scale_x * scale_y

    A_max = np.max(a)
    bar_area_max = A_max * scale_area
    
    N_max = bar_area_max / elem_area # Compute how many elements it should cover
    tk = N_max*1 / (A_max + 1e-8)
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

        n_cells = a[i] * tk # how many elements this bar should fill
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
    return rho2D

def get_3D_rho_from_2D(mesh: hex_mesher.HexMesher, use_binary_fill: bool = False, b_plot: bool = False)-> np.ndarray:
    
    Nd, Cn, a, q = trussopt(width = 20, height = 10, st = 1, sc =1, jc = 1) #much larger jc value if you want only a handful of members in the final design.
    rho2D = get_2d_rho_from_structural_output(Nd, Cn, a, mesh, use_binary_fill = use_binary_fill, threshold = max(a) * 1e-3)
    if b_plot:
        plotTruss(Nd, Cn, a, q, max(a) * 1e-3, "Finished", False)
        plot_rho2D(rho2D, mesh)
    nx, ny, nz = mesh.grid

    rho3D = np.zeros((nx, ny, nz))  # shape = (nx, ny, nz)
    for k in range(nz):
        rho3D[:, :, k] = rho2D.T  # transpose to match (nx, ny) from (ny, nx)

    rho_flat = rho3D.flatten(order='F')  # shape: (6633,)
    assert rho_flat.shape[0] == mesh.num_elems  

    # Compare first layer
    # plt.imshow(rho3D[:, :, 0], cmap='viridis', origin='lower')
    # plt.title("rho3D[:, :, 0] — first Z slice")
    # plt.show()
    #  # Compare second layer
    # plt.imshow(rho3D[:, :, 1], cmap='viridis', origin='lower')
    # plt.title("rho3D[:, :, 0] — second Z slice")
    # plt.show()
    #  # Compare second layer
    # plt.imshow(rho3D[:, :, 2], cmap='viridis', origin='lower')
    # plt.title("rho3D[:, :, 0] — third Z slice")
    # plt.show()

    # Final rho for 3D problems to initialize TO
    x = rho_flat.copy()
    return x

def plot_truss_from_csv(data, dx=1.0, dy=1.0):
    """
    Plot the truss members from CSV data on the same 2D domain as the density grid.
    
    Parameters:
    - data: DataFrame containing columns [x1, y1, x2, y2, area]
    - dx, dy: element sizes for scaling (optional)
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Plot each member as a line
    for _, row in data.iterrows():
        x1, y1, x2, y2, A = row
        lw = 0.5 + 3 * A / data['area'].max()  # line width scaled by area
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=lw)
    
    ax.set_aspect('equal')
    plt.title("Truss Layout from Ground Structure Optimization")
    plt.xlabel("X (physical units)")
    plt.ylabel("Y (physical units)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
