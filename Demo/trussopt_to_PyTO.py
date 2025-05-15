import numpy as np
import pandas as pd
import hex_structural_fea
# run the trussopt.py script to generate the truss_output.csv file
# ToDo: integrate the trussopt.py directly in the TO. Eliminate the CSV file. The current code is being tested for the CantileverMidLoad example.
def get_2d_rho(fe_solver: hex_structural_fea.HexStructuralFEA)-> np.ndarray: 
    """
    Rasterize the truss members to a 2D density grid.
    """
    # Load the truss data
    data = pd.read_csv("truss_output.csv")  # header should be: x1,y1,x2,y2,area

    nx, ny, nz = fe_solver.mesh.grid

    # Initialize 2D density grid
    rho2D = np.zeros((ny, nx))  
    dx, dy, dz = fe_solver.mesh.elem_size


    # Rasterize each member to the 2D grid
    for i in range(len(data)):
        x1, y1, x2, y2, A = data.iloc[i]

        length = np.hypot(x2 - x1, y2 - y1)
        N = max(10, int(5 * length))  # number of sample points
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)

        for x, y in zip(xs, ys):
            ix = int(np.clip(np.floor(x / dx), 0, nx - 1))
            iy = int(np.clip(np.floor(y / dy), 0, ny - 1))

            rho2D[iy, ix] = max(rho2D[iy, ix], A)

    # Normalize density
    rho2D /= np.max(rho2D) + 1e-8  # scale to [0, 1]
    return rho2D

def get_3D_rho_from_2D(fe_solver: hex_structural_fea.HexStructuralFEA)-> np.ndarray:
    # Load the 2D density grid
    rho2D = get_2d_rho(fe_solver)

    # Extrude to 3D
    nz = fe_solver.mesh.grid[2]  # number of layers in z direction
    rho3D = np.repeat(rho2D[:, :, np.newaxis], nz, axis=2)  # shape: (39, 57, 3) 

    rho_flat = rho3D.flatten(order='F')  # shape: (6633,)
    assert rho_flat.shape[0] == fe_solver.mesh.num_elems  

    # Final rho for 3D problems to initialize TO
    x = rho_flat.copy()
    return x
