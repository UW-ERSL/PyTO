import pyvista as pv
import numpy as np
import trimesh
import torch
import torch.nn as nn
from topopt_structural_benchmarks import *


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple 3D Autoencoder-like structure for smoothing
        # Input: 1 channel (density), Output: 1 channel (smoothed density)
        
        self.encoder = nn.Sequential(
            # Layer 1: Input -> 16 channels
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            
            # Layer 2: 16 -> 32 channels
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
        )
        
        self.decoder = nn.Sequential(
            # Layer 3: 32 -> 16 channels
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            
            # Layer 4: Output layer (1 channel)
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Output between 0 and 1
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def extract_isosurface_cnn(vtu, isovalue=0.5):
    print("Starting Direct 3D CNN extraction with PADDING...")
    
    # --- STEP 1: DETECT GRID DIMENSIONS ---
    x_coords = np.unique(vtu.points[:, 0])
    y_coords = np.unique(vtu.points[:, 1])
    z_coords = np.unique(vtu.points[:, 2])
    
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    print(f"Original Grid: {nx} x {ny} x {nz}")
    
    # Calculate spacing (robustly)
    dx = (x_coords[-1] - x_coords[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (y_coords[-1] - y_coords[0]) / (ny - 1) if ny > 1 else 1.0
    dz = (z_coords[-1] - z_coords[0]) / (nz - 1) if nz > 1 else 1.0

    # --- STEP 2: PREPARE DATA (RESHAPE OR RESAMPLE) ---
    if "density" not in vtu.point_data:
        vtu = vtu.cell_data_to_point_data()
    
    raw_density = vtu.point_data["density"]
    expected_size = nx * ny * nz

    #just giving a chk point to see, to make sure
    # Check if the VTU is a perfect grid or sparse/unstructured
    if raw_density.size == expected_size:
        # structured grid
        grid_density = raw_density.reshape((nx, ny, nz), order='F')
        #grid_density = raw_density.reshape((nx, ny, nz), order='C')

    else:
        print(f"Mismatch: Points {raw_density.size} vs Grid {expected_size}. Resampling to structured grid...")
        # Create the full target grid
        grid = pv.ImageData(dimensions=(nx, ny, nz), 
                            spacing=(dx, dy, dz),
                            origin=(x_coords[0], y_coords[0], z_coords[0]))
        
        # Sample the VTU onto the full structured grid
        # This maps values from the mesh to the grid points
        grid = grid.sample(vtu)
        
        grid_density = grid.point_data["density"]
        # Fill NaNs (points outside the mesh) with 0.0 (Void) (Iwant to optimize this step!!)
        grid_density = np.nan_to_num(grid_density, nan=0.0)
        
        # Reshape
        grid_density = grid_density.reshape((nx, ny, nz), order='F')

    # --- ADD PADDING --- due to marching cubes issues at boundaries and previous code failed for torque plate 
    # We pad with 1.0 (Solid) so the Void (0.0) is capped inside.
    # np.pad adds layers around the 3D array.
    # ((1,1), (1,1), (1,1)) means add 1 layer before and 1 layer after for each axis.
    padded_density = np.pad(grid_density, ((1,1), (1,1), (1,1)), mode='constant', constant_values=1.0)
    
    # Update dimensions for the new padded grid
    pnx, pny, pnz = padded_density.shape
    print(f"Padded Grid: {pnx} x {pny} x {pnz}")

    # Normalize and Invert (unnecessary if input is already 0-1)
    # (If not, normalize first)
    padded_density = (padded_density - padded_density.min()) / (padded_density.max() - padded_density.min() + 1e-12)
    padded_density = 1.0 - padded_density # Invert: 1=Void, 0=Material (Padding becomes 0)
    
    input_tensor = torch.tensor(padded_density).float().unsqueeze(0).unsqueeze(0)

    # --- STEP 3: TRAIN CNN ---
    model = CNN3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print("Training CNN...")
    model.train()
    for epoch in range(400): # Fast training
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, input_tensor.to(model.device))
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0002:
            print(f"Converged at epoch {epoch}, Loss: {loss.item():.6f}")
            break
        if epoch % 20 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # --- STEP 4: INFERENCE ---
    model.eval()
    with torch.no_grad():
        smoothed = model(input_tensor).cpu().numpy().flatten(order='F')

    # Create Padded PyVista Grid
    # Origin must be shifted back by one spacing unit
    origin = (x_coords[0] - dx, y_coords[0] - dy, z_coords[0] - dz)
    
    grid = pv.ImageData(dimensions=(pnx, pny, pnz), 
                        spacing=(dx, dy, dz),
                        origin=origin)
    
    grid.point_data["cnn_density"] = smoothed
    
    return grid.contour([isovalue], scalars="cnn_density", method='marching_cubes')

def visualize(original_stl, other_surface,other_surface_label="Void Isosurface"):
    p = pv.Plotter()
    p.add_mesh(original_stl, color='lightblue', opacity=0.3, label="Original STL")
    p.add_mesh(other_surface, color='red', opacity=0.9, label=other_surface_label)
    p.add_legend()
    p.camera_position = 'iso'
    p.show()

def to_trimesh(mesh):
    """Convert a PyVista mesh to Trimesh."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=mesh.points, faces=faces)

def subtract_voids_from_stl(stl, void_surface):
    # Load STL as PyVista, convert both to Trimesh

    tm_base = to_trimesh(stl)
    tm_void = to_trimesh(void_surface)
    
    # Fill holes and fix normals for robust boolean
    trimesh.repair.fill_holes(tm_base)
    trimesh.repair.fix_normals(tm_base)
    trimesh.repair.fill_holes(tm_void)
    trimesh.repair.fix_normals(tm_void)
    
    # Do boolean subtraction (STL - voids)
    result_tm = trimesh.boolean.difference([tm_base, tm_void])
    if result_tm is None or result_tm.faces.shape[0] == 0:
         print("Boolean subtraction failed, returning original STL.")
         result_pv = stl
    else:
         result_pv = pv.PolyData(result_tm.vertices, faces=np.hstack(
             [np.full((len(result_tm.faces), 1), 3), result_tm.faces]
         ))

    return result_pv

if __name__ == "__main__":


    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    design_domain_stl_file = os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL.STL')
    print(f"Design Domain STL file: {design_domain_stl_file}")
    vtu_file= os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL.vtu')
    output_file = os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL_recovered.STL')
    
    design_domain = pv.read(design_domain_stl_file).triangulate().compute_normals()  # Read and triangulate the STL file
    vtu = pv.read(vtu_file)

    void_region_stl = extract_isosurface_cnn(vtu)
    visualize(design_domain, void_region_stl,"Void Isosurface")

    optimized_topology_stl = subtract_voids_from_stl(design_domain, void_region_stl)
    visualize(design_domain, optimized_topology_stl,"Optimized STL")
    optimized_topology_stl.save(output_file)