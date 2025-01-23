# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 06:51:26 2025

@author: ksure
"""
import numpy as np # pip install numpy
import pyvista as pv # pip install pyvista
# stl-to-voxel  requires C++ compiler. For Windows, Visual Studio 2022 Community works
import stltovoxel # pip install stl-to-voxel (see above)
from PIL import Image
import os
from tkinter import filedialog

class Voxelizer:
    def __init__(self, stl_file_path=None, resolution=100):
        self.voxel_grid = None
        self.grid_coords = None
        if stl_file_path:
            self.create_from_stl(stl_file_path, resolution)

    def create_from_stl(self, stl_file_path, resolution=100):
        png_dir = "."
        stltovoxel.convert_file(stl_file_path, 'output.png', resolution=resolution)
        
        png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])
        slices = []
        for file in png_files:
            img = Image.open(os.path.join(png_dir, file)).convert("L")
            slices.append(np.array(img))
        
        self.voxel_grid = np.stack(slices, axis=0)
        
        for file in png_files:
            os.remove(os.path.join(png_dir, file))
            
        x = np.arange(self.voxel_grid.shape[2])
        y = np.arange(self.voxel_grid.shape[1])
        z = np.arange(self.voxel_grid.shape[0])
        self.grid_coords = (x, y, z)

    def dump(self, filename="voxel_grid.bin"):
        if self.voxel_grid is None:
            raise ValueError("No voxel grid data available")
        
        def _create_node_coords(self, n_nodes, shape):
            i_vals = np.arange(shape[2] + 1)
            j_vals = np.arange(shape[1] + 1)
            k_vals = np.arange(shape[0] + 1)
            I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
            node_coords = np.zeros((n_nodes, 4), dtype=np.int32)
            node_coords[:, 0] = I.flatten()
            node_coords[:, 1] = J.flatten()
            node_coords[:, 2] = K.flatten()
            return node_coords

        def _calculate_node_indices(self, i, j, k, nelx, nely):
            n1 = i + j*(nelx+1) + k*(nelx+1)*(nely+1)
            n2 = (i+1) + j*(nelx+1) + k*(nelx+1)*(nely+1)
            n3 = (i+1) + (j+1)*(nelx+1) + k*(nelx+1)*(nely+1)
            n4 = i + (j+1)*(nelx+1) + k*(nelx+1)*(nely+1)
            n5 = n1 + (nelx+1)*(nely+1)
            n6 = n2 + (nelx+1)*(nely+1)
            n7 = n3 + (nelx+1)*(nely+1)
            n8 = n4 + (nelx+1)*(nely+1)
            return np.column_stack([n1,n2,n3,n4,n5,n6,n7,n8]).astype(np.int32)

        
        # Create binary output file
        with open(filename, 'wb') as f:
            # 1. Write dimensions
            np.array([self.voxel_grid.shape[2], self.voxel_grid.shape[1], self.voxel_grid.shape[0]], dtype=np.int32).tofile(f)
            
                # Create coordinate grids for visualization
            x = np.arange(self.voxel_grid.shape[2])
            y = np.arange(self.voxel_grid.shape[1])
            z = np.arange(self.voxel_grid.shape[0])
            grid_coords = (x, y, z)
            # 2. Write origin coordinates
            np.array([x[0], y[0], z[0]], dtype=np.float32).tofile(f)
            
            # 3. Write spacing
            dx = x[1] - x[0] if len(x) > 1 else 1
            dy = y[1] - y[0] if len(y) > 1 else 1
            dz = z[1] - z[0] if len(z) > 1 else 1
            np.array([dx, dy, dz], dtype=np.float32).tofile(f)
            
            # 4-5. Write nodes (0 facegroup for all nodes)
            n_nodes = (self.voxel_grid.shape[2]+1) * (self.voxel_grid.shape[1]+1) * (self.voxel_grid.shape[0]+1)
            np.array([n_nodes], dtype=np.int32).tofile(f)
            # Convert the nested loops to a parallel operation using numpy
            node_coords = np.zeros((n_nodes, 4), dtype=np.int32)
            i_vals = np.arange(self.voxel_grid.shape[2] + 1)
            j_vals = np.arange(self.voxel_grid.shape[1] + 1)
            k_vals = np.arange(self.voxel_grid.shape[0] + 1)
            I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
            node_coords[:, 0] = I.flatten()
            node_coords[:, 1] = J.flatten()
            node_coords[:, 2] = K.flatten()
            node_coords.tofile(f)

            
            # 6-7. Write elements
            n_elems = np.sum(self.voxel_grid > 0)
            np.array([n_elems], dtype=np.int32).tofile(f)
            nelx, nely, nelz = self.voxel_grid.shape[2], self.voxel_grid.shape[1], self.voxel_grid.shape[0]
            # Create indices for all possible elements
            i, j, k = np.where(self.voxel_grid > 0)
            # Calculate node indices in parallel
            n1 = i + j*(nelx+1) + k*(nelx+1)*(nely+1)
            n2 = (i+1) + j*(nelx+1) + k*(nelx+1)*(nely+1)
            n3 = (i+1) + (j+1)*(nelx+1) + k*(nelx+1)*(nely+1)
            n4 = i + (j+1)*(nelx+1) + k*(nelx+1)*(nely+1)
            n5 = n1 + (nelx+1)*(nely+1)
            n6 = n2 + (nelx+1)*(nely+1)
            n7 = n3 + (nelx+1)*(nely+1)
            n8 = n4 + (nelx+1)*(nely+1)
            # Stack all nodes and write at once
            nodes = np.column_stack([n1,n2,n3,n4,n5,n6,n7,n8]).astype(np.int32)
            nodes.tofile(f)
    
            # 8. Write material IDs (all 0)
            np.zeros(n_elems, dtype=np.int32).tofile(f)
            
            # 9. Write densities (all 1.0)
            np.ones(n_elems, dtype=np.float32).tofile(f)

    def plot(self):
        if self.voxel_grid is None:
            raise ValueError("No voxel grid data available")
        x, y, z = self.grid_coords
    
        # Create a PyVista grid for visualization
        grid = pv.ImageData()
        
        # Set grid dimensions
        grid.dimensions = np.array(self.voxel_grid.shape) + 1
        
        # Set grid spacing (assumes uniform spacing)
        spacing = (
            (x[-1] - x[0]) / (len(x) - 1),
            (y[-1] - y[0]) / (len(y) - 1),
            (z[-1] - z[0]) / (len(z) - 1),
        )
        grid.spacing = spacing
        
        # Set grid origin
        grid.origin = (x[0], y[0], z[0])
        # Convert boolean array to float for better visualization
        voxel_data = self.voxel_grid.astype(float)

        # Add the voxel data to the grid
        grid.cell_data["voxel"] = voxel_data.ravel(order="F")
        
        # Threshold the grid to only show filled voxels
        threshed = grid.threshold(value=0.5)  # Only show voxels that are "inside" (value = 1)
        
        # Visualize using PyVista
        plotter = pv.Plotter()
        plotter.add_mesh(threshed, opacity=0.3, color='blue')  # Changed to add_mesh for thresholded data
        plotter.add_axes()  # Add axes for better orientation
        plotter.show()

if __name__ == "__main__":
    # Open file dialog to select STL file
    stl_file = filedialog.askopenfilename(
        title="Select STL file",
        filetypes=[("STL files", "*.stl *.STL")]
    )
    if not stl_file:
        print("No file selected")
        exit()

    vox = Voxelizer()
    vox.create_from_stl(stl_file, resolution=50)
    # Visualize the voxel grid
    vox.plot()
    # Dump the voxel grid to a binary file
    #vox.dump("voxel_grid.msh")
    
    