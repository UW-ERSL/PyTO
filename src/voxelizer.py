# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 06:51:26 2025

@author: ksure
"""
import numpy as np # pip install numpy
import pyvista as pv # pip install pyvista
# stl-to-voxel  requires C++ compiler. For Windows, Visual Studio 2022 Community works
from PIL import Image
import os
from tkinter import filedialog

class Voxelizer:
    def __init__(self, stl_file_path=None, nVoxels=10**5):
        self.voxels = None
        if stl_file_path:
            self.create_from_stl(stl_file_path, nVoxels)

    def getVoxelDimensionOld(self, nVoxelsDesired=10**5, max_aspect_ratio=1.2):
        # Get bounds and dimensions
        bounds = self.stlMesh.bounds
        print("STL Bounds: ", bounds)
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        volume = self.stlMesh.volume

        # Calculate grid sizes
        h_size = [0, 0, 0]
        eta = [0, 0, 0]
        
        if Lx >= Ly and Lx >= Lz:
            eta[0] = 1.0
            eta[1] = max(Ly/Lx, 1/max_aspect_ratio)
            eta[2] = max(Lz/Lx, 1/max_aspect_ratio)
            
            h_size[0] = (volume/nVoxelsDesired/(eta[0]*eta[1]*eta[2]))**(1/3)
            h_size[1] = h_size[0] * eta[1]
            h_size[2] = h_size[0] * eta[2]
        
        elif Ly >= Lx and Ly >= Lz:
            eta[1] = 1.0
            eta[0] = max(Lx/Ly, 1/max_aspect_ratio)
            eta[2] = max(Lz/Ly, 1/max_aspect_ratio)
            
            h_size[1] = (volume/nVoxelsDesired/(eta[0]*eta[1]*eta[2]))**(1/3)
            h_size[0] = h_size[1] * eta[0]
            h_size[2] = h_size[1] * eta[2]
        
        else:
            eta[2] = 1.0
            eta[0] = max(Lx/Lz, 1/max_aspect_ratio)
            eta[1] = max(Ly/Lz, 1/max_aspect_ratio)
            
            h_size[2] = (volume/nVoxelsDesired/(eta[0]*eta[1]*eta[2]))**(1/3)
            h_size[0] = h_size[2] * eta[0]
            h_size[1] = h_size[2] * eta[1]

        vox_nels = [0, 0, 0]
        vox_nels[0] = max(round(Lx/h_size[0]), 2)
        vox_nels[1] = max(round(Ly/h_size[1]), 2)
        vox_nels[2] = max(round(Lz/h_size[2]), 2)
        print(vox_nels)
        print( h_size)
        h_size[0] = Lx/vox_nels[0]
        h_size[1] = Ly/vox_nels[1]
        h_size[2] = Lz/vox_nels[2]
        print( h_size)

        return h_size[0], h_size[1], h_size[2]
    
    def getVoxelDimension(self, nVoxelsDesired=10**5, max_aspect_ratio=1.2):
        # Get bounds and dimensions
        bounds = self.stlMesh.bounds
        print("STL Bounds: ", bounds)
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        volume = self.stlMesh.volume
        alpha = (nVoxelsDesired/(Lx*Ly*Lz))**(1/3)
        vox_nels = [0, 0, 0]
        vox_nels[0] = max(round(alpha*Lx), 2)
        vox_nels[1] = max(round(alpha*Ly), 2)
        vox_nels[2] = max(round(alpha*Lz), 2)
        print(vox_nels)
        h_size = [0, 0, 0]
        h_size[0] = Lx/vox_nels[0]
        h_size[1] = Ly/vox_nels[1]
        h_size[2] = Lz/vox_nels[2]
        print( h_size)

        return h_size[0], h_size[1], h_size[2]
    
    def create_from_stl(self, stl_file, nVoxelsDesired=10**5):

        self.stlMesh = pv.read(stl_file)
        # Get the volume of the mesh
        volume = self.stlMesh.volume
        voxelDimensions = self.getVoxelDimension(nVoxelsDesired)
      
        # Voxels near the boundary are being removed. So scale the mesh slightly
        scale = 1.001
        # Scale the mesh by 10% about its center
        center = np.array(self.stlMesh.center)
        self.stlMesh.points = (self.stlMesh.points - center) * scale + center
        self.voxels = pv.voxelize(self.stlMesh, density=voxelDimensions, check_surface=False)

        # Scale back to original size
        center = np.array(self.stlMesh.center)
        self.stlMesh.points = (self.stlMesh.points - center) / scale + center

        #extract the data
        self.nVoxels = self.voxels.n_cells
        self.nPoints = self.voxels.n_points
        
        self.dx = voxelDimensions[0] 
        self.dy = voxelDimensions[1]
        self.dz = voxelDimensions[2]
        # Calculate the size of the voxel      
     
        self.origin = self.voxels.bounds[0], self.voxels.bounds[2], self.voxels.bounds[4]
        self.totalVoxelVolume = self.dx*self.dy*self.dz*self.nVoxels
        self.percentVolErr = (volume - self.totalVoxelVolume)/volume*100

        print("***********Voxel grid created***********")
        print("stl volume: ", volume)
        print("vox volume: ", self.totalVoxelVolume )
        print(f"#Voxels: {self.nVoxels}")
        print(f"#Points: {self.nPoints}")
        print(f"VolErr: {self.percentVolErr:.2f}%")
        print(f"(dx,dy,dz): {voxelDimensions[0]:.2e}, {voxelDimensions[1]:.2e}, {voxelDimensions[2]:.2e}")

       
    def getVoxelPoint(self, i):
        if self.voxels is None:
            raise ValueError("No voxel grid data available")    
        return self.voxels.points[i]
    
    def getVoxelElement(self, i):
        if self.voxels is None:
            raise ValueError("No voxel grid data available")
        return self.voxels.get_cell(i).point_ids

    def plot(self):
        if self.voxels is None:
            raise ValueError("No voxel grid data available")
        p = pv.Plotter()
        # Plot inside elements only
        p.add_mesh(self.stlMesh, color= 'grey',opacity=0.4, show_edges=True)
        p.add_mesh(self.voxels, color= 'green', opacity=0.9, show_edges=True)
        p.show()

    def findVoxelsNearTriangle(self, triangle_id, distance):
        """
        Find all voxel points within a specified distance from a triangle in the STL mesh.
            
        Args:
            triangle_id: Index of the triangle in the STL mesh
            distance: Maximum distance to search for voxel points
                
        Returns:
            numpy array of point indices that are within the specified distance
        """
        if self.voxels is None or self.stlMesh is None:
            raise ValueError("No voxel grid or STL mesh data available")
            
        # Get the triangle from the mesh
        triangle = self.stlMesh.extract_cells(triangle_id)
            
        # Create selection based on distance
        # Create a surface from the triangle and compute distances
        surface = triangle.extract_surface()
        
        # Compute distances and threshold
        distances = self.voxels.compute_implicit_distance(surface)
        surface_with_thickness = self.voxels.threshold([-distance, distance], scalars=distances)
        
        selection = self.voxels.select_enclosed_points(surface_with_thickness, check_surface=False)
            
        # Get indices of points within distance
        return np.where(selection)[0]
    
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
    vox.create_from_stl(stl_file, nVoxelsDesired=1000)
    # Visualize the voxel grid
    vox.plot()
   
    
    