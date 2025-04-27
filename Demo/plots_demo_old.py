"""Plot functions for visualization of mesh and results."""
import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import mesher


def retainOuterGeom(mesh: mesher.Mesher,
                   fp_original_stl: str,
                   u=None,
                   Binarization = False,
                  ):
  """retain Outer geometry.

  Args:
  """
  # Create vertices array
  vertices = mesh.node_xyz
  
  # Handle deformation if provided
  if (u is not None) and (np.max(np.abs(u)) > 0):
    delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    deltaMax = np.max(delta)
    scale = 0.1*mesh.bbox.diag_length/deltaMax
    uVertex = u.reshape(vertices.shape)
    vertices = vertices + scale*uVertex
    values = delta
  else:
    values = None

  # Create cells array for PyVista
  cells = np.hstack((
                      np.full((mesh.num_elems, 1), 8),  # 8 vertices per hexahedron
                      mesh.elemArray
                    ))

  # Create PyVista mesh
  pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

  elemPseudoDensity = mesh.elemPseudoDensity
  if (Binarization):
    elemPseudoDensity = np.where(elemPseudoDensity > 0.5, 1, 0)
  # Add element densities to cell data
  pv_mesh.cell_data['density'] = elemPseudoDensity

  # Extract cells with density less than 0.5
  low_density_elements = pv_mesh.threshold([-np.inf, 0.5], scalars='density')

  # Optionally, visualize these low-density elements:
  #low_density_elements.plot(show_edges=True, color='red')
  
  # Split into connected bodies (patches)
  patches = low_density_elements.split_bodies()

  # Check if the result is a MultiBlock
  if isinstance(patches, pv.MultiBlock):
      largest_patch = None
      max_cells = -1
      # Iterate over each patch in the MultiBlock
      for patch in patches:
          if patch is not None and patch.n_cells > max_cells:
              max_cells = patch.n_cells
              largest_patch = patch
  else:
      # If split_bodies() returns a single mesh, then that's our patch
      largest_patch = patches

  # Extract the outer surface of these low-density elements
  low_density_surface = largest_patch.triangulate().extract_surface().clean()
  #low_density_surface.plot(show_edges=True, color='red')
  #low_density_surface = low_density_surface.smooth(n_iter=50)
  #low_density_surface.plot(show_edges=True, color='red')

  # Save the resulting surface as an STL file (STL requires PolyData)
  fp_low_density_surface = 'low_density_elements_isosurface.stl'
  low_density_surface.save(fp_low_density_surface, binary=True)

  original_stl = pv.read(fp_original_stl).clean().extract_surface().triangulate()
  #original_stl.plot(show_edges=True, color='lightblue')
  #original_stl.plot(color='lightblue')
  # Compute normals if not available
  if 'Normals' not in original_stl.point_data:
      original_stl.compute_normals(inplace=True)

  # Define a small dilation factor (adjust as needed) #Required as small bits remain at the outer surface
  bounds = low_density_surface.bounds
  diag_length = np.sqrt((bounds[1]-bounds[0])**2 +
                        (bounds[3]-bounds[2])**2 +
                        (bounds[5]-bounds[4])**2)
  dilation_factor = 5.0 * diag_length
  inflated_points = low_density_surface.points + dilation_factor * low_density_surface.point_data['Normals']
  inflated_low_density_surface = pv.PolyData(inflated_points, faces=low_density_surface.faces)
 # Translate the mesh along the z-axis
  z_min, z_max = inflated_low_density_surface.bounds[4], inflated_low_density_surface.bounds[5]  # Get the bounds of the mesh
  z_length = z_max - z_min # Calculate the length in the z-direction
  # Define the translation distance as a percentage of the z_length
  percentage = 0.01  # For example, 10 for 10%
  dz = (percentage / 100.0) * z_length
  inflated_low_density_surface = inflated_low_density_surface.translate((0, 0, -dz), inplace=False)

  plotter = pv.Plotter()
  _ = plotter.add_mesh(inflated_low_density_surface, color='red', show_edges=True)
  _ = plotter.add_mesh(original_stl, color='green', show_edges=True, style='wireframe')
  plotter.show_axes()
  plotter.show()

  
  mesh_diff = original_stl.boolean_difference(inflated_low_density_surface).clean()
  #mesh_diff.plot(show_edges=True, color='lightblue')
  mesh_diff.plot(color='lightskyblue')
  
  FinalRcoveredNoseCone = 'FinalRcoveredNoseCone.stl'
  mesh_diff.save(FinalRcoveredNoseCone, binary=True)
  

def retainOuterGeomUsingIsoSurf(mesh: mesher.Mesher,
                   fp_original_stl: str,
                   u=None,
                   Binarization = False,
                   isovalue: float = 0.5,
                  ):
  """retain Outer geometry.

  Args:
  """
  # Create vertices array
  vertices = mesh.node_xyz
  
  # Handle deformation if provided
  if (u is not None) and (np.max(np.abs(u)) > 0):
    delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    deltaMax = np.max(delta)
    scale = 0.1*mesh.bbox.diag_length/deltaMax
    uVertex = u.reshape(vertices.shape)
    vertices = vertices + scale*uVertex
    values = delta
  else:
    values = None

  # Create cells array for PyVista
  cells = np.hstack((
                      np.full((mesh.num_elems, 1), 8),  # 8 vertices per hexahedron
                      mesh.elemArray
                    ))

  # Create PyVista mesh
  pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

  elemPseudoDensity = mesh.elemPseudoDensity
  if (Binarization):
    elemPseudoDensity = np.where(elemPseudoDensity > 0.5, 1, 0)
  # Add element densities to cell data
  pv_mesh.cell_data['density'] = elemPseudoDensity

  # Extract cells with density less than 0.5
  low_density_elements = pv_mesh.threshold([-np.inf, 0.5], scalars='density')

  # Optionally, visualize these low-density elements:
  #low_density_elements.plot(show_edges=True, color='red')
  fp_outputstlpath = 'final_retained_geom.stl'
  getRetainedOuterGeomSTL(fp_original_stl,
                   low_density_elements,
                   fp_outputstlpath, isovalue = 0.5)
  

def getRetainedOuterGeomSTL(fp_original_stl: str,
                   low_density_elements: pv.PolyData,
                   fp_outputstlpath: str, getOnlyLargestPatchDiff = False,
):               
  
  low_density_elements.plot(show_edges=True, color='red')
  # Extract cells with density less than 0.5
  low_density_elements = low_density_elements.threshold([-np.inf, 0.5], scalars='density')
  low_density_elements.plot(show_edges=True, color='red')
  # Split into connected bodies (patches)
  patches = low_density_elements.split_bodies()
  min_cells = 100
  filtered_patches = [patch for patch in patches if patch.n_cells > min_cells]
  patches_list = list(filtered_patches)
  
  # Check if the result is a MultiBlock
  if isinstance(patches, pv.MultiBlock):
    for patch in patches_list:
          print(f"Patch: {patch.n_cells}")
    if getOnlyLargestPatchDiff:
      # Get the largest patch
      largest_patch = max(patches_list, key=lambda p: p.n_cells)
      print(f"Largest Patch: {largest_patch.n_cells}")
      removePatchFromSTL(fp_original_stl, largest_patch, fp_outputstlpath)
    else:
        # Iterate over each patch in the MultiBlock
        for patch in patches_list:
          print(f"Patch: {patch.n_cells}")
          removePatchFromSTL(fp_original_stl, patch, fp_outputstlpath)
        
        
 
def removePatchFromSTL(fp_original_stl: str, patch: pv.PolyData, fp_outputstlpath: str):
  low_density_surface = patch.triangulate().extract_surface().clean()
  low_density_surface = low_density_surface.smooth(n_iter=100)

  # Save the resulting surface as an STL file (STL requires PolyData)
  fp_low_density_surface = 'low_density_elements_isosurface.stl'
  low_density_surface.save(fp_low_density_surface, binary=True)

  original_stl = pv.read(fp_original_stl).clean().extract_surface().triangulate()
  #original_stl.plot(show_edges=True, color='lightblue')
  #original_stl.plot(color='lightblue')
  # Compute normals if not available
  if 'Normals' not in original_stl.point_data:
      original_stl.compute_normals(inplace=True)

  # Define a small dilation factor (adjust as needed) #Required as small bits remain at the outer surface
  bounds = low_density_surface.bounds
  diag_length = np.sqrt((bounds[1]-bounds[0])**2 +
                        (bounds[3]-bounds[2])**2 +
                        (bounds[5]-bounds[4])**2)
  dilation_factor = 0.001 * diag_length
  inflated_points = low_density_surface.points + dilation_factor * low_density_surface.point_data['Normals']
  inflated_low_density_surface = pv.PolyData(inflated_points, faces=low_density_surface.faces)
  #inflated_low_density_surface.plot(show_edges=True, color='lightblue')

  plotter = pv.Plotter()
  _ = plotter.add_mesh(inflated_low_density_surface, color='red', show_edges=True)
  _ = plotter.add_mesh(original_stl, color='green', show_edges=True, style='wireframe')
  plotter.show_axes()
  plotter.show()

  
  mesh_diff = original_stl.boolean_difference(inflated_low_density_surface).clean()
  mesh_diff.plot(show_edges=True, color='lightblue')
  #mesh_diff.plot(color='lightblue')
  
  #mesh_diff.save(fp_outputstlpath, binary=True)

def plot_stl(stl_file: str, color = 'lightskyblue', show_edges = False):
  plotter = pv.Plotter()
  component = pv.read(stl_file)
  
  plotter.add_mesh(component, color=color, show_edges=show_edges)

  plotter.add_axes()
  plotter.show_grid()
  plotter.show()


# Example usage
if __name__ == "__main__":
    
  fp_org_stl = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/BliskSectionWithBlade2.STL"
  fp_vtu_mesh = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/test1.vtu"
  outputstlpath = "./BliskSectionWithBlade2Recovered.stl"

  vtu_mesh = pv.read(fp_vtu_mesh)
  

  getRetainedOuterGeomSTL(fp_org_stl,
                  vtu_mesh,
                  outputstlpath, getOnlyLargestPatchDiff = True)