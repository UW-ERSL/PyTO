"""Plot functions for visualization of mesh and results."""
import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pyvista as pv
import vtk
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
  low_density_surface.plot(show_edges=True, color='red')
  #low_density_surface = low_density_surface.smooth(n_iter=50)
  #low_density_surface.plot(show_edges=True, color='red')

  # Save the resulting surface as an STL file (STL requires PolyData)
  fp_low_density_surface = 'low_density_elements_isosurface.stl'
  low_density_surface.save(fp_low_density_surface, binary=True)

  original_stl = pv.read(fp_original_stl).clean().extract_surface().triangulate()
  #original_stl.plot(show_edges=True, color='lightblue')
  original_stl.plot(color='lightblue')
  # Compute normals if not available
  if 'Normals' not in original_stl.point_data:
      original_stl.compute_normals(inplace=True)

  # Define a small dilation factor (adjust as needed) #Required as small bits remain at the outer surface
  bounds = low_density_surface.bounds
  diag_length = np.sqrt((bounds[1]-bounds[0])**2 +
                        (bounds[3]-bounds[2])**2 +
                        (bounds[5]-bounds[4])**2)
  dilation_factor = 0.0015 * diag_length
  inflated_points = low_density_surface.points + dilation_factor * low_density_surface.point_data['Normals']
  inflated_low_density_surface = pv.PolyData(inflated_points, faces=low_density_surface.faces)

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
  #low_density_elements.plot(show_edges=True, color='red')
  # Split into connected bodies (patches)
  patches = low_density_elements.split_bodies()
  min_cells = 100 #ToDo: get top 90percent of the patches wrt to the largest patch
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
          fp_original_stl = fp_outputstlpath

# 2. Clean + triangulate + normals on *both* meshes
def preprocess(pd: pv.PolyData) -> pv.PolyData:
    # a) Clean coincident points, degenerate cells, etc.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.Update()
    pd_clean = pv.wrap(cleaner.GetOutput())                              # :contentReference[oaicite:0]{index=0}

    # b) Ensure only triangles
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd_clean)
    tri.Update()
    pd_tri = pv.wrap(tri.GetOutput())                                    # :contentReference[oaicite:1]{index=1}

    # c) Recompute normals (consistent, no splits)
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(pd_tri)
    norms.ConsistencyOn()
    norms.SplittingOff()
    norms.Update()
    pd_norm = pv.wrap(norms.GetOutput())                                 # :contentReference[oaicite:2]{index=2}

    return pd_norm


def removePatchFromSTL(fp_original_stl: str, patch: pv.PolyData, fp_outputstlpath: str):
  low_density_surface = patch.triangulate().extract_surface().clean()
  # Volume preserving smoothing
  #pass_band=0.05 # varies betn 0 to 2. if 2, more of the original shape is retained
  low_density_surface = low_density_surface.smooth_taubin(n_iter=100, pass_band=0.05, non_manifold_smoothing=True) 

  # Save the resulting surface as an STL file (STL requires PolyData)
  fp_low_density_surface = 'low_density_elements_isosurface.stl'
  low_density_surface.save(fp_low_density_surface, binary=True)
  original_stl = pv.read(fp_original_stl).clean().extract_surface().triangulate()
  #original_stl = original_stl.subdivide_adaptive()
  original_stl.plot(show_edges=True, color='lightblue')

  #Compute normals if not available
  if 'Normals' not in original_stl.point_data:
      original_stl.compute_normals(inplace=True)

  # Define a small dilation factor (adjust as needed) #Required as small bits remain at the outer surface
  bounds = low_density_surface.bounds
  diag_length = np.sqrt((bounds[1]-bounds[0])**2 +
                        (bounds[3]-bounds[2])**2 +
                        (bounds[5]-bounds[4])**2)
  dilation_factor = 0.005 * diag_length
  inflated_points = low_density_surface.points + dilation_factor * low_density_surface.point_data['Normals']
  inflated_low_density_surface = pv.PolyData(inflated_points, faces=low_density_surface.faces)
  
  plotter = pv.Plotter()
  _ = plotter.add_mesh(inflated_low_density_surface, color='red', show_edges=True, style='wireframe')
  _ = plotter.add_mesh(original_stl, color='green', show_edges=True, style='wireframe')
  plotter.show_axes()
  plotter.show()
  #inflated_low_density_surface.plot(show_edges=True, color='lightblue')

  mesh_diff = original_stl.boolean_difference(inflated_low_density_surface).clean()
  cleaned_mesh = mesh_diff.connectivity('largest')
  #mesh_diff.triangulate(inplace=True).subdivide(2)
  cleaned_mesh = preprocess(cleaned_mesh)
  cleaned_mesh.plot(show_edges=True, color='lightblue')
  cleaned_mesh.save(fp_outputstlpath, binary=True)

def plot_stl(stl_file: str, color = 'lightskyblue', show_edges = False):
  plotter = pv.Plotter()
  component = pv.read(stl_file)
  
  plotter.add_mesh(component, color=color, show_edges=show_edges)

  plotter.add_axes()
  plotter.show_grid()
  plotter.show()


# Example usage
if __name__ == "__main__":
    
  fp_org_stl = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/BliskSectionWithBlade2test.STL"
  fp_vtu_mesh = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/test1.vtu"
  outputstlpath = "./BliskSectionWithBlade2Recovered.stl"

  vtu_mesh = pv.read(fp_vtu_mesh)
  

  getRetainedOuterGeomSTL(fp_org_stl,
                  vtu_mesh,
                  outputstlpath, getOnlyLargestPatchDiff = False)
