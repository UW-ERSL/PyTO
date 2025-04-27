"""Plot functions for visualization of mesh and results."""
import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import numpy as np
import pyvista as pv
import vtk
import vedo

import matplotlib.pyplot as plt

import mesher
import pymeshfix
import enum


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
  
  #low_density_elements.plot(show_edges=True, color='red')
  # Extract cells with density less than 0.5
  low_density_elements = low_density_elements.threshold([-np.inf, 0.5], scalars='density')
  low_density_elements.plot(show_edges=True, color='red')
  # Split into connected bodies (patches)
  patches = low_density_elements.split_bodies()
  min_cells = 1000 #ToDo: get top 90percent of the patches wrt to the largest patch
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
        for i, patch in enumerate(patches_list):
          print(f"Patch: {patch.n_cells}")
          dxfactor, dyfactor, dzfactor = 0.0, 0.0, 1.0
          smooth_iter = 100
          dilation_factor_scale = 0.005
          b_extra_preprocess = False
          if i==1:
             dxfactor, dyfactor, dzfactor = 0.0, 0.0, 0.0
             smooth_iter = 200
             dilation_factor_scale = 0.02
             b_extra_preprocess = True
          removePatchFromSTL(fp_original_stl, patch, fp_outputstlpath, dxfactor, dyfactor, dzfactor, smooth_iter, dilation_factor_scale, b_extra_preprocess)
          fp_original_stl = fp_outputstlpath

# 2. Clean + triangulate + normals on *both* meshes
def preprocess(pd: pv.PolyData) -> pv.PolyData:
    # a) Clean coincident points, degenerate cells, etc.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.Update()
    pd_clean = pv.wrap(cleaner.GetOutput())                             

    # b) Ensure only triangles
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd_clean)
    tri.Update()
    pd_tri = pv.wrap(tri.GetOutput())                                    

    # c) Recompute normals (consistent, no splits)
    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(pd_tri)
    norms.ConsistencyOn()
    norms.SplittingOff()
    norms.Update()
    pd_norm = pv.wrap(norms.GetOutput())                                
    
    # CHeck if the mesh is manifold
    # vmesh = vedo.Mesh(pd_norm)  
    # vmesh.non_manifold_faces(remove=True, tol="auto")
    # mesh = pv.wrap(vmesh)  # back to PyVista  

    return pd_norm


def removePatchFromSTL(fp_original_stl: str, patch: pv.PolyData, fp_outputstlpath: str, 
                       dxfactor: float = 0.0, dyfactor: float = 0.0, dzfactor: float = 0.0, 
                       smooth_iter: int = 100,
                       dilation_factor_scale: float = 0.005,
                       b_extra_preprocess: bool = False,):
  check_and_verify_mesh(patch)
  
  low_density_surface = patch.triangulate().extract_surface().clean()
  low_density_surface.plot(show_edges=True, color='red')
  low_density_surface = low_density_surface.smooth(n_iter=4)
  check_and_verify_mesh(low_density_surface)

  # Volume preserving smoothing
  #pass_band=0.05 # varies betn 0 to 2. if 2, more of the original shape is retained
  # low_density_surface = low_density_surface.smooth_taubin(n_iter=smooth_iter,
  #   pass_band=0.055,
  #   boundary_smoothing=False,
  #   feature_smoothing=False, #preserve holes or features if present
  #   normalize_coordinates=True, #Prevents numerical issues on oddly sized models.
  #   non_manifold_smoothing=True) 

  if b_extra_preprocess:
    low_density_surface = low_density_surface.delaunay_3d()
    low_density_surface = low_density_surface.extract_surface().clean()
    low_density_surface.plot(show_edges=True, color='red')
    #low_density_surface = low_density_surface.smooth(n_iter=100)
  # Save the resulting surface as an STL file (STL requires PolyData)
  fp_low_density_surface = 'low_density_elements_isosurface.stl'
  low_density_surface.save(fp_low_density_surface, binary=True)
  original_stl = pv.read(fp_original_stl).clean().extract_surface().triangulate()
  #original_stl = original_stl.subdivide_adaptive()
  #original_stl.plot(show_edges=True, color='lightblue')

  #Compute normals if not available
  if 'Normals' not in original_stl.point_data:
      original_stl.compute_normals(inplace=True)

  # Define a small dilation factor (adjust as needed) #Required as small bits remain at the outer surface
  bounds = low_density_surface.bounds
  diag_length = np.sqrt((bounds[1]-bounds[0])**2 +
                        (bounds[3]-bounds[2])**2 +
                        (bounds[5]-bounds[4])**2)
  dilation_factor = dilation_factor_scale * diag_length
  inflated_points = low_density_surface.points + dilation_factor * low_density_surface.point_data['Normals']
  inflated_low_density_surface = pv.PolyData(inflated_points, faces=low_density_surface.faces)

 # Translate the mesh along the z-axis
  z_min, z_max = inflated_low_density_surface.bounds[4], inflated_low_density_surface.bounds[5]  # Get the bounds of the mesh
  z_length = z_max - z_min # Calculate the length in the z-direction
  # Define the translation distance as a percentage of the z_length
  percentage = 1.0  # For example, 10 for 10%
  dz = (percentage / 100.0) * z_length * dzfactor
  dy = dz * dyfactor
  dx = dz * dxfactor
  #dx, dy = -dz*1.5, dz = -dz*2

    
  
  plotter = pv.Plotter()
  _ = plotter.add_mesh(inflated_low_density_surface, color='red', show_edges=True)
  _ = plotter.add_mesh(original_stl, color='green', show_edges=True)
  plotter.show_axes()
  plotter.show()
  inflated_low_density_surface = inflated_low_density_surface.translate((dx, dy, dz))
  inflated_low_density_surface.plot(show_edges=True, color='lightblue')

  mesh_diff = original_stl.boolean_difference(inflated_low_density_surface).clean()
  cleaned_mesh = mesh_diff.connectivity('largest')
  cleaned_mesh = cleaned_mesh.clean(point_merging=True, tolerance=1e-6)  # merges coincident points, drops unused/degenerate faces   

  # 2. Compute diag and choose fraction
  # bounds = cleaned_mesh.bounds
  # diag   = ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5
  # hole_size = 0.05 * diag   # fill holes up to 5% of model size

  # # 3. Fill holes
  # cleaned_mesh = cleaned_mesh.fill_holes(hole_size)
    
  #mf = pymeshfix.PyTMesh()  
  #mf.repair(verbose=True)  
  #fixed = mf.mesh                         
  # Fill holes
  #mf.LoadFile(cleaned_mesh)
  #mf.fill_small_boundaries()
  #print('There are {:d} boundaries'.format(mf.boundaries()))

  # Clean (removes self intersections)
  #mf.clean(max_iters=10, inner_loops=3)                       
 

  #mesh_diff.triangulate(inplace=True).subdivide(2)
  cleaned_mesh = preprocess(cleaned_mesh)
  cleaned_mesh.save(fp_outputstlpath, binary=True)
  #cleaned_mesh = fix_mesh(fp_meshstl=fp_outputstlpath)
  cleaned_mesh.plot(show_edges=True, color='lightblue')
  #cleaned_mesh.save(fp_outputstlpath, binary=True)



def plot_stl(stl_file: str, color = 'lightskyblue', show_edges = False):
  plotter = pv.Plotter()
  component = pv.read(stl_file)
  
  plotter.add_mesh(component, color=color, show_edges=show_edges)

  plotter.add_axes()
  plotter.show_grid()
  plotter.show()


def fix_mesh(fp_meshstl: str)->pv.PolyData:
  mfix = pymeshfix._meshfix.PyTMesh(False)  # False removes extra verbose output
  mfix.load_file(fp_meshstl)

  # Fills all the holes having at at most 'nbe' boundary edges. If
  # 'refine' is true, adds inner vertices to reproduce the sampling
  # density of the surroundings. Returns number of holes patched.  If
  # 'nbe' is 0 (default), all the holes are patched.
  mfix.fill_small_boundaries(nbe=100, refine=True)

  vert, faces = mfix.return_arrays()
  triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
  triangles[:, -3:] = faces
  triangles[:, 0] = 3

  repaired_mesh = pv.PolyData(vert, triangles)
  return repaired_mesh

def check_and_verify_mesh(mesh: pv.PolyData)->pv.PolyData:
  # Check before every boolean operation
  #Check manifoldness. #Non-zero means you’ll likely fail 
  non_manifold = mesh.extract_feature_edges(
    boundary_edges=False,
    non_manifold_edges=True,
    feature_edges=False,
    manifold_edges=False)
  if non_manifold.n_cells:
      print("Non-manifold edges detected:", non_manifold.n_cells)

  #Check watertightness. #Any shall indicate “leaks” in the shell
  boundary_edges = mesh.extract_feature_edges(
    boundary_edges=True,
    non_manifold_edges=False,
    feature_edges=False,
    manifold_edges=False)
  if boundary_edges.n_cells:
      print("Mesh has holes (boundary edges):", boundary_edges.n_cells)

  # mesh = mesh.triangulate()
  # # Ensure triangulation
  # # Verify every face is a triangle;
  # if mesh.is_all_triangles:
  #   print("Mesh is fully triangulated")
  # else:
  #     print("Mesh contains non-triangle faces")

  # #After all cleanup, recompute normals:
  # mesh.compute_normals(consistent=True, splitting=False)

  return mesh


class ExamplesCAD(enum.Enum):
	EdgeCantileverDemo = enum.auto()
	BliskSectionWithBlade = enum.auto()
	KnuckleAssembly = enum.auto()
  
def get_example_cad(example: ExamplesCAD):
  if example == ExamplesCAD.EdgeCantileverDemo:
    fp_original_stl = "../Models/EdgeCantilever/EdgeCantilever.STL"
    fp_vtu_mesh = "./EdgeCantilever.vtu"
    fp_outputstlpath = "./results/EdgeCantileverRecovered.stl"
    fp_outputfixedstlpath = "./results//EdgeCantileverRecoveredFixed.stl"
  elif example == ExamplesCAD.BliskSectionWithBlade:
    fp_original_stl = "../Models/Saketh/BliskSectionWithBlade2test.STL"
    fp_vtu_mesh = "../Models/Saketh/test1.vtu"
    fp_outputstlpath = "../Models/Saketh/BliskSectionWithBlade2Recovered.stl"
    fp_outputfixedstlpath = "../Models/Saketh/BliskSectionWithBlade2RecoveredFixed.stl"
  elif example == ExamplesCAD.KnuckleAssembly:
    fp_original_stl = "../Models/KnuckleAssembly/KnuckleAssembly.STL"
    fp_vtu_mesh = "./KnuckleAssembly (2).vtu"
    fp_outputstlpath = "../Models/KnuckleAssembly/KnuckleAssemblyRecovered.stl"
    fp_outputfixedstlpath = "../Models/KnuckleAssembly/KnuckleAssemblyRecoveredFixed.stl"
  else:
    raise ValueError(f"Unknown example: {example}")
  return fp_original_stl, fp_vtu_mesh, fp_outputstlpath, fp_outputfixedstlpath

# Example usage
if __name__ == "__main__":
    
  # fp_org_stl = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/BliskSectionWithBlade2test.STL"
  # fp_vtu_mesh = "C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Saketh/test1.vtu"
  # outputstlpath = "./BliskSectionWithBlade2Recovered.stl"
  # outputfixedstlpath = "./BliskSectionWithBlade2RecoveredFixed.stl"
  example = ExamplesCAD.KnuckleAssembly
  fp_org_stl, fp_vtu_mesh, outputstlpath, outputfixedstlpath = get_example_cad(example)


  vtu_mesh = pv.read(fp_vtu_mesh)
  getRetainedOuterGeomSTL(fp_org_stl,
                  vtu_mesh,
                  outputstlpath, getOnlyLargestPatchDiff = False)

  mesh = fix_mesh(fp_meshstl=outputstlpath)
  mesh.save(outputfixedstlpath, binary=True)

  pv.read(outputfixedstlpath).plot(show_edges=False, color='lightblue')


