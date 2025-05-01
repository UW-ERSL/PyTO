#!/usr/bin/env python3
"""
Mesh Processing Pipeline
------------------------

This script loads an original STL mesh and a topology-optimized volumetric VTU mesh,
identifies low-density regions (elements with density below a given threshold) in the VTU mesh,
and removes those regions from the original STL geometry. The pipeline uses libraries
(PyVista, pymeshfix, trimesh) to clean and preprocess meshes, perform boolean operations,
and ensure the final output is a clean, watertight STL file that retains the original's critical features.

Modules:
- Loading: Load STL and VTU meshes.
- Cleaning: Remove duplicate/degenerate elements, ensure triangulation, compute normals.
- Feature Extraction: Identify low-density regions via thresholding of volumetric mesh.
- Boolean Operation: Subtract the low-density region from the original mesh.
- Visualization: (Optional) Visualize meshes at key steps (using PyVista or Vedo).
- Saving: Output the final, repaired STL mesh.

Usage:
    python mesh_filter.py --input_stl original.stl --input_vtu topology.vtu \
         --threshold 0.5 --output_stl result.stl
"""

import sys
import numpy as np
import pyvista as pv
import trimesh
import pymeshfix
import vedo
import vtk
import os
import traceback
from plots_examples import *
from plots_demo_2 import *


# Attempt to import visualization and mesh-fixing libraries
try:
    import vedo
    from pymeshfix import MeshFix
    import trimesh
except ImportError as e:
    print(f"Required library missing: {e}", file=sys.stderr)
    sys.exit(1)


def load_mesh(file_path: str) -> pv.PolyData:
    """
    Load a mesh file and return a PyVista mesh.
    Supports STL (surface mesh) and VTU (unstructured volume mesh).
    If a volume mesh is loaded but represents a surface, it is converted to PolyData.
    """
    mesh = None
    try:
        mesh = pv.read(file_path)
    except Exception as e:
        raise RuntimeError(f"Error reading '{file_path}': {e}")

    if mesh is None:
        raise RuntimeError(f"Failed to load mesh from {file_path}")

    return mesh


def clean_mesh(mesh: pv.PolyData, hole_fraction=0.05) -> pv.PolyData:
    """
    Clean the mesh:
    - Merge duplicate points.
    - Remove unused points and degenerate cells.
    - Triangulate faces (ensure all triangles).
    - Compute and orient normals (outward by default).
    """
    mesh = mesh.clean(inplace=False)  # merge duplicates, remove degenerate
    mesh = preprocess(mesh)
    
    #mesh = check_and_repair(mesh)
    mesh = clean_using_trimesh(mesh)
    
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate(inplace=False)

    if mesh.n_faces > 0:
        mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
    else:
        print("Warning: Mesh has no faces to compute normals.")

    
    # Fill holes smaller than some fraction of bounding box diagonal
    bounds = mesh.bounds
    diag = np.linalg.norm([bounds[1] - bounds[0],
                           bounds[3] - bounds[2],
                           bounds[5] - bounds[4]])
    hole_size = diag * hole_fraction
    mesh = mesh.fill_holes(hole_size)

    # Keep only largest connected component
    mesh = mesh.connectivity('largest')
    mesh = mesh.clean()
    return mesh


def clean_using_trimesh(mesh: pv.PolyData) -> pv.PolyData:
    """
    Clean a pyvista.PolyData mesh using trimesh operations:
    - Merge duplicate points.
    - Remove unused points and degenerate cells.
    - Fill small holes.
    - Return a cleaned pyvista.PolyData.
    """
    # Convert pyvista mesh to trimesh
    vertices = np.array(mesh.points)
    faces = np.array(mesh.faces.reshape(-1, 4))[:, 1:]  # PyVista faces are (N_faces, 4), with first number being 3 (for triangle)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Cleaning operations
    tri_mesh.fill_holes()        # fill small holes
    tri_mesh.merge_vertices()    # merge close vertices
    tri_mesh.remove_unreferenced_vertices()
    tri_mesh.update_faces(tri_mesh.nondegenerate_faces())

    # Convert back to pyvista
    cleaned_mesh = pv.PolyData(tri_mesh.vertices, np.hstack(
        [np.full((tri_mesh.faces.shape[0], 1), 3), tri_mesh.faces]).astype(np.int64))

    return cleaned_mesh

def check_and_repair(mesh: pv.PolyData) -> pv.PolyData:
    """
    Check for mesh issues (non-manifold edges, holes) and attempt to repair using pymeshfix.
    If repair fails or mesh is already fine, return the (possibly cleaned) mesh.
    """
    if not mesh.is_manifold:
        # Prepare numpy arrays for MeshFix
        verts = np.array(mesh.points)
        faces = np.array(mesh.faces).reshape(-1, 4)[:, 1:4]
        try:
            meshfix = MeshFix(verts, faces)
            meshfix.repair(verbose=False)
            repaired = pv.PolyData(meshfix.points, meshfix.faces)
            repaired = clean_mesh(repaired)
            if not repaired.is_manifold:
                print("Warning: Mesh is still not manifold after repair.", file=sys.stderr)
            return repaired
        except Exception as e:
            print(f"MeshFix repair failed: {e}", file=sys.stderr)
            return mesh.clean(inplace=False)
    else:
        return mesh

def extract_low_density_patches(vtu: pv.UnstructuredGrid, density_field: str,
                                threshold: float) -> pv.PolyData:
    """
    Threshold the UnstructuredGrid 'vtu' on the given 'density_field' scalar.
    Return the surface mesh (PolyData) of the low-density regions below 'threshold'.
    """
    # Apply a cell threshold: keep cells with density < threshold
    low_density_region = vtu.threshold(value=threshold, scalars=density_field,
                                       invert=True)  # invert=True selects cells below threshold
    
    # Extract the outer surface (polydata) of these regions
    #patches = low_density_region.extract_geometry()
    return low_density_region

def split_and_filter_patches(all_low_density_elems: pv.PolyData, cells_threshold_percentage: float = 10) -> list:
    """
    Split the combined patches PolyData into separate bodies (connected components).
    Then filter out any patch with fewer than min_cells cells.
    Returns a list of PolyData patches that remain.
    """
    patches = all_low_density_elems.split_bodies()  # This returns a MultiBlock
    cell_counts = [patch.n_cells for patch in patches]
    min_cells = cells_threshold_percentage * 0.01 * max(cell_counts)
    filtered_patches = [patch for patch in patches if patch.n_cells > min_cells]
    patches_list = list(filtered_patches)
    return patches_list

def inflate_patch(patch, scale_factor=1.02):
    # Determine the center of the patch
    center = patch.center

    # Translate the patch to the origin
    patch.translate(-np.array(center), inplace=True)

    # Scale the patch
    patch.scale([scale_factor] * 3, inplace=True)

    # Translate the patch back to its original position
    patch.translate(center, inplace=True)

    return patch

def directional_inflate_patch(patch, scale_factor=1.02, direction="z"):
    """
    Scales a patch in a specific direction ("x", "y", or "z").

    Parameters:
        patch (pv.PolyData): The mesh to scale.
        scale_factor (float): Factor by which to scale in the specified direction.
        direction (str): One of "x", "y", or "z".

    Returns:
        pv.PolyData: The scaled patch.
    """
    dir_map = {"x": 0, "y": 1, "z": 2}
    if direction not in dir_map:
        raise ValueError("Direction must be one of 'x', 'y', or 'z'.")

    # Compute center
    center = patch.center

    # Translate to origin
    patch.translate(-np.array(center), inplace=True)

    # Build scaling vector
    scale_vec = [1.0, 1.0, 1.0]
    scale_vec[dir_map[direction]] = scale_factor

    # Scale only in the specified direction
    patch.scale(scale_vec, inplace=True)

    # Translate back
    patch.translate(center, inplace=True)

    return patch


def extract_low_density_region(volume_mesh: pv.DataSet, threshold: float, scalars: str = None) -> pv.PolyData:
    """
    Identify low-density region in the volumetric mesh and extract its boundary surface.
    - threshold: Value below which the material is considered 'low density'.
    - scalars: Name of the scalar array to threshold on (if None, uses active scalars or first available).
    Returns a PyVista PolyData of the surface of the low-density region.
    """
    # Determine which scalar array to use
    if scalars is None:
        if volume_mesh.active_scalars_name is not None:
            scalars = volume_mesh.active_scalars_name
        else:
            arrs = volume_mesh.array_names
            if arrs:
                scalars = arrs[0]
            else:
                raise RuntimeError("No scalar data available in volume mesh for thresholding.")

    # Apply threshold: invert=True to select cells below threshold value
    try:
        low_density = volume_mesh.threshold(value=threshold, scalars=scalars,
                                           invert=True, preference='cell')
    except Exception as e:
        raise RuntimeError(f"Thresholding failed: {e}")

    if low_density.n_cells == 0:
        print("Warning: No cells found under the threshold.", file=sys.stderr)
        return None

    # Extract the outer surface of the low-density region
    try:
        surface = low_density.extract_surface().triangulate()
    except Exception:
        # Fallback for some unstructured data
        surface = low_density.extract_geometry().triangulate()

    surface = clean_mesh(surface)
    return surface


def boolean_difference_mesh(mesh_a: pv.PolyData, mesh_b: pv.PolyData) -> pv.PolyData:
    """
    Perform boolean difference (mesh_a - mesh_b).
    Tries PyVista's boolean_difference first; on failure, uses trimesh boolean.
    Returns the resulting PyVista PolyData.
    """
    # Ensure both meshes are triangles and oriented
    mesh_a = clean_mesh(mesh_a)
    mesh_b = clean_mesh(mesh_b)

    # Try PyVista boolean difference
    try:
        result = mesh_a.boolean_difference(mesh_b, tolerance=1e-6)
        if result is None or result.n_points == 0:
            raise RuntimeError("Boolean difference returned empty mesh.")
        return result
    except Exception as e:
        print(f"PyVista boolean_difference failed: {e}", file=sys.stderr)

    # Fallback: convert to trimesh and use its boolean
    try:
        def to_trimesh(poly: pv.PolyData) -> trimesh.Trimesh:
            verts = np.array(poly.points)
            faces = np.array(poly.faces).reshape(-1, 4)[:, 1:4]
            return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        tm_a = to_trimesh(mesh_a)
        tm_b = to_trimesh(mesh_b)
        tm_a.fix_normals()
        tm_b.fix_normals()

        tm_result = trimesh.boolean.difference(tm_a, tm_b, engine='scad', resolution=100000)
        if tm_result is None or tm_result.vertices.shape[0] == 0 or tm_result.faces.shape[0] == 0:
            raise RuntimeError("Trimesh boolean difference produced no result.")

        verts = np.array(tm_result.vertices)
        faces = np.array(tm_result.faces, dtype=int)
        # Create PyVista mesh from triangular faces
        result = pv.PolyData.from_regular_faces(verts, faces)
        result = clean_mesh(result)
        return result

    except Exception as e:
        print(f"Trimesh boolean difference failed: {e}", file=sys.stderr)
        raise RuntimeError("Both PyVista and Trimesh boolean operations failed.")


def ensure_triangles(poly: pv.PolyData) -> pv.PolyData:
    """
    Ensure the mesh has only triangular faces. If not, triangulate it.
    """
    if isinstance(poly, pv.UnstructuredGrid):
        poly = poly.extract_surface().clean()
    if not poly.is_all_triangles:
        poly = poly.triangulate()
    return poly

def repair_with_pymeshfix(poly: pv.PolyData) -> pv.PolyData:
    """
    Repair a mesh using PyMeshFix to ensure it is watertight/manifold.
    Returns a new PolyData mesh (watertight triangles)
    """
    if pymeshfix is None:
        raise RuntimeError("pymeshfix is not installed for mesh repair.")
    # Extract vertices and faces (as Nx3 indices) from PolyData
    verts = np.array(poly.points)
    faces = np.array(poly.faces).reshape(-1, 4)[:, 1:4]  # remove the leading count (should be 3)
    meshfix = pymeshfix.MeshFix(verts, faces)
    meshfix.repair()
    cleaned = meshfix.mesh  # this is a vtkPolyData
    return pv.PolyData(cleaned)

def safe_boolean_difference(original: pv.PolyData, cutter: pv.PolyData, smoothing_n_iter: int = 15, pass_band: float = 0.8) -> pv.PolyData:
    """
    Subtract 'cutter' from 'original' using boolean difference.
    Performs pre-checks (triangulation, manifoldness) and catches errors.
    smoothing_n_iter: Number of smoothing iterations for the cutter mesh.
    pass_band: Smoothing pass band for the cutter mesh. 0.0 to 2.0. Where 2.0 is preserving geometry.
    Returns the resulting PolyData (or raises if it fails).
    """
    # Ensure triangles
    orig = ensure_triangles(original)
    cut = ensure_triangles(cutter)
    if orig.is_manifold and cut.is_manifold:
        print("Mesh has open or non-manifold edges!")
    
    #cut = cut.smooth_taubin(n_iter=10)
    cut = cut.smooth_taubin(n_iter=smoothing_n_iter, pass_band=pass_band,
                              boundary_smoothing=True, feature_smoothing=False)
    # Perform boolean difference
    try:
        result = orig.boolean_difference(cut, tolerance=1e-5)
        return result
    except Exception as e:
        print("PyVista boolean_difference failed:", e)
        # Attempt repair of cutter if not manifold
        try:
            cut = repair_with_pymeshfix(cut)
            result = orig.boolean_difference(cut, tolerance=1e-5)
            return result
        except Exception as e2:
            print("Retry with PyMeshFix also failed:", e2)
        # As last resort, try trimesh boolean difference if available
        if trimesh is not None:
            print("Attempting trimesh boolean difference as fallback...")
            try:
                # Convert to trimesh
                tm_res = trimesh_boolean(orig, cut)
                # Convert back to PyVista
                verts2 = tm_res.vertices
                faces2 = tm_res.faces
                faces_flat = np.hstack([np.full((len(faces2),1), 3), faces2]).astype(np.int64)
                result = pv.PolyData(verts2, faces_flat)
                return result
            except Exception as e3:
                traceback.print_exc()
                raise RuntimeError("Boolean difference failed with both PyVista and trimesh.") from e3
        else:
            raise RuntimeError("Boolean difference failed and trimesh is not available.") from e
        
def visualize_mesh(mesh: pv.PolyData, title: str = "Mesh"):
    """
    Visualize the mesh. Tries PyVista (interactive window); if it fails, tries Vedo.
    """
    try:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgrey', show_edges=True)
        plotter.add_title(title)
        plotter.show()
    except Exception:
        # Fallback to Vedo
        vedo_plotter = vedo.Plotter()
        vedo_plotter.show(vedo.Mesh(mesh.points, polys=mesh.faces.reshape(-1,4)[:,1:4],
                                   c='lightblue', alpha=0.8), title, interactive=True)


def save_mesh(mesh: pv.PolyData, file_path: str):
    """
    Save mesh to file. Ensures normals are up-to-date.
    """
    mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
    mesh.save(file_path)
    print(f"Saved cleaned mesh to '{file_path}'.")

def filter_and_export_mesh(
    input_stl, input_vtu, output_stl, output_fixed_stl, threshold=0.3, scalars="density"
):
    os.makedirs(os.path.dirname(output_stl), exist_ok=True)
    os.makedirs(os.path.dirname(output_fixed_stl), exist_ok=True)

    print(f"Loading VTU from: {input_vtu}")
    vtu = pv.read(input_vtu)

    if scalars not in vtu.array_names:
        raise ValueError(f"Scalar '{scalars}' not found. Available: {vtu.array_names}")

    thresholded = vtu.threshold(value=threshold, scalars=scalars)
    surf = thresholded.extract_surface()

    print("Exporting surface as STL...")
    surf.save(output_stl)

    print("Cleaning mesh...")
    mesh = trimesh.load_mesh(output_stl)
    meshfix = pymeshfix.MeshFix(mesh)
    meshfix.repair(verbose=False)
    meshfix_mesh = meshfix.mesh
    meshfix_mesh.export(output_fixed_stl)
    
    print("Displaying final mesh...")
    vedo.load(output_fixed_stl).show()

def trimesh_boolean(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Perform boolean difference using trimesh.
    Returns the resulting trimesh object.
    """
    try:
        tm_orig = trimesh.Trimesh(vertices=np.array(mesh_a.points),
                                    faces=np.array(mesh_a.faces).reshape(-1,4)[:,1:4],
                                    process=False)
        tm_cut = trimesh.Trimesh(vertices=np.array(mesh_b.points),
                                    faces=np.array(mesh_b.faces).reshape(-1,4)[:,1:4],
                                    process=False)
        trimesh_verification(tm_orig)
        trimesh_verification(tm_cut)
        result = trimesh.boolean.difference([mesh_a, mesh_b], engine='scad')
        return result
    except Exception as e:
        print(f"Trimesh boolean difference failed: {e}", file=sys.stderr)
        raise RuntimeError("Boolean difference failed with trimesh.") from e
    
def trimesh_verification(mesh: trimesh.Trimesh):
    """
    Verify if the mesh is valid and watertight using trimesh.
    Returns True if valid, False otherwise.
    """
    
    print("Is watertight:", mesh.is_watertight)
    print("Is manifold:", mesh.is_watertight and mesh.is_winding_consistent)
    # Fill small holes
    trimesh.repair.fill_holes(mesh)
    # Fix face normals
    trimesh.repair.fix_normals(mesh)

    # Fix inverted faces
    trimesh.repair.fix_inversion(mesh)
    return mesh


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

def main():
    example = ExamplesCAD.Mitchell_1  # Change to the one you want
    input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example)
    threshold=0.5
    scalars="density"
    no_visualization = False
    # num cells in a patch below this percentage of the largest patch will not be considered for removal
    cells_threshold_percentage = 20 
    inflate_scale_factor = 1.04
    smoothing_n_iter=15
    pass_band=0.8
    # Load meshes
    print("Loading original STL mesh...")
    mesh_original = load_mesh(input_stl)
    print("Loading topology VTU mesh...")
    mesh_volume = load_mesh(input_vtu)

    # Clean original mesh
    #print("Cleaning original mesh...")
    #mesh_original = clean_mesh(mesh_original)

    # Extract low-density region surface
    print(f"Extracting low-density region (threshold = {threshold})...")
        
    #assume the VTU has point or cell data named "density"
    patches_poly = extract_low_density_patches(mesh_volume, scalars, threshold=0.5)
        
    # Split and filter out tiny patches
    patch_list = split_and_filter_patches(patches_poly, cells_threshold_percentage) 
    print(f"Found {len(patch_list)} large void patches.")

    
    # Inflate all patches slightly
    inflated_patches = [inflate_patch(p, scale_factor=inflate_scale_factor) for p in patch_list]
    
    # Perform the subtraction for each patch
    recovered = mesh_original
    for patch in inflated_patches:
        print("Subtracting patch with", patch.n_cells, "cells")
        diagnostic_report(recovered, "recovered")
        diagnostic_report(patch, "patch")

        recovered = safe_boolean_difference(recovered, patch, smoothing_n_iter=smoothing_n_iter, pass_band=pass_band)
        #recovered = recovered.connectivity('largest')

        # Clean intermediate result
        recovered = clean_mesh(recovered)
        #recovered.plot(color='lightblue', show_edges=True)
        

    # Clean the final mesh and ensure largest component
    print("Cleaning final mesh...")
    #result_mesh = clean_mesh(recovered)
    result_mesh = recovered
    # if result_mesh.n_cells > 0:
    #     result_mesh = result_mesh.extract_largest()

    # Visualize final result if enabled
    if not no_visualization:
        print("Visualizing final mesh...")
        visualize_mesh(result_mesh, title="Final Mesh")

    # Save output STL
    save_mesh(result_mesh, output_stl)
    result_mesh = fix_mesh(fp_meshstl=output_stl)
    result_mesh.plot(color='lightblue', show_edges=True)
    save_mesh(result_mesh, output_stl)
    


if __name__ == "__main__":
    main()
