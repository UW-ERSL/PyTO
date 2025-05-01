import pyvista as pv
import numpy as np
import trimesh
import pymeshfix
from plots_examples import *
from plots_demo_1 import *
import trimesh
from fast_simplification import simplify


def diagnostic_report(mesh, name="mesh"):
    report = {}
    report['type'] = type(mesh).__name__

    # Count non-manifold edges
    non_manifold = mesh.extract_feature_edges(
        non_manifold_edges=True, feature_edges=False, boundary_edges=False)
    report['non_manifold_edges'] = non_manifold.n_lines

    # Count boundary (hole) edges
    boundary = mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False)
    report['boundary_edges'] = boundary.n_lines

    # Watertight check
    report['is_watertight'] = report['boundary_edges'] == 0

    # Disconnected parts
    conn = mesh.connectivity()
    report['disconnected_parts'] = len(np.unique(conn['RegionId']))

    #report['n_cells'] = mesh.n_cells
    #report['n_points'] = mesh.n_points

    # Triangle check
    if isinstance(mesh, pv.PolyData):
        report['is_all_triangles'] = mesh.is_all_triangles
    else:
        report['is_all_triangles'] = "N/A (not PolyData)"

    # Bounding box size
    bounds = mesh.bounds
    report['bounds'] = bounds
    diag = np.linalg.norm([bounds[1] - bounds[0],
                           bounds[3] - bounds[2],
                           bounds[5] - bounds[4]])
    report['bounding_box_diag'] = diag

    print(f"\nDiagnostic Report for {name}:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report

def clean_mesh(mesh, name="mesh", hole_fraction=0.05):
    print(f"\nCleaning {name}...")

    mesh = mesh.clean()  # merge close points
    if isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface().clean()
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    mesh = mesh.extract_geometry()  # now guaranteed all faces
    mesh = mesh.clean()

    # 4) Compute normals only if there is actually a polygon mesh
    if mesh.n_faces > 0:
        mesh = mesh.compute_normals(
            consistent_normals=True,
            auto_orient_normals=True
        )

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

def to_trimesh(pvmesh):
    faces = pvmesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=pvmesh.points, faces=faces)

def trimesh_boolean_difference(meshA, meshB):
    try:
        t1 = to_trimesh(meshA)
        t2 = to_trimesh(meshB)
        result = trimesh.boolean.difference([t1, t2], engine='blender')
        if result:
            return pv.PolyData(result.vertices, np.hstack([np.full((len(result.faces), 1), 3), result.faces]))
    except Exception as e:
        print("Trimesh boolean failed:", e)
    return None

def robust_boolean_difference(fileA, fileB):
    # Load meshes
    meshA = pv.read(fileA)
    meshB = pv.read(fileB)

    # Diagnostics
    diagnostic_report(meshA, "Mesh A (raw)")
    diagnostic_report(meshB, "Mesh B (raw)")

    # Clean meshes
    meshA = clean_mesh(meshA, "Mesh A")
    meshB = clean_mesh(meshB, "Mesh B")

    # Re-run diagnostics
    diagnostic_report(meshA, "Mesh A (cleaned)")
    diagnostic_report(meshB, "Mesh B (cleaned)")

    try:
        # Boolean Difference
        print("\nAttempting VTK Boolean Difference...")
        result = meshA.boolean_difference(meshB, tolerance=1e-5)
        if result is None or result.n_points == 0:
            raise RuntimeError("VTK boolean returned empty mesh.")
        print("VTK Boolean successful.")
        return result
    except Exception as e:
        print("VTK Boolean failed:", e)
        print("Attempting Trimesh fallback...")
        return trimesh_boolean_difference(meshA, meshB)

def using_clipping_surfaces():
    """"
    This example demonstrates how to use clipping surfaces to create a cut patch
    from a cube and then smooth it before re-subtracting it from the original cube.
    """
    cube = pv.Sphere()
    sphere = pv.Sphere(center=(0.5, 0, 0))
    # Clip away the part of cube *outside* the sphere, keeping only the inside:
    patch = cube.clip_surface(sphere, invert=True)  # keep part *inside* sphere
    patch.plot(color='red', show_edges=True)
    patch = patch.extract_surface().triangulate().clean()
    patch.plot(color='red', show_edges=True)

    # 1) Extract the “cut patch”
    patch = cube.boolean_intersection(sphere).extract_surface().triangulate().clean()

    # 2) Smooth it
    patch_smooth = patch.smooth_taubin(n_iter=15, pass_band=0.05,
                                    boundary_smoothing=True,
                                    feature_smoothing=False)
    patch_smooth.plot(color='red', show_edges=True)

    # 3) Re-subtract using the smoothed patch
    cube_cut = cube.boolean_difference(patch_smooth, tolerance=1e-9).clean()
    cube_cut.plot(color='red', show_edges=True)

    # 4) (Optional) fill any holes and final clean
    cube_cut = cube_cut.fill_holes(hole_size=1e-3).clean()

    # 5) Visualize
    pl = pv.Plotter()
    pl.add_mesh(cube_cut, color='lightblue', show_edges=True)
    pl.show()

def marching_cubes_isosurf(voxel_cut, pad=1):
    b = np.array(voxel_cut.bounds).reshape(3,2)
    # add one cell of padding on each side:
    pad_size = [(b[i,1] - b[i,0]) / 49 for i in range(3)]
    b[:,0] -= pad_size
    b[:,1] += pad_size

    # build a 50³ grid over the _padded_ bounds
    nx = ny = nz = 50
    spacing = [(b[i,1] - b[i,0])/(nx-1) for i in range(3)]
    vol = pv.ImageData(dimensions=(nx,ny,nz),
                       spacing=spacing,
                       origin=(b[0,0],b[1,0],b[2,0]))

    dg = vol.compute_implicit_distance(voxel_cut, inplace=False)
    dg["dist"] = dg.point_data["implicit_distance"]
    mc = dg.contour([0.0], scalars="dist")
    return mc.triangulate().clean()

def repair_meshfix(mesh: pv.PolyData) -> pv.PolyData:
    # 1) Ensure we have a PolyData surface
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate().clean()

    # 2) Pull out points & faces
    verts = mesh.points
    # faces come in a flat array: [3, i, j, k, 3, i2, j2, k2, ...]
    face_array = mesh.faces.reshape(-1, 4)[:, 1:4]

    mfix = pymeshfix.MeshFix(verts, face_array)
    mfix.repair()
    # Access the repaired mesh with vtk
    #mesh = mfix.mesh
    # Or, access the resulting arrays directly from the object
    v2 = mfix.v # numpy np.float64 array
    f2 = mfix.f
    # 5) Rebuild a clean PyVista PolyData
    new_faces = np.hstack([np.full((f2.shape[0],1), 3, dtype=int), f2])
    repaired = pv.PolyData(v2, new_faces)
    repaired.compute_normals(auto_orient_normals=True, inplace=True)
    return repaired


def advanced_repair_meshfix(mesh: pv.PolyData) -> pv.PolyData:
    # 1) Ensure we have a PolyData surface
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate().clean()

    # 2) Pull out points & faces
    verts = mesh.points
    # faces come in a flat array: [3, i, j, k, 3, i2, j2, k2, ...]
    face_array = mesh.faces.reshape(-1, 4)[:, 1:4]

    # # 3) Feed into the PyMeshFix “PyTMesh” engine
    mfix = pymeshfix._meshfix.PyTMesh(False)  
    mfix.load_array(verts, face_array)  

    #mfix.join_closest_components()

    # Fill holes
    mfix.fill_small_boundaries(refine=True)
    print('There are {:d} boundaries'.format(mfix.boundaries()))

    # Clean (removes self intersections)
    #mfix.clean(max_iters=10, inner_loops=3)

    # Check mesh for holes again
    print('There are {:d} boundaries'.format(mfix.boundaries()))

    #4) Pull repaired data back out
    v2, f2 = mfix.return_arrays()  # v2.shape=(n_vert,3), f2.shape=(n_faces,3)

    # 5) Rebuild a clean PyVista PolyData
    new_faces = np.hstack([np.full((f2.shape[0],1), 3, dtype=int), f2])
    repaired = pv.PolyData(v2, new_faces)
    repaired.compute_normals(auto_orient_normals=True, inplace=True)
    return repaired


if __name__ == "__main__":

    example = ExamplesCAD.Mitchell_1  # Change to the one you want
    input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example)
    # Precompute an implicit distance field on the cut surface
    original = pv.read(input_stl).triangulate().compute_normals()
    voxels = pv.read(input_vtu)
    negative_voxels = extract_low_density_patches(voxels, 'density', threshold=0.5)
    voxel_cut = negative_voxels.extract_surface().triangulate().clean()

    marching_cubes =  marching_cubes_isosurf(voxel_cut, original)

    smoothed = marching_cubes.smooth_taubin(
    n_iter=20,
    pass_band=0.1,
    boundary_smoothing=True,    # hold boundary (coincident edges) fixed
    feature_smoothing=False      # keep sharp folds intact
    )

    #inflated = inflate_patch(smoothed, scale_factor=1.02)
    inflated = directional_inflate_patch(smoothed, scale_factor=1.05) 

    # --- 4) Visualize everything together in one Plotter ---
    pl = pv.Plotter()
    pl.add_text("Marching Cubes Reconstruction", font_size=14)
    pl.add_mesh(original, color="lightblue", opacity=0.3, show_edges=True)
    pl.add_mesh(voxel_cut, style="wireframe", color="black", opacity=0.2)
    pl.add_mesh(inflated, color="red", show_edges=False)
    pl.show()
    #closed_patches = fix_with_meshfix(inflated)
    
    save_mesh(inflated, output_stl)
    #result_mesh = fix_mesh(fp_meshstl=output_stl)
    
    # 1. Load your complex mesh
    mesh = trimesh.load(output_stl)
    # 2. Decide on a target face count
    target_faces = 1000  # e.g. reduce to ~1k triangles
    # 3. Perform quadratic decimation
    simple = mesh.simplify_quadric_decimation(face_count = target_faces)
    # 4. (Optional) If you want the “outer shell” only:
    # simple = simple.convex_hull
    # 5. Export simplified STL
    export_stl = 'simple_model.stl'
    simple.export(export_stl)

    # mesh = pv.read(output_stl)
    # dec = mesh.decimate_pro(0.99, preserve_topology=True, splitting = False, boundary_vertex_deletion = False)  # Keep 10% of triangles
    # dec.save(export_stl)

    inflated = pv.read(export_stl)
    inflated.plot(color='red', show_edges=True)
    # Split and filter out tiny patches
    patch_list = split_and_filter_patches(inflated, cells_threshold_percentage=10) 
    print(f"Found {len(patch_list)} large void patches.")
    
    # Inflate all patches slightly
    
    
    inflated_patches = [inflate_patch(p, scale_factor=1.07) for p in patch_list]
    
    # Perform the subtraction for each patch
    recovered = original
    for patch in inflated_patches:
        print("Subtracting patch with", patch.n_cells, "cells")
        diagnostic_report(patch, "patch")
        # Ensure triangles
        recovered = ensure_triangles(recovered)
        recovered = clean_mesh(recovered)

        patch = ensure_triangles(patch)
        #recovered = recovered.triangulate().subdivide(2).clean()
        patch = patch.triangulate().subdivide(2).clean()
        patch = clean_mesh(patch)


        #recovered.plot(color='lightblue', show_edges=True)
        #patch.plot(color='red', show_edges=True)
        recovered = safe_boolean_difference(recovered, patch, smoothing_n_iter=10, pass_band=0.57)
        #recovered = recovered.boolean_difference(patch, tolerance=1e-5).clean()
        #recovered.plot(color='lightblue', show_edges=True)
        #recovered = recovered.connectivity('largest')
        # Clean intermediate result
        save_mesh(recovered, output_stl)
        recovered = fix_mesh(fp_meshstl=output_stl)
        
        #recovered = advanced_repair_meshfix(recovered)

    #closed_patches = advanced_repair_meshfix(recovered)
    #recovered = repair_meshfix(recovered)

    
    # Force it to PolyData
    
    save_mesh(recovered, output_stl)
    recovered = fix_mesh(fp_meshstl=output_stl)
    closed_patches = recovered.extract_surface().triangulate().clean()
    closed_patches.plot(color='red', show_edges=True)
    plotter = pv.Plotter()
    plotter.add_mesh(closed_patches, color='lightblue')
    plotter.add_mesh(voxel_cut, color='red', show_edges=True, style='wireframe')
    plotter.show()
    closed_patches.plot(color='lightblue')

    #final = original.boolean_difference(closed_patches, tolerance=1e-5).clean()
    #final.plot()

