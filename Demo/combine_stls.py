import pyvista as pv
from plots_examples import *
from cad_recovery import *

def cube_sphere_example():
    # Create and save the cube
    cube = pv.Cube().triangulate().subdivide(3)
    cube.save("cube.stl", binary=True)

    # Create and save the sphere
    sphere = pv.Sphere(radius=0.4)
    sphere.save("sphere.stl", binary=True)

    # Read the saved STLs
    mesh_cube = pv.read("cube.stl")
    mesh_sphere = pv.read("sphere.stl")

    # Combine them
    combined = mesh_cube.merge(mesh_sphere)

    # Save the combined mesh as STL
    combined.save("combined.stl")

    clipped = combined.clip(normal='z', invert=False)  # Cut away top half
    clipped.plot(show_edges=True, color='lightblue')


    # Slice through the Z-plane at the center
    sliced = combined.slice(normal='z')

    # Plot the slice and the full mesh together
    plotter = pv.Plotter()
    plotter.add_mesh(combined, color='lightblue', opacity=0.2, show_edges=True)
    plotter.add_mesh(sliced, color='red', line_width=3)
    plotter.show()


    # Plot the difference mesh
    result = mesh_cube.boolean_difference(mesh_sphere)
    result.plot(show_edges=True, color='lightblue')

# Working perfectly!
def BasePlateOptimizableVol_example():
    example = ExamplesCAD.BasePlateOptimizableVol  # Change to the one you want
    input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example)
    # Precompute an implicit distance field on the cut surface
    original = pv.read(input_stl)
    voxels = pv.read(input_vtu)
    negative_voxels = extract_low_density_patches(voxels, 'density', threshold=0.5)
    voxel_cut = negative_voxels.extract_surface().triangulate().clean()
    marching_cubes =  marching_cubes_isosurf(voxel_cut, original)

    smoothed = marching_cubes.smooth_taubin(
    n_iter=20,
    pass_band=0.3,
    boundary_smoothing=True,    # hold boundary (coincident edges) fixed
    feature_smoothing=False      # keep sharp folds intact
    )
   
    
    patch_list = split_and_filter_patches(smoothed, cells_threshold_percentage=10) 
    print(f"Found {len(patch_list)} large void patches.")
    combined = original.merge(patch_list)
    combined = combined.extract_surface().triangulate()
    combined.save("BasePlateOptimizableVol_recovered.stl")

    clipped = combined.clip(normal='z', invert=False)  # Cut away top half
    clipped.plot(show_edges=True, color='lightblue')


def EdgeCantilever_example():
    example = ExamplesCAD.EdgeCantileverDemo  # Change to the one you want
    input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example)
    # Precompute an implicit distance field on the cut surface
    original = pv.read(input_stl)
    voxels = pv.read(input_vtu)
    negative_voxels = extract_low_density_patches(voxels, 'density', threshold=0.5)
    voxel_cut = negative_voxels.extract_surface().triangulate().clean()
    marching_cubes =  marching_cubes_isosurf(voxel_cut, original)

    smoothed = marching_cubes.smooth_taubin(
    n_iter=20,
    pass_band=0.3,
    boundary_smoothing=True,    # hold boundary (coincident edges) fixed
    feature_smoothing=False      # keep sharp folds intact
    )
   
    
    patch_list = split_and_filter_patches(smoothed, cells_threshold_percentage=10) 
    print(f"Found {len(patch_list)} large void patches.")
    combined = original.merge(patch_list)
    combined = combined.extract_surface().triangulate()
    combined.save("final_recovered.stl")

    clipped = combined.clip(normal='z', invert=False)  # Cut away top half
    clipped.plot(show_edges=True, color='lightblue')

def combine_inner_TO_w_outer_STL():
    example = ExamplesCAD.EdgeCantileverDemo  # Change to the one you want
    input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example)
    original = pv.read(input_stl)
    voxels = pv.read(input_vtu)
    negative_voxels = extract_low_density_patches(voxels, 'density', threshold=0.5)
    voxel_cut = negative_voxels.extract_surface().triangulate().clean()
    # Precompute an implicit distance field on the cut surface
    cut = voxel_cut.compute_implicit_distance(original, inplace=False)
    # cut.point_data['implicit_distance'] now holds signed distance to the original surface
    
    # For each cell, take the max absolute distance of its points
    pts = cut.points
    dists = cut.point_data['implicit_distance']
    # 3) Build the (n_cells × 3) array of point indices:
    # PyVista stores faces as [3, i0, j0, k0, 3, i1, j1, k1, …].
    faces = cut.faces.reshape((-1, 4))    # one row per cell: [3, pt0, pt1, pt2]
    cell_pt_ids = faces[:, 1:4]           # drop the leading “3”

    # 4) For each cell, take the max absolute distance of its vertices:
    max_dist = np.max(np.abs(dists[cell_pt_ids]), axis=1)

    # 5) Split into coincident vs new cells
    tol = 1e-3
    coincident_ids = np.where(max_dist < tol)[0]
    all_ids       = np.arange(cut.n_cells)
    remainder_ids = np.setdiff1d(all_ids, coincident_ids)

    coincident = cut.extract_cells(coincident_ids)
    coincident.plot()
    remainder  = cut.extract_cells(remainder_ids)
    

    # turn it into a surface PolyData again
    remainder_poly = (
        remainder
        .extract_surface()   # collapse to PolyData
        .triangulate()       # ensure triangles
        .clean()             # merge any dup verts
    )
    #remainder_poly.plot(color='red', show_edges=True)
    marching_cubes =  marching_cubes_isosurf(remainder_poly, original)

    smoothed = marching_cubes.smooth_taubin(
    n_iter=20,
    pass_band=0.1,
    boundary_smoothing=True,    # hold boundary (coincident edges) fixed
    feature_smoothing=False      # keep sharp folds intact
    )
    #smoothed.plot(color='red', show_edges=True)

    inflated = inflate_patch(smoothed, scale_factor=1.0)
    # --- 4) Visualize everything together in one Plotter ---
    pl = pv.Plotter()
    pl.add_text("Marching Cubes Reconstruction", font_size=14)
    pl.add_mesh(original, color="lightblue", opacity=0.3, show_edges=True)
    pl.add_mesh(voxel_cut, style="wireframe", color="black", opacity=0.2)
    pl.add_mesh(inflated, color="red", show_edges=False)
    pl.add_mesh(coincident, color="green", style="wireframe", show_edges=False)
    pl.show()
    
    combined = coincident.merge(inflated, merge_points=True, main_has_priority=False, tolerance = 0.0)
    closed_patches = advanced_repair_meshfix(combined) 
    closed_patches = closed_patches.smooth_taubin(
    n_iter=20,
    pass_band=0.1,
    boundary_smoothing=True,    # hold boundary (coincident edges) fixed
    feature_smoothing=False      # keep sharp folds intact
    )
    closed_patches.plot(color='red', show_edges=True)
    result = original.boolean_difference(closed_patches)
    result.plot()

    ######

if __name__ == "__main__":
    #cube_sphere_example # Working perfectly!
    #BasePlateOptimizableVol_example() # Working perfectly!
    #EdgeCantilever_example()
    combine_inner_TO_w_outer_STL()