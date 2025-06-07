import pyvista as pv
import numpy as np
import os
import traceback
import enum

class ExamplesCAD(enum.Enum):
    EdgeCantileverDemo = enum.auto()
    BliskSectionWithBlade = enum.auto()
    KnuckleAssembly = enum.auto()
    Mitchell_1 = enum.auto()
    ShortCantileverMidLoad = enum.auto()
    CantileverMidLoad = enum.auto()
    TwoBar = enum.auto()
    MBBB = enum.auto()
    DistributedLoad = enum.auto()
    LBracketMidLoad = enum.auto()
    VerticalBar = enum.auto()
    FilletedBeam = enum.auto()
    ThreeHoleBracket = enum.auto()
    CircularPlateHole = enum.auto()

def get_example_cad(example: ExamplesCAD):
    fp_stl_folder = "Models/"  # Models folder in PyTO directory
    fp_output_folder = "output/"  # Create output folder in PyTO directory
    
    if example == ExamplesCAD.EdgeCantileverDemo:
        str_output_name = "EdgeCantilever"
        fp_original_stl = fp_stl_folder + "EdgeCantilever/EdgeCantilever.STL"
        fp_vtu_mesh = fp_stl_folder + "EdgeCantilever/EdgeCantilever.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.BliskSectionWithBlade:
        fp_original_stl = fp_stl_folder + "Saketh/BliskSectionWithBlade2test.STL"
        fp_vtu_mesh = fp_stl_folder + "Saketh/BliskSectionWithBlade2test.vtu"
        fp_outputstlpath = fp_output_folder + "BliskSectionWithBlade2_internal_voids.stl"
    elif example == ExamplesCAD.KnuckleAssembly:
        fp_original_stl = fp_stl_folder + "KnuckleAssembly/KnuckleAssembly.STL"
        fp_vtu_mesh = fp_stl_folder + "KnuckleAssembly/KnuckleAssembly.vtu"
        fp_outputstlpath = fp_output_folder + "KnuckleAssembly_internal_voids.stl"
    elif example == ExamplesCAD.Mitchell_1:
        str_output_name = "Mitchell_1"
        fp_original_stl = fp_stl_folder + "Mitchell/Mitchell.STL"
        fp_vtu_mesh = fp_stl_folder + "Mitchell/Mitchell_1.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.ShortCantileverMidLoad:
        str_output_name = "ShortCantileverMidLoad"
        fp_original_stl = fp_stl_folder + "ShortCantilever/ShortCantilever.STL"
        fp_vtu_mesh = fp_stl_folder + "ShortCantilever/ShortCantileverMidLoad.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.CantileverMidLoad:
        str_output_name = "CantileverMidLoad"
        fp_original_stl = fp_stl_folder + "Cantilever/Cantilever.STL"
        fp_vtu_mesh = fp_stl_folder + "Cantilever/CantileverMidLoad.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.TwoBar:
        str_output_name = "TwoBar"
        fp_original_stl = fp_stl_folder + "TwoBar/TwoBar.STL"
        fp_vtu_mesh = fp_stl_folder + "TwoBar/TwoBar.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.MBBB:
        str_output_name = "MBBB"
        fp_original_stl = fp_stl_folder + "MBBB/MBBB.STL"
        fp_vtu_mesh = fp_stl_folder + "MBBB/MBBB.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.DistributedLoad:
        str_output_name = "DistributedLoad"
        fp_original_stl = fp_stl_folder + "DistributedLoad/DistributedLoad.STL"
        fp_vtu_mesh = fp_stl_folder + "DistributedLoad/DistributedLoad.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.LBracketMidLoad:
        str_output_name = "LBracketMidLoad"
        fp_original_stl = fp_stl_folder + "LBracket/LBracket.STL"
        fp_vtu_mesh = fp_stl_folder + "LBracket/LBracketMidLoad.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.VerticalBar:
        str_output_name = "VerticalBar"
        fp_original_stl = fp_stl_folder + "VerticalBar/VerticalBar.STL"
        fp_vtu_mesh = fp_stl_folder + "VerticalBar/VerticalBar.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.FilletedBeam:
        str_output_name = "FilletedBeam"
        fp_original_stl = fp_stl_folder + "FilletedBeam/FilletedBeam.STL"
        fp_vtu_mesh = fp_stl_folder + "FilletedBeam/FilletedBeam.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.ThreeHoleBracket:
        str_output_name = "ThreeHoleBracket"
        fp_original_stl = fp_stl_folder + "ThreeHoleBracket/ThreeHoleBracket.STL"
        fp_vtu_mesh = fp_stl_folder + "ThreeHoleBracket/ThreeHoleBracket.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    elif example == ExamplesCAD.CircularPlateHole:
        str_output_name = "CircularPlateHole"
        fp_original_stl = fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL"
        fp_vtu_mesh = fp_stl_folder + "CircularPlateHole/CircularPlateHole.vtu"
        fp_outputstlpath = fp_output_folder + f'{str_output_name}_internal_voids.stl'
    else:
        raise ValueError(f"Unknown example: {example}")
    
    return fp_original_stl, fp_vtu_mesh, fp_outputstlpath

try:
    import trimesh
    import pymeshfix
except ImportError as e:
    print(f"Required library missing: {e}")
    raise

def to_trimesh(pv_mesh):
    """Convert PyVista mesh to Trimesh."""
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)

def extract_low_density_patches(vtu, density_field='density', threshold=0.5):
    """
    Threshold the UnstructuredGrid 'vtu' on the given 'density_field' scalar.
    Return the cells of the low-density regions below 'threshold'.
    """
    # Apply a cell threshold: keep cells with density < threshold
    low_density_region = vtu.threshold(value=threshold, scalars=density_field,
                                     invert=True)  # invert=True selects cells below threshold
    
    print(f"Extracted low density regions: {low_density_region.n_cells} cells")
    return low_density_region

def marching_cubes_isosurf(voxel_cut, reference_mesh=None, pad=1, visualize=True):
    """
    Generate an isosurface using marching cubes algorithm with visualization.
    Optionally uses a reference mesh's bounds for context.
    """
    print("Constructing marching cubes isosurface...")
    
    # Use voxel_cut bounds by default
    b = np.array(voxel_cut.bounds).reshape(3,2)
    
    # If reference mesh provided, consider its bounds too
    if reference_mesh is not None:
        ref_bounds = np.array(reference_mesh.bounds).reshape(3,2)
        # Expand bounds to include reference mesh
        for i in range(3):
            b[i,0] = min(b[i,0], ref_bounds[i,0])
            b[i,1] = max(b[i,1], ref_bounds[i,1])
    
    # add padding on each side:
    pad_size = [(b[i,1] - b[i,0]) / 49 * pad for i in range(3)]
    b[:,0] -= pad_size
    b[:,1] += pad_size

    # build a 100³ grid over the padded bounds
    nx = ny = nz = 100
    spacing = [(b[i,1] - b[i,0])/(nx-1) for i in range(3)]
    vol = pv.ImageData(dimensions=(nx,ny,nz),
                     spacing=spacing,
                     origin=(b[0,0],b[1,0],b[2,0]))
    
    # Visualize the initial grid
    if visualize:
        print("Visualizing the initial grid structure...")
        p = pv.Plotter()
        
        # Add the original mesh if available
        if reference_mesh is not None:
            p.add_mesh(reference_mesh, color='lightblue', opacity=0.3, show_edges=True)
        
        # Add the voxel cut
        p.add_mesh(voxel_cut, color='red', opacity=0.5, show_edges=True)
        
        # Show the bounds of our grid
        outline = vol.outline()
        p.add_mesh(outline, color='black', line_width=2)
        
        # Add grid points visualization (show only a subset for clarity)
        grid_points = vol.points.reshape(nx, ny, nz, 3)
        sample_points = grid_points[::5, ::5, ::5].reshape(-1, 3)
        point_cloud = pv.PolyData(sample_points)
        p.add_mesh(point_cloud, color='blue', point_size=5, render_points_as_spheres=True)
        
        p.add_title("Marching Cubes: Initial Grid Structure")
        p.show()

    # Compute the distance field
    print("Computing implicit distance field...")
    dg = vol.compute_implicit_distance(voxel_cut, inplace=False)
    dg["dist"] = dg.point_data["implicit_distance"]
    
    # Visualize the distance field
    if visualize:
        print("Visualizing distance field...")
        p = pv.Plotter()
        
        # Add a slice through the distance field
        slice_x = dg.slice(normal='x')
        slice_y = dg.slice(normal='y')
        slice_z = dg.slice(normal='z')
        
        # Show the slices colored by distance
        p.add_mesh(slice_x, scalars="dist", cmap="RdBu", opacity=0.7)
        p.add_mesh(slice_y, scalars="dist", cmap="RdBu", opacity=0.7)
        p.add_mesh(slice_z, scalars="dist", cmap="RdBu", opacity=0.7)
        
        # Add the voxel cut outline
        p.add_mesh(voxel_cut, color='black', opacity=0.2, style='wireframe', line_width=1)
        
        p.add_title("Marching Cubes: Distance Field Visualization")
        p.show()

    # Extract the isosurface
    print("Extracting zero-isosurface...")
    mc = dg.contour([0.0], scalars="dist")
    result = mc.triangulate().clean()
    
    # Visualize the raw isosurface result
    if visualize:
        print("Visualizing raw marching cubes result...")
        p = pv.Plotter()
        
        # Add the original mesh if available
        if reference_mesh is not None:
            p.add_mesh(reference_mesh, color='lightblue', opacity=0.2, show_edges=True)
        
        # Add the isosurface
        p.add_mesh(result, color='green', opacity=0.7, show_edges=True)
        
        p.add_title("Marching Cubes: Raw Isosurface Result")
        p.show()
    
    print(f"Marching cubes isosurface: {result.n_cells} cells, {result.n_points} points")
    return result

def split_and_filter_patches(all_low_density_elems, cells_threshold_percentage=5):
    """
    Split the combined patches into separate bodies and filter out small ones.
    Returns a list of separate void patches.
    """
    patches = all_low_density_elems.split_bodies()  # This returns a MultiBlock
    if patches.n_blocks == 0:
        print("Warning: No patches found after splitting")
        return []
        
    cell_counts = [patch.n_cells for patch in patches]
    min_cells = cells_threshold_percentage * 0.01 * max(cell_counts)
    filtered_patches = [patch for patch in patches if patch.n_cells > min_cells]
    patches_list = list(filtered_patches)
    print(f"Found {len(patches_list)} significant patches")
    
    # Print information about each patch
    for i, patch in enumerate(patches_list):
        print(f"Patch {i+1}: {patch.n_cells} cells, {patch.n_points} points")
        
    return patches_list

def clean_mesh(mesh, name="mesh", hole_fraction=0.01, feature_angle=30):
    """
    Enhanced mesh cleaning with more conservative parameters to preserve detail.
    """
    print(f"\nCleaning {name}...")

    # Less aggressive merging of points to preserve detail
    mesh = mesh.clean(tolerance=1e-6, absolute=False)
    
    if isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface().clean()
    
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    mesh = mesh.extract_geometry()
    
    # Remove small disconnected pieces but preserve main structure
    mesh = mesh.connectivity('largest')
    
    # Less aggressive smoothing to preserve detail
    mesh = mesh.smooth(n_iter=10, relaxation_factor=0.1)
    
    # Compute normals with compatible parameters
    if mesh.n_cells > 0:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=True,
            consistent_normals=True,
            auto_orient_normals=True,
            feature_angle=feature_angle
        )

    # Fill holes with improved parameters - smaller hole size to preserve detail
    bounds = mesh.bounds
    diag = np.linalg.norm([bounds[1] - bounds[0],
                         bounds[3] - bounds[2],
                         bounds[5] - bounds[4]])
    hole_size = diag * hole_fraction
    mesh = mesh.fill_holes(hole_size)
    
    # Final light cleaning
    mesh = mesh.clean(tolerance=1e-6, absolute=False)
    
    return mesh

def enhanced_smooth(mesh, iterations=50, pass_band=0.2):
    """
    Apply enhanced Taubin smoothing to preserve structures.
    """
    smoothed = mesh.smooth_taubin(
        n_iter=iterations,
        pass_band=pass_band,
        boundary_smoothing=True,
        feature_smoothing=True,
        feature_angle=15
    )
    
    # Minimal additional relaxation smoothing
    smoothed = smoothed.smooth(n_iter=5, relaxation_factor=0.05)
    
    return smoothed

def advanced_repair_meshfix(mesh):
    """
    Advanced mesh repair using PyMeshFix with careful preservation of structure.
    """
    # Ensure we have a PolyData surface
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate().clean()

    # Pull out points & faces
    verts = mesh.points
    face_array = mesh.faces.reshape(-1, 4)[:, 1:4]

    # Feed into the PyMeshFix "PyTMesh" engine
    mfix = pymeshfix._meshfix.PyTMesh(False)  
    mfix.load_array(verts, face_array)  

    # Fill small boundaries - using compatible parameters
    mfix.fill_small_boundaries(refine=True)  # Remove nsize parameter
    print('There are {:d} boundaries'.format(mfix.boundaries()))

    # Pull repaired data back out
    v2, f2 = mfix.return_arrays()

    # Rebuild a clean PyVista PolyData
    new_faces = np.hstack([np.full((f2.shape[0],1), 3, dtype=int), f2])
    repaired = pv.PolyData(v2, new_faces)
    repaired.compute_normals(auto_orient_normals=True, inplace=True)
    return repaired

def post_process_final_mesh(mesh):
    """
    Post-process the final mesh to remove any remaining artifacts.
    """
    # Convert to PolyData if not already
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    # Ensure all triangles
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    # Remove any small isolated pieces
    mesh = mesh.connectivity('largest')

    # Final smoothing
    mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.09)

    # Fix self-intersections
    mesh = advanced_repair_meshfix(mesh)

    return mesh

def process_void_patch(patch, original_mesh, boundary_threshold=0.01, shrink_factor=0.9):
    """
    Enhanced process for a single void patch to prepare it for boolean operations.
    """
    # Store original patch for comparison
    original_patch = patch.copy()
    
    # Initial light cleaning
    patch = patch.extract_surface().triangulate().clean(tolerance=1e-6)
    
    # Apply minimal smoothing
    patch = enhanced_smooth(patch, iterations=50, pass_band=0.2)
    
    # Clean and repair with light touch
    patch = clean_mesh(patch, "void patch", feature_angle=15, hole_fraction=0.01)
    
    # Apply post-processing to the void patch
    print("Applying post-processing to void patch...")
    patch_before_postprocess = patch.copy()
    
    try:
        patch = post_process_final_mesh(patch)
        print(f"Post-processed void patch: {patch.n_cells} cells, {patch.n_points} points")
    except Exception as e:
        print(f"Post-processing failed: {e}. Using patch before post-processing.")
        patch = patch_before_postprocess
    
    # Visualize before and after post-processing
    print("Visualizing before/after post-processing...")
    p = pv.Plotter(shape=(1, 2))
    
    p.subplot(0, 0)
    p.add_mesh(patch_before_postprocess, color='red', opacity=0.8, show_edges=True)
    p.add_title("Before Post-Processing")
    
    p.subplot(0, 1)
    p.add_mesh(patch, color='green', opacity=0.8, show_edges=True)
    p.add_title("After Post-Processing")
    
    p.show()
    
    # Check if the patch is too close to the boundary
    try:
        # More careful Delaunay to preserve structure
        print("Attempting Delaunay triangulation...")
        delaunay = patch.delaunay_3d(alpha=1.5)
        
        # Check if Delaunay produced a valid result
        if delaunay.n_cells == 0:
            print("Delaunay produced empty result. Using original patch.")
            delaunay = patch
        else:
            delaunay = delaunay.extract_surface().triangulate().clean()
            delaunay = clean_mesh(delaunay, "delaunay void", hole_fraction=0.01)
            
            # Only apply advanced repair if we have a good mesh
            if delaunay.n_cells > 0:
                try:
                    delaunay = advanced_repair_meshfix(delaunay)
                except Exception as repair_e:
                    print(f"Mesh repair failed: {repair_e}. Using unrepaired delaunay.")
                    
    except Exception as e:
        print(f"Delaunay failed: {e}. Using original patch.")
        delaunay = patch
    
    # Convert to trimesh with careful repair and validation
    try:
        tm_void = to_trimesh(delaunay)
        
        # Validate the mesh has faces
        if len(tm_void.faces) == 0:
            print("Warning: Converted mesh has no faces!")
            return None
        
        # Careful trimesh repair
        print("Applying trimesh repairs...")
        trimesh.repair.fill_holes(tm_void)
        trimesh.repair.fix_normals(tm_void)
        trimesh.repair.fix_inversion(tm_void)
        
        # Remove degenerate faces - fixed for compatibility with newer Trimesh versions
        try:
            if hasattr(tm_void, 'remove_degenerate_faces'):
                tm_void.remove_degenerate_faces()
            else:
                
                nondegenerate = tm_void.nondegenerate_faces()
                tm_void.update_faces(nondegenerate)
        except Exception as e:
            print(f"Warning: Could not remove degenerate faces: {e}")
        
        # Check if mesh is valid for boolean operations
        if not tm_void.is_volume:
            print("Warning: Mesh is not a valid volume. Attempting to make it watertight...")
            try:
                # Try to make it watertight
                if not tm_void.is_watertight:
                    trimesh.repair.fill_holes(tm_void)
                    
                # If still not a volume, try convex hull as last resort
                if not tm_void.is_volume:
                    print("Using convex hull as fallback...")
                    tm_void = tm_void.convex_hull
                    
            except Exception as vol_e:
                print(f"Could not create valid volume: {vol_e}")
                return None
        
        # Final validation
        if not tm_void.is_watertight:
            print("Warning: Final void mesh is not watertight")
            
        print(f"Created void mesh: {len(tm_void.faces)} faces, {len(tm_void.vertices)} vertices")
        print(f"Is volume: {tm_void.is_volume}, Is watertight: {tm_void.is_watertight}")
        
        # Adjust void if it's close to the boundary
        tm_void = adjust_void_for_boundary(tm_void, original_mesh, 
                                          boundary_threshold=boundary_threshold,
                                          shrink_factor=shrink_factor)
        
        return tm_void
        
    except Exception as e:
        print(f"Error creating trimesh: {e}")
        return None

def validate_void_patch(void_mesh, original_mesh):
    """
    Check if void mesh is properly positioned relative to the original mesh.
    """
    # Convert to numpy arrays for bounds checking
    void_points = np.array(void_mesh.vertices)
    orig_points = np.array(original_mesh.points)
    
    # Get bounds
    void_min = void_points.min(axis=0)
    void_max = void_points.max(axis=0)
    orig_min = orig_points.min(axis=0)
    orig_max = orig_points.max(axis=0)
    
    # Check if completely outside
    for dim in range(3):
        if void_max[dim] < orig_min[dim] or void_min[dim] > orig_max[dim]:
            return False
    
    return True

def scale_void_for_subtraction(void_mesh, original_mesh, z_scale_factor=1.2):
    """
    Scale a void mesh in the z-direction to ensure it fully penetrates the original mesh.
    More conservative scaling to avoid over-deforming the structure.
    """
    print(f"Scaling void mesh by factor {z_scale_factor} in z-direction...")
    
    # Get the center of the void mesh
    void_center = np.mean(void_mesh.vertices, axis=0)
    
    # Create a scaling matrix that only scales in z-direction
    scale_matrix = np.eye(4)
    scale_matrix[2, 2] = z_scale_factor  # Scale z-dimension
    
    # Create a translation matrix to center the mesh at origin before scaling
    to_origin = np.eye(4)
    to_origin[:3, 3] = -void_center
    
    # Create a translation matrix to move the mesh back after scaling
    from_origin = np.eye(4)
    from_origin[:3, 3] = void_center
    
    # Apply the transformations: move to origin, scale, move back
    void_mesh_scaled = void_mesh.copy()
    void_mesh_scaled.apply_transform(from_origin @ scale_matrix @ to_origin)
    
    # Ensure the scaled void spans beyond the original mesh in z-direction
    # This makes sure it fully penetrates through the part
    orig_bounds = np.array(original_mesh.bounds).reshape(3, 2)
    void_bounds = void_mesh_scaled.bounds
    
    # If needed, translate the void to ensure it extends beyond the original in z
    z_min_orig = orig_bounds[2, 0]
    z_max_orig = orig_bounds[2, 1]
    z_min_void = void_bounds[0][2]
    z_max_void = void_bounds[1][2]
    
    # Check if void should be extended further
    z_translate = 0
    if z_min_void > z_min_orig:
        z_translate = z_min_orig - z_min_void - 0.5  # Extra 0.5 unit for safety (didnt have any effect overall)
    elif z_max_void < z_max_orig:
        z_translate = z_max_orig - z_max_void + 0.5  # Extra 0.5 unit for safety
        
    if z_translate != 0:
        translate_matrix = np.eye(4)
        translate_matrix[2, 3] = z_translate
        void_mesh_scaled.apply_transform(translate_matrix)
    
    return void_mesh_scaled

def is_boundary_void(void_mesh, original_mesh, boundary_threshold=0.01):
    """
    Check if a void is too close to the boundary of the original mesh.
    
    Parameters:
    -----------
    void_mesh : trimesh.Trimesh
        The void mesh to check
    original_mesh : pyvista.PolyData
        The original mesh
    boundary_threshold : float
        The threshold distance (as a fraction of the mesh diameter) to consider "too close"
        
    Returns:
    --------
    bool
        True if the void is near the boundary and should be modified/skipped
    """
    # Convert to numpy arrays for bounds checking
    void_points = np.array(void_mesh.vertices)
    orig_points = np.array(original_mesh.points)
    
    # Get bounds
    void_min = void_points.min(axis=0)
    void_max = void_points.max(axis=0)
    orig_min = orig_points.min(axis=0)
    orig_max = orig_points.max(axis=0)
    
    # Calculate mesh diagonal (for relative distance)
    diag = np.linalg.norm([orig_max[0] - orig_min[0], 
                           orig_max[1] - orig_min[1], 
                           orig_max[2] - orig_min[2]])
    
    threshold_dist = diag * boundary_threshold
    
    # Check if void is too close to any boundary in X or Y (not Z)
    for dim in range(2):  # Only check X and Y dimensions
        if (void_min[dim] - orig_min[dim] < threshold_dist or
            orig_max[dim] - void_max[dim] < threshold_dist):
            return True
    
    return False

def adjust_void_for_boundary(void_mesh, original_mesh, boundary_threshold=0.01, shrink_factor=0.9):
    """
    Adjust a void mesh that's close to the boundary by shrinking it in X and Y directions.
    
    Parameters:
    -----------
    void_mesh : trimesh.Trimesh
        The void mesh to adjust
    original_mesh : pyvista.PolyData
        The original mesh (for reference bounds)
    boundary_threshold : float
        The threshold distance to consider "too close"
    shrink_factor : float
        How much to shrink the void in X and Y (values < 1.0 shrink, > 1.0 expand)
        
    Returns:
    --------
    trimesh.Trimesh
        The adjusted void mesh
    """
    print(f"Adjusting void mesh for boundary preservation...")
    
    # Check if void is close to boundary
    if not is_boundary_void(void_mesh, original_mesh, boundary_threshold):
        print("Void is not near boundary, no adjustment needed.")
        return void_mesh
    
    # Get the center of the void mesh
    void_center = np.mean(void_mesh.vertices, axis=0)
    
    # Create a scaling matrix that scales in X and Y directions but not Z
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = shrink_factor  # Scale x-dimension
    scale_matrix[1, 1] = shrink_factor  # Scale y-dimension
    
    # Create a translation matrix to center the mesh at origin before scaling
    to_origin = np.eye(4)
    to_origin[:3, 3] = -void_center
    
    # Create a translation matrix to move the mesh back after scaling
    from_origin = np.eye(4)
    from_origin[:3, 3] = void_center
    
    # Apply the transformations: move to origin, scale, move back
    void_mesh_adjusted = void_mesh.copy()
    void_mesh_adjusted.apply_transform(from_origin @ scale_matrix @ to_origin)
    
    return void_mesh_adjusted

def apply_boundary_mask(original_mesh, final_mesh, boundary_thickness=0.05):
    """
    Apply a boundary mask to preserve the original mesh boundary.
    This creates a new mesh that uses the boundaries from the original mesh
    and the internal geometry from the final mesh.
    
    Parameters:
    -----------
    original_mesh : pyvista.PolyData
        The original mesh with intact boundaries
    final_mesh : pyvista.PolyData
        The final mesh with internal voids
    boundary_thickness : float
        How thick the boundary should be (as a fraction of mesh size)
        
    Returns:
    --------
    pyvista.PolyData
        A new mesh with preserved boundaries
    """
    print(f"Applying boundary mask to preserve original boundaries...")
    
    try:
        # Get the original mesh bounds
        bounds = np.array(original_mesh.bounds).reshape(3, 2)
        diag = np.linalg.norm([bounds[0, 1] - bounds[0, 0],
                              bounds[1, 1] - bounds[1, 0],
                              bounds[2, 1] - bounds[2, 0]])
        
        # Calculate the boundary thickness in absolute units
        thickness = diag * boundary_thickness
        
        # Create a mask for the boundary region
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        
        # Create the boundary mask for the original mesh
        boundary_mask = ((original_mesh.points[:, 0] <= x_min + thickness) | 
                         (original_mesh.points[:, 0] >= x_max - thickness) |
                         (original_mesh.points[:, 1] <= y_min + thickness) | 
                         (original_mesh.points[:, 1] >= y_max - thickness))
        
        # Get the boundary points and cells from the original mesh
        boundary_cells = []
        for i in range(original_mesh.n_cells):
            cell = original_mesh.get_cell(i)
            
            # If any point in the cell is in the boundary region, keep the cell
            if np.any(boundary_mask[cell.point_ids]):
                boundary_cells.append(i)
        
        # Extract the boundary cells from the original mesh if any exist
        if len(boundary_cells) > 0:
            boundary_mesh = original_mesh.extract_cells(boundary_cells)
        else:
            print("Warning: No boundary cells found, using original mesh")
            return final_mesh
        
        # Create a mask for the interior region (inverse of boundary)
        interior_mask = ~boundary_mask
        
        # Get the interior cells from the final mesh
        interior_cells = []
        for i in range(final_mesh.n_cells):
            cell = final_mesh.get_cell(i)
            
            # If all points in the cell are in the interior region, keep the cell
            interior_flag = True
            for pt_id in cell.point_ids:
                # Skip if point index is out of bounds
                if pt_id >= len(boundary_mask):
                    interior_flag = False
                    break
                    
                # Check if point is in boundary
                if boundary_mask[pt_id]:
                    interior_flag = False
                    break
                    
            if interior_flag:
                interior_cells.append(i)
        
        # Extract the interior cells from the final mesh if any exist
        if len(interior_cells) > 0:
            interior_mesh = final_mesh.extract_cells(interior_cells)
        else:
            print("Warning: No interior cells found, using boundary mesh only")
            return boundary_mesh.clean()
        
        # Combine the boundary and interior meshes
        combined_mesh = boundary_mesh.merge(interior_mesh)
        
        # Clean the combined mesh
        combined_mesh = combined_mesh.clean()
        
        # Ensure we return a valid mesh
        if combined_mesh is None or combined_mesh.n_cells == 0:
            print("Warning: Combined mesh is invalid, returning final mesh")
            return final_mesh
        
        return combined_mesh
        
    except Exception as e:
        print(f"Error in boundary preservation: {str(e)}")
        print("Returning final mesh without boundary preservation")
        return final_mesh

def create_internal_voids(input_stl=None, input_vtu=None, output_stl=None, 
                         visualize=True, preserve_boundary=True, boundary_thickness=0.05,
                         boundary_threshold=0.01, shrink_factor=0.9):
    """
    Create a model with multiple internal voids with boundary preservation and post-processing.
    
    Parameters:
    -----------
    input_stl : str
        Path to input STL file
    input_vtu : str
        Path to input VTU file with density data
    output_stl : str
        Path for output STL file
    visualize : bool
        Whether to show visualization (default is True)
    preserve_boundary : bool
        Whether to preserve the original mesh boundary (default is True)
    boundary_thickness : float
        How thick the boundary should be (as a fraction of mesh size)
    boundary_threshold : float
        Distance threshold to consider a void "too close" to the boundary
    shrink_factor : float
        How much to shrink voids near boundaries (values < 1.0 shrink)
    """
    print("=== Creating Model with Multiple Internal Voids + Post-Processing ===")
    
    # Check if files exist
    if not os.path.exists(input_stl):
        print(f"Error: Input STL file not found at {input_stl}")
        return None
    
    if not os.path.exists(input_vtu):
        print(f"Error: Input VTU file not found at {input_vtu}")
        return None
    
    # Make sure output directory exists
    output_dir = os.path.dirname(output_stl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load the files
        print(f"Loading STL from: {input_stl}")
        original = pv.read(input_stl).triangulate().compute_normals()
        print(f"Original mesh: {original.n_points} points, {original.n_cells} cells")
        
        # Save a clean copy of the original mesh for boundary preservation
        original_clean = original.clean(tolerance=1e-6, absolute=False)
        
        # Visualize the original mesh before any processing
        if visualize:
            print("Visualizing original mesh...")
            p = pv.Plotter()
            p.add_mesh(original, color='lightblue', show_edges=True)
            p.add_title("Original Mesh Before Processing")
            p.camera_position = 'iso'
            p.show()
            
            # Also show a cross-section view
            p = pv.Plotter()
            half_original = original.clip(normal=(1, 1, 1), origin=(0, 0, 0))
            p.add_mesh(half_original, color='lightblue', show_edges=True)
            p.add_title("Cross-Section of Original Mesh")
            p.camera_position = 'iso'
            p.show()
        
        print(f"Loading VTU from: {input_vtu}")
        voxels = pv.read(input_vtu)
        print(f"Voxel mesh: {voxels.n_points} points, {voxels.n_cells} cells")
        
        # Step 2: Extract low density regions with lower threshold
        print("Extracting low density regions...")
        negative_voxels = extract_low_density_patches(voxels, 'density', threshold=0.4)
        voxel_cut = negative_voxels.extract_surface().triangulate().clean()
        
        # Step 3: Process with marching cubes
        print("Applying marching cubes algorithm with enhanced parameters...")
        marching_cubes = marching_cubes_isosurf(voxel_cut, reference_mesh=original, pad=2, visualize=visualize)
        
        # Step 4: Smooth the surface gently
        print("Applying gentle smoothing to the surface...")
        smoothed = enhanced_smooth(marching_cubes, iterations=50, pass_band=0.2)
        
        # Step 5: Split into separate void patches with lower threshold
        patch_list = split_and_filter_patches(smoothed, cells_threshold_percentage=5)
        
        if len(patch_list) == 0:
            print("No void patches found. Using the entire smoothed mesh.")
            patch_list = [smoothed]
        
        # Visualize all void patches together
        if visualize:
            p = pv.Plotter()
            p.add_mesh(original, color='lightblue', opacity=0.3, show_edges=True)
            for i, patch in enumerate(patch_list):
                # Use different colors for each patch
                colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
                color = colors[i % len(colors)]
                p.add_mesh(patch, color=color, opacity=0.7)
            p.add_title("Original Mesh with All Void Patches")
            p.show()
        
        # Step 6: Convert original to trimesh
        tm_base = to_trimesh(original)
        trimesh.repair.fill_holes(tm_base)
        trimesh.repair.fix_normals(tm_base)
        trimesh.repair.fix_inversion(tm_base)
        
        # Visualize the starting point for boolean operations
        if visualize:
            pv_start = pv.PolyData(tm_base.vertices, faces=np.hstack(
                [np.full((len(tm_base.faces), 1), 3), tm_base.faces]))
            p = pv.Plotter()
            p.add_mesh(pv_start, color='lightblue', opacity=0.7, show_edges=True)
            p.add_title("Starting Mesh for Boolean Operations")
            p.show()
        
        # Step 7: Process each void patch with post-processing and apply boolean operations
        result_tm = tm_base  # Start with the original mesh
        
        for i, patch in enumerate(patch_list):
            print(f"\nProcessing void patch {i+1} of {len(patch_list)} with post-processing...")
            
            # Process the patch with post-processing
            tm_void = process_void_patch(patch, original, 
                                       boundary_threshold=boundary_threshold,
                                       shrink_factor=shrink_factor)
            
            # Check if patch processing failed
            if tm_void is None:
                print(f"Skipping void patch {i+1} due to processing failure.")
                continue
            
            # Scale the void in z-direction to ensure it penetrates the original mesh
            tm_void = scale_void_for_subtraction(tm_void, original, z_scale_factor=1.2)
            
            # Visualize this void
            if visualize:
                pv_void = pv.PolyData(tm_void.vertices, faces=np.hstack(
                    [np.full((len(tm_void.faces), 1), 3), tm_void.faces]))
                
                # Validate void position
                if not validate_void_patch(tm_void, original):
                    print(f"Skipping void patch {i+1} as it's outside the original mesh.")
                    continue
                    
                p = pv.Plotter()
                p.add_mesh(original, color='lightblue', opacity=0.3, show_edges=True)
                p.add_mesh(pv_void, color='red', opacity=0.7, show_edges=True)
                p.add_title(f"Void Patch {i+1} (Post-Processed & Scaled)")
                p.show()
            
            # Create a backup of the current result
            result_tm_backup = trimesh.Trimesh(vertices=result_tm.vertices.copy(), 
                                              faces=result_tm.faces.copy())
            
            # Perform boolean operation
            print(f"Applying boolean operation for void {i+1}...")
            try:
                # Use built-in engine for boolean operations
                temp_result = trimesh.boolean.difference([result_tm, tm_void])
                
                # Validate result
                if len(temp_result.faces) > len(result_tm.faces) * 0.5:
                    result_tm = temp_result
                    print(f"Boolean for void {i+1} successful with built-in engine!")
                else:
                    print(f"Built-in engine produced invalid result. Skipping this void.")
                    # Restore from backup
                    result_tm = result_tm_backup
                    continue
                    
            except Exception as e:
                print(f"Boolean operation failed for void {i+1}: {str(e)}")
                print("Skipping this void and continuing with the next one...")
                # Restore from backup
                result_tm = result_tm_backup
                continue
            
            # Display intermediate result info
            print(f"After void {i+1}: {len(result_tm.faces)} faces, {len(result_tm.vertices)} vertices")
            
            # Visualize the result after this boolean operation
            if visualize:
                current_result = pv.PolyData(result_tm.vertices, faces=np.hstack(
                    [np.full((len(result_tm.faces), 1), 3), result_tm.faces]))
                
                p = pv.Plotter()
                p.add_mesh(current_result, color='lightblue', opacity=0.8, show_edges=True)
                p.add_title(f"Result After Void {i+1}")
                p.show()
                
                # Show a cross-section view to better see the void
                p = pv.Plotter()
                half_result = current_result.clip(normal=(1, 1, 1), origin=(0, 0, 0))
                p.add_mesh(half_result, color='lightblue', show_edges=True)
                p.add_title(f"Cross-Section After Void {i+1}")
                p.show()
            
            # Save intermediate result after each successful boolean operation
            output_dir = os.path.dirname(output_stl)
            intermediate_path = os.path.join(output_dir, f"intermediate_result_{i+1}.stl")
            result_tm.export(intermediate_path)
            print(f"Saved intermediate result after void {i+1} to: {intermediate_path}")
        
        # If the result is empty, use the original mesh
        if len(result_tm.faces) == 0 or len(result_tm.faces) < len(tm_base.faces) * 0.1:
            print("WARNING: Final result is empty or too small. Using original mesh.")
            result_tm = tm_base
        
        # Step 8: Export the result
        print(f"\nSaving result with all voids to: {output_stl}")
        result_tm.export(output_stl)
        
        # Convert to PyVista for visualization
        pv_result = pv.PolyData(result_tm.vertices, faces=np.hstack(
            [np.full((len(result_tm.faces), 1), 3), result_tm.faces]))
        
        # Apply minimal cleanup
        print("Applying minimal cleanup...")
        pv_result = pv_result.clean(tolerance=1e-6, absolute=False)
        
        # Step 9: Apply boundary preservation if requested
        if preserve_boundary:
            print("Applying boundary preservation...")
            preserved_result = apply_boundary_mask(original_clean, pv_result, boundary_thickness)
            
            # Convert to trimesh for export
            preserved_tm = to_trimesh(preserved_result)
            
            # Save the boundary-preserved result
            output_base = os.path.splitext(output_stl)[0]
            preserved_path = f"{output_base}_boundary_preserved.stl"
            preserved_tm.export(preserved_path)
            print(f"Saved boundary-preserved result to: {preserved_path}")
            
            # Use the preserved result for further visualization
            pv_result = preserved_result
        
        # Re-export the final result
        result_tm = to_trimesh(pv_result)
        print(f"Re-saving final result to: {output_stl}")
        result_tm.export(output_stl)
        
        # Visualize the final result
        if visualize:
            p = pv.Plotter()
            p.add_mesh(pv_result, color='lightblue', opacity=0.7, show_edges=True)
            p.add_title("Final Result with All Internal Voids")
            p.camera_position = 'iso'
            p.show()
            
            # Half view
            p = pv.Plotter()
            half = pv_result.clip(normal=(1, 1, 1), origin=(0, 0, 0))
            p.add_mesh(half, color='lightblue', show_edges=True)
            p.add_title("Half View - Internal Voids")
            p.camera_position = 'iso'
            p.show()
        
        return pv_result
        
    except Exception as e:
        print(f"Error creating internal voids: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== Working Solution with Post-Processing for Multiple Voids ===")

    example = ExamplesCAD.Mitchell_1  # Change to the one you want
    input_stl, input_vtu, output_stl = get_example_cad(example)
    
    # Create the model with multiple internal voids and post-processing
    result = create_internal_voids(
        input_stl=input_stl,
        input_vtu=input_vtu,
        output_stl=output_stl,
        shrink_factor=1.08
    )
    
    if result is not None:
        print("\n Successfully created model with multiple internal voids + post-processing!")
        print("Post-processing applied to each void patch")
        print("Before/after visualizations")
        print("Boundary preservation maintained")
        print("Regular smoothing")
    else:
        print("\n Failed to create model with internal voids.")