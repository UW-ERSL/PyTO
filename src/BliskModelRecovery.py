import pyvista as pv
import numpy as np
import trimesh
from topopt_structural_benchmarks import *
from scipy.spatial import cKDTree

'''
Comments/Questions:
1. Should we simplify the final stl?
'''

# def idw_interpolate_grid(grid, vtu, field="inverted_density", radius=0.1, p=2, null_value=-1):
#     # Convert mesh to points and data array
#     mesh_points = vtu.points
#     mesh_values = vtu.point_data[field]
#     grid_points = grid.points
#     interpolated = np.full(grid.n_points, null_value, dtype=float)
    
#     # Build KDTree for fast neighbor search
#     tree = cKDTree(mesh_points)

#     # Query all grid points at once
#     # neighbors_indices: list of arrays, one per grid point, with indices of nearby mesh points
#     neighbors_indices = tree.query_ball_point(grid_points, r=radius)

#     for i, inds in enumerate(neighbors_indices):
#         if len(inds) == 0:
#             continue  # No nearby mesh points; leave as null_value
#         close_points = mesh_points[inds]
#         close_values = mesh_values[inds]
#         # Compute distances to grid point i
#         dists = np.linalg.norm(close_points - grid_points[i], axis=1)
#         dists[dists == 0] = 1e-12  # avoid division by zero
#         weights = 1.0 / (dists ** p)
#         interpolated[i] = np.sum(weights * close_values) / np.sum(weights)
#     grid.point_data[field] = interpolated
#     return grid

def get_principal_axis(mesh):
    points = mesh.points - mesh.center
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    return principal_axis

def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

def clean_and_repair_stl(stl_mesh):
    tm = trimesh.Trimesh(vertices=stl_mesh.points, faces=stl_mesh.faces.reshape(-1, 4)[:, 1:4])
    # Update deprecated methods
    tm.update_faces(tm.unique_faces())
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(tm)
    trimesh.repair.fix_normals(tm)
    tm = tm.split(only_watertight=False)[0]  # Keep largest component

    if not tm.is_watertight:
        print("Warning: Mesh is not watertight after repair. TetGen may fail.")

    repaired = pv.PolyData(tm.vertices, faces=np.hstack([np.full((len(tm.faces), 1), 3), tm.faces]))
    repaired = repaired.clean(tolerance=1e-8)
    return repaired

def plot_density_above_threshold(vtu_input, field="density", threshold=0.5):
    # Accept either a file path or a mesh object
    if isinstance(vtu_input, str):
        vtu = pv.read(vtu_input)
    else:
        vtu = vtu_input
    if field not in vtu.point_data and field not in vtu.cell_data:
        raise ValueError(f"{field} not found in mesh.")
    # Convert to cell data if needed
    if field in vtu.point_data:
        vtu = vtu.point_data_to_cell_data()
    # Threshold the mesh
    mesh_above = vtu.threshold(value=threshold, scalars=field)
    # Plot
    mesh_above.plot(scalars=field, cmap="viridis", show_edges=True)

# Isosurface extraction 
def extract_isosurface(vtu,  isovalue=0.5, resolution=5,null_value=-1):
    field = "density"
    if field in vtu.point_data:
        #data = vtu.point_data[field] # Extract point data if available it checks if the field is in point data
        vtu = vtu.point_data_to_cell_data() # and if yes it converts point data to cell data
        data = vtu.cell_data[field]
        
    elif field in vtu.cell_data:
        data = vtu.cell_data[field] #Grab the field directly if not point and only cell data is available
    else:
        raise ValueError(f"{field} not found in mesh.")
    
    # Add this line to print min/max density
    print("Density min:", np.min(data), "Density max:", np.max(data))
     
    normalizdedDensity = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
    vtu.cell_data["inverted_density"] = 1.0 - normalizdedDensity
    #first get the bounding box of the vtu mesh for eg. [x max, x min, y max, y min, z max, z min]
    bounds = vtu.bounds

   
    #Calculate padding as 10% of the largest mesh dimension 
    padding = max(bounds[1]-bounds[0], 
                  bounds[3]-bounds[2], 
                  bounds[5]-bounds[4]) * 0.1
   
    nVoxels = resolution * vtu.n_cells # n_cells gives the number of cells in the mesh,
    alpha = (nVoxels / np.prod(np.array(bounds[1::2]) - np.array(bounds[0::2])))**(1/3)
    
    # Compute number of points (grid dimensions) in x, y, z directions
    dimensions = int(alpha * (bounds[1] - bounds[0]) ), int(alpha * (bounds[3] - bounds[2]) ),  int(alpha * (bounds[5] - bounds[4]) )
   

    spacing = (
            (bounds[1] - bounds[0] + 2*padding) / (dimensions[0] - 1),
            (bounds[3] - bounds[2] + 2*padding) / (dimensions[1] - 1),
            (bounds[5] - bounds[4] + 2*padding) / (dimensions[2] - 1)
        )

    origin = (bounds[0] - padding, bounds[2] - padding, bounds[4] - padding)

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)

    vtu = vtu.cell_data_to_point_data()
    radius = padding / 4  # larger value for smoother interpolation but we can loose features
    

    #p = 8 # Power parameter for IDW, higher values give more weight to closer points, weight = 1 / (distance ** p)
    # Interpolate the inverted density field onto the grid
    #grid = idw_interpolate_grid(grid, vtu, field="inverted_density", radius=radius, p=p, null_value=null_value)
    grid = grid.interpolate(vtu, radius= radius, null_value=null_value)
   
    return grid.contour([isovalue], scalars="inverted_density")

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
   
    model_path = "./Models/BliskModel/BliskFullWithBlades.stl"
    vtu_path = "./Models/BliskModel/BliskFullNoBlades.vtu"
    output_path = "./Models/BliskModel/BliskFullOptimizedBlades.stl"


    design_domain_stl = pv.read(model_path).clean().triangulate().compute_normals()  # Read and triangulate the STL file
    vtu = pv.read(vtu_path)

    void_region_stl = extract_isosurface(vtu)
    visualize(design_domain_stl, void_region_stl,"Void Isosurface")

    optimized_topology_stl = subtract_voids_from_stl(design_domain_stl, void_region_stl)

    visualize(design_domain_stl, optimized_topology_stl,"Optimized STL")
    optimized_topology_stl.save(output_path)

