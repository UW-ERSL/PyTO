import pyvista as pv
import numpy as np
import trimesh
from topopt_structural_benchmarks import *

'''
Comments/Questions:
1. Can't we use the topopt_structural_benchmarks directly to get the STL and VTU paths?
2. Understand what each parameter in the extract_isosurface does and explain in simple terms
3. Should we simplify the final stl?
'''
def idw_interpolate_grid(grid, vtu, field="inverted_density", radius=0.1, p=2, null_value=-1):
    # Convert mesh to points and data array
    mesh_points = vtu.points
    mesh_values = vtu.point_data[field]
    # Prepare grid for output
    interpolated = np.full(grid.n_points, null_value, dtype=float)
    
    for i, pt in enumerate(grid.points):
        dists = np.linalg.norm(mesh_points - pt, axis=1) # Calculate distances from grid point to mesh points Euclidean Distance, distance = sqrt((x_mesh - x_pt)^2 + (y_mesh - y_pt)^2 + (z_mesh - z_pt)^2), calculates the norm along each row
        mask = dists <= radius # Mask points within the specified radius, mask is True wherever the distance is less than or equal to radius
        close_dists = dists[mask] # Get distances of close points
        close_values = mesh_values[mask] # Get values of close points
        if close_dists.size == 0: # If no points are close, skip this point
            continue  
        close_dists[close_dists == 0] = 1e-12 # Avoid division by zero
        weights = 1.0 / (close_dists ** p) # IDW weights
        interpolated[i] = np.sum(weights * close_values) / np.sum(weights) # Weighted average of close values
    # Attach it to grid
    grid.point_data[field] = interpolated 
    return grid

# Isosurface extraction 
def extract_isosurface(vtu,  isovalue=0.5, resolution=2,null_value=-1):
    field = "density"
    if field in vtu.point_data:
        #data = vtu.point_data[field] # Extract point data if available it checks if the field is in point data
        vtu = vtu.point_data_to_cell_data() # and if yes it converts point data to cell data
        data = vtu.cell_data[field]
        
    elif field in vtu.cell_data:
        data = vtu.cell_data[field] #Grab the field directly if not point and only cell data is available
    else:
        raise ValueError(f"{field} not found in mesh.")
     
    normalizdedDensity = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
    vtu.cell_data["inverted_density"] = 1.0 - normalizdedDensity
    #first get the bounding box of the vtu mesh for eg. [x max, x min, y max, y min, z max, z min]
    bounds = vtu.bounds
   
    #Calculate padding as 10% of the largest mesh dimension 
    padding = max(bounds[1]-bounds[0], 
                  bounds[3]-bounds[2], 
                  bounds[5]-bounds[4]) * 0.1
    
    nVoxels = resolution * vtu.n_cells # n_cells gives the number of cells in the mesh,
    #vtu.plot(show_edges=True, opacity=0.5, scalars="density")
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
    radius = padding / 3  # larger value for smoother interpolation but we can loose features
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

def runAllExamples():
    benchmarks_2_5D_problems = [StructuralTOExamples.Mitchell_1, StructuralTOExamples.Mitchell_2,
						StructuralTOExamples.Mitchell_3, 
						StructuralTOExamples.ShortCantileverTipLoad, StructuralTOExamples.ShortCantileverMidLoad,
						StructuralTOExamples.CantileverTipLoad, StructuralTOExamples.CantileverMidLoad,
						StructuralTOExamples.MBBB,
						StructuralTOExamples.LBracketTopLoad, StructuralTOExamples.LBracketMidLoad,
						StructuralTOExamples.TwoBar, 
						StructuralTOExamples.DistributedLoad,
						StructuralTOExamples.TorquePlate,
						StructuralTOExamples.ThreeHoleBracket,]

    benchmarks_3D_problems = [StructuralTOExamples.EdgeCantilever, 
						   StructuralTOExamples.ThreeHoleBracketThick, 
						 StructuralTOExamples.Multiload,
						   StructuralTOExamples.LBracketThickTopLoad,
						StructuralTOExamples.LBracketThickMidLoad,
						StructuralTOExamples.Table]
    
    for to_problem in benchmarks_2_5D_problems + benchmarks_3D_problems:
        to_name = to_problem.name
        print("-" * 50)
        print(f"Running {to_problem.name} problem")
        print("-" * 50)
        model_path = getSTLPath_TOProblem(to_problem)  # Get the STL path for the problem
        vtu_path = "Results/VTU/" + to_name + ".vtu"
        output_path = "Results/FinalTopology/" + to_name + "_optimized.stl"
        
        try:
            design_domain_stl = pv.read(model_path).triangulate().compute_normals()  # Read and triangulate the STL file
        except Exception as e:
            print(f"Failed to load STL for {to_name}: {e}")
            continue

        try:
            vtu = pv.read(vtu_path)
        except Exception as e:
            print(f"Failed to load VTU for {to_name}: {e}")
            continue
    
        void_region_stl = extract_isosurface(vtu)
        optimized_topology_stl = subtract_voids_from_stl(design_domain_stl, void_region_stl)
      
        try:
            optimized_topology_stl.save(output_path)
        except Exception as e:
            print(f"Failed to save optimized STL for {to_name}: {e}")
            continue

if __name__ == "__main__":

    #runAllExamples(); exit(0)  # Uncomment to run all examples 

    to_problem =  StructuralTOExamples.CantileverMidLoad # Change this to any example

    to_name = to_problem.name
    model_path = getSTLPath_TOProblem(to_problem)  # Get the STL path for the problem
    vtu_path = "Results/VTU/" + to_name + ".vtu"
    output_path = "Results/FinalTopology/" + to_name + "_optimized.stl"
    
    design_domain_stl = pv.read(model_path).triangulate().compute_normals()  # Read and triangulate the STL file
    vtu = pv.read(vtu_path)
  
    void_region_stl = extract_isosurface(vtu)
    visualize(design_domain_stl, void_region_stl,"Void Isosurface")

    optimized_topology_stl = subtract_voids_from_stl(design_domain_stl, void_region_stl)
    visualize(design_domain_stl, optimized_topology_stl,"Optimized STL")
    optimized_topology_stl.save(output_path)

