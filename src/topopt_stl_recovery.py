import pyvista as pv
import numpy as np
import enum
import trimesh
from scipy import ndimage as ndi

#for now 
# Step 1 Enum Class 
#Step 2 Once loaded the vtu file extract the negative density eleemnts
#Step 3 Isosurface extraction
#Step4 Visualization
#Step5 Save the extracted isosurface as STL


class ExamplesCAD(enum.Enum):
    EdgeCantilever = enum.auto()
    #BliskSectionWithBlade = enum.auto()
    Mitchell_1 = enum.auto()
    Mitchell_2 = enum.auto()
    Mitchell_3 = enum.auto()
    ShortCantileverTipLoad = enum.auto()
    ShortCantileverMidLoad = enum.auto()
    CantileverTipLoad = enum.auto()
    CantileverMidLoad = enum.auto()
    MBBB = enum.auto()
    LBracketTopLoad = enum.auto()
    LBracketMidLoad = enum.auto()
    TwoBar = enum.auto()
    DistributedLoad = enum.auto()
    #VerticalBar = enum.auto()
    # FilletedBeam = enum.auto()
    # ThreeHoleBracket = enum.auto()
    # CircularPlateHole = enum.auto()
    # cube = enum.auto()
    # BasePlateOptimizableVol = enum.auto()
    TorquePlate = enum.auto()

def get_example_cad(example: ExamplesCAD):
    fp_stl_folder = "Models/"
    fp_output_folder = "output/"

    if example == ExamplesCAD.EdgeCantilever:
        return (fp_stl_folder + "EdgeCantilever/EdgeCantilever.STL",
                fp_stl_folder + "EdgeCantilever/EdgeCantilever.vtu",
                fp_output_folder + "EdgeCantilever_internal_voids.stl")
    # elif example == ExamplesCAD.BliskSectionWithBlade:
    #     return (fp_stl_folder + "Saketh/BliskSectionWithBlade2test.STL",
    #             fp_stl_folder + "Saketh/BliskSectionWithBlade2test.vtu",
    #             fp_output_folder + "BliskSectionWithBlade2_internal_voids.stl")
    # elif example == ExamplesCAD.KnuckleAssembly:
    #     return (fp_stl_folder + "KnuckleAssembly/KnuckleAssembly.STL",
    #             fp_stl_folder + "KnuckleAssembly/KnuckleAssembly.vtu",
    #             fp_output_folder + "KnuckleAssembly_internal_voids.stl")
    elif example == ExamplesCAD.Mitchell_1:
        return (fp_stl_folder + "Mitchell/Mitchell.STL",
                fp_stl_folder + "Mitchell/Mitchell_1.vtu",
                fp_output_folder + "Mitchell_1_internal_voids.stl")
    elif example == ExamplesCAD.Mitchell_2:
        return (fp_stl_folder + "Mitchell/Mitchell.STL",
                fp_stl_folder + "Mitchell/Mitchell_2.vtu",
                fp_output_folder + "Mitchell_2_internal_voids.stl")
    elif example == ExamplesCAD.Mitchell_3:
        return (fp_stl_folder + "Mitchell/Mitchell.STL",
                fp_stl_folder + "Mitchell/Mitchell_3.vtu",
                fp_output_folder + "Mitchell_3_internal_voids.stl")
    elif example == ExamplesCAD.ShortCantileverTipLoad:
        return (fp_stl_folder + "ShortCantilever/ShortCantilever.STL",
                fp_stl_folder + "ShortCantilever/ShortCantileverTipLoad.vtu",
                fp_output_folder + "ShortCantileverMidLoad_internal_voids.stl")
    elif example == ExamplesCAD.ShortCantileverMidLoad:
        return (fp_stl_folder + "ShortCantilever/ShortCantilever.STL",
                fp_stl_folder + "ShortCantilever/ShortCantileverMidLoad.vtu",
                fp_output_folder + "ShortCantileverMidLoad_internal_voids.stl")
    elif example == ExamplesCAD.CantileverTipLoad:
        return (fp_stl_folder + "Cantilever/Cantilever.STL",
                fp_stl_folder + "Cantilever/CantileverTipLoad.vtu",
                fp_output_folder + "CantileverMidLoad_internal_voids.stl")
    elif example == ExamplesCAD.CantileverMidLoad:
        return (fp_stl_folder + "Cantilever/Cantilever.STL",
                fp_stl_folder + "Cantilever/CantileverMidLoad.vtu",
                fp_output_folder + "CantileverMidLoad_internal_voids.stl")
    elif example == ExamplesCAD.TwoBar:
        return (fp_stl_folder + "TwoBar/TwoBar.STL",
                fp_stl_folder + "TwoBar/TwoBar.vtu",
                fp_output_folder + "TwoBar_internal_voids.stl")
    elif example == ExamplesCAD.MBBB:
        return (fp_stl_folder + "MBBB/MBBB.STL",
                fp_stl_folder + "MBBB/MBBB.vtu",
                fp_output_folder + "MBBB_internal_voids.stl")
    elif example == ExamplesCAD.DistributedLoad:
        return (fp_stl_folder + "DistributedLoad/DistributedLoad.STL",
                fp_stl_folder + "DistributedLoad/DistributedLoad.vtu",
                fp_output_folder + "DistributedLoad_internal_voids.stl")
    elif example == ExamplesCAD.LBracketTopLoad:
        return (fp_stl_folder + "LBracket/LBracket.STL",
                fp_stl_folder + "LBracket/LBracketTopLoad.vtu",
                fp_output_folder + "LBracketTopLoad_internal_voids.stl")
    elif example == ExamplesCAD.LBracketMidLoad:
        return (fp_stl_folder + "LBracket/LBracket.STL",
                fp_stl_folder + "LBracket/LBracketMidLoad.vtu",
                fp_output_folder + "LBracketMidLoad_internal_voids.stl")
    # elif example == ExamplesCAD.VerticalBar:
    #     return (fp_stl_folder + "VerticalBar/VerticalBar.STL",
    #             fp_stl_folder + "VerticalBar/VerticalBar.vtu",
    #             fp_output_folder + "VerticalBar_internal_voids.stl")
    # elif example == ExamplesCAD.FilletedBeam:
    #     return (fp_stl_folder + "FilletedBeam/FilletedBeam.STL",
    #             fp_stl_folder + "FilletedBeam/FilletedBeam.vtu",
    #             fp_output_folder + "FilletedBeam_internal_voids.stl")
    # elif example == ExamplesCAD.ThreeHoleBracket:
    #     return (fp_stl_folder + "ThreeHoleBracket/ThreeHoleBracket.STL",
    #             fp_stl_folder + "ThreeHoleBracket/ThreeHoleBracket.vtu",
    #             fp_output_folder + "ThreeHoleBracket_internal_voids.stl")
    # elif example == ExamplesCAD.CircularPlateHole:
    #     return (fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL",
    #             fp_stl_folder + "CircularPlateHole/CircularPlateHole.vtu",
    #             fp_output_folder + "CircularPlateHole_internal_voids.stl")
    elif example == ExamplesCAD.TorquePlate:
        return (fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL",
                fp_stl_folder + "CircularPlateHole/TorquePlate.vtu",
                fp_output_folder + "TorquePlate_internal_voids.stl")
    else:
        raise ValueError(f"Unknown example: {example}")


def negative_density(vtu, field="density"):
    if field in vtu.point_data:
        #data = vtu.point_data[field] # Extract point data if available it checks if the field is in point data
        vtu = vtu.point_data_to_cell_data() # and if yes it converts point data to cell data
        data = vtu.cell_data[field]
        '''
        1D example
        Points:    ●────●────●────●────●
        Index:     0    1    2    3    4

        Cells:     |====|====|====|====|
        Index:        0    1    2    3

        '''
        
    elif field in vtu.cell_data:
        data = vtu.cell_data[field] #Grab the field directly if not point and only cell data is available
    else:
        raise ValueError(f"{field} not found in mesh.")
    
    #so here I am normalizing the data to the range [0, 1] and then inverting it
    #Normalization formula: norm = (data - min) / (max - min)  
    # then use threshold like 0.5 to extract the isosurface
    # Inversion formula: inverted = 1.0 - norm  
    norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
    vtu.cell_data["inverted_density"] = 1.0 - norm
    return vtu

 

#Step 3 Isosurface extraction (change each parameter and visualize the result)
def extract_isosurface(vtu, field="inverted_density", isovalue=0.5, resolution=1):

    #first get the bounding box of the vtu mesh for eg. [x max, x min, y max, y min, z max, z min]
    bounds = vtu.bounds
    print(f"Bounds: {bounds}")

    #Calculate padding as 10% of the largest mesh dimension (I changed it to 1 percent and it shows the void patches in separate tiny blocks, 
    #but if I chnage it to 10 percent it blends those and shows a whole voulme)
    #It ensures the grid fully surrounds the mesh and avoids cutting off important features
    padding = max(bounds[1]-bounds[0], 
                  bounds[3]-bounds[2], 
                  bounds[5]-bounds[4]) * 0.1
    print(f"Padding: {padding}, Bounds: {bounds}")
    
    nVoxels = resolution * vtu.n_cells # n_cells gives the number of cells in the mesh, so it is like how many blocks I have in my mesh, this just refines my mesh
    alpha = (nVoxels / np.prod(np.array(bounds[1::2]) - np.array(bounds[0::2])))**(1/3)
    print(f"Alpha: {alpha}, nVoxels: {nVoxels}, Bounds: {bounds}")
    # Compute number of points (grid dimensions) in x, y, z directions
    dimensions = int(alpha * (bounds[1] - bounds[0]) ), int(alpha * (bounds[3] - bounds[2]) ),  int(alpha * (bounds[5] - bounds[4]) )
    print(f"Dimensions: {dimensions}, Bounds: {bounds}, Padding: {padding}")


    #Calculate spacing between grid points along each axis
    #ensuring that the padded bounding box fits evenly within the grid dimensions
    # spacing = L / (n - 1) more precisely like if I have x = 10 points then I add padding of 1 cm to each side, so the total length is 10 + 2 = 12 cm
    # and the spacing between each point is (12 cm) / (10 - 1) = 1.33 cm
    #spacing = (10 + 2 × padding) / (n_points - 1) = 12 / 9 ≈ 1.33 cm
    spacing = (
            (bounds[1] - bounds[0] + 2*padding) / (dimensions[0] - 1),
            (bounds[3] - bounds[2] + 2*padding) / (dimensions[1] - 1),
            (bounds[5] - bounds[4] + 2*padding) / (dimensions[2] - 1)
        ) # [0, 2, 4] = x_min, y_min, z_min (spacing between each block) (start, stop, step)

    #we just have to dfine the origin for our gridd so we have to shift back the padding by which we start from the starting point and we do it for each axis
    origin = (bounds[0] - padding, bounds[2] - padding, bounds[4] - padding)

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)
    vtu = vtu.cell_data_to_point_data()
    import time
    start = time.time()
    grid = grid.interpolate(vtu, radius=padding/3, null_value=-1)
    print(f"Interpolation took {time.time() - start:.2f} seconds")
    return grid.contour([isovalue], scalars=field)


def visualize(original_stl, void_surface):
    p = pv.Plotter()
    p.add_mesh(original_stl, color='lightblue', opacity=0.3, label="Original STL")
    p.add_mesh(void_surface, color='red', opacity=0.9, label="Void Isosurface")
    p.add_legend()
    p.camera_position = 'iso'
    p.show()

def to_trimesh(mesh):
    """Convert a PyVista mesh to Trimesh."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=mesh.points, faces=faces)

def subtract_voids_from_stl(stl, void_surface, output_path=None, visualize=True):
    # Load STL as PyVista, convert both to Trimesh
    #stl_pv = pv.read(stl_path).triangulate()
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

    # Save
    if output_path:
        result_pv.save(output_path)
        print(f"Result saved to {output_path}")

    return result_pv


if __name__ == "__main__":
    example = ExamplesCAD.Mitchell_1  # Change this to any example you want to visualize 
    stl_path, vtu_path, output_path = get_example_cad(example)

    stl = pv.read(stl_path).triangulate().compute_normals()  # Read and triangulate the STL file
    vtu = pv.read(vtu_path)

    vtu = negative_density(vtu)
    void_surface = extract_isosurface(vtu, isovalue=0.5, resolution=2.5)
    

    visualize(stl, void_surface)

    subtract_voids_from_stl(stl, void_surface, output_path=output_path, visualize=True)


    # When the part is very thin in the z-direction, changing the resolution parameter causes the extracted voids or isosurfaces to appear stretched or squished/thin along z.
    # I guess because the number of grid slices in z is determined by the total resolution and the physical thickness 
