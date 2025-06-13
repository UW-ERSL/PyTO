from stl_reader_inertia import *

def parallel_axis_shift(I_com, cg_part, cg_assembly, mass):
    d = cg_part - cg_assembly
    return I_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

def shift_inertia_to_new_point(I_com, mass, from_com, to_com):
    d = to_com - from_com
    d_outer = np.outer(d, d)
    d_sq = np.dot(d, d)
    shift = mass * (d_sq * np.eye(3) - d_outer)
    return I_com + shift

def stack_and_align_three_parts_in_z(top: STLGeom, middle: STLGeom, bottom: STLGeom):
    """
    Stacks three STL parts vertically along the Z-axis: top above middle, middle above bottom.
    The parts are also aligned concentrically in the X-Y plane using their COM.
    The bottom part remains fixed.
    """
    # Get bounding boxes and COMs
    _, _, _, _, zmin_top, _ = top.get_bounding_box()
    _, _, _, _, zmin_mid, zmax_mid = middle.get_bounding_box()
    _, _, _, _, _, zmax_bot = bottom.get_bounding_box()

    _, _, com_top, _ = top.compute_mass_properties()
    _, _, com_mid, _ = middle.compute_mass_properties()
    _, _, com_bot, _ = bottom.compute_mass_properties()

    # Step 1: Align middle on top of bottom
    shift_mid_xy = com_bot[:2] - com_mid[:2]  # XY alignment
    shift_mid_z = zmax_bot - zmin_mid         # Z stacking
    shift_mid = np.array([*shift_mid_xy, shift_mid_z])
    middle.mesh.vectors += shift_mid
    print(f"Shifted middle part by {shift_mid} to sit on top of bottom.")

    # Step 2: Align top on top of newly shifted middle
    # Get updated bounding box and COM for middle after shift
    _, _, _, _, zmin_mid_new, zmax_mid_new = middle.get_bounding_box()
    _, _, com_mid_new, _ = middle.compute_mass_properties()

    shift_top_xy = com_mid_new[:2] - com_top[:2]  # XY alignment
    shift_top_z = zmax_mid_new - zmin_top         # Z stacking
    shift_top = np.array([*shift_top_xy, shift_top_z])
    top.mesh.vectors += shift_top
    print(f"Shifted top part by {shift_top} to sit on top of middle.")

def stack_and_align_parts_in_z(part1: STLGeom, part2: STLGeom):
    # Bounding boxes
    _, _, _, _, zmin1, _ = part1.get_bounding_box()
    _, _, _, _, _, zmax2 = part2.get_bounding_box()

    # Centers of mass
    area, volume, com1, inertia = part1.compute_mass_properties()
    area, volume, com2, inertia = part2.compute_mass_properties()

    # Compute full 3D shift vector
    shift_x = com2[0] - com1[0]
    shift_y = com2[1] - com1[1]
    shift_z = zmax2 - zmin1
    shift_vec = np.array([shift_x, shift_y, shift_z])

    # Translate part1
    part1.mesh.vectors += shift_vec
    print(f"Shifted part1 by {shift_vec} to align concentrically above part2.")

def compute_part_mass_and_inertia(stl_geom, density):
    area, volume, com, inertia = stl_geom.compute_mass_properties()
    mass = density * volume
    I_origin = inertia * density
    I_com = shift_inertia_to_com(I_origin, com, mass)

    print(f"Area: {area}")
    print(f"Volume: {volume}")
    print(f"Center of Mass: {com}")
    print("Moments of inertia: ( kilograms * square meters (if the stl is in meters) )/" \
    "Taken at the center of mass and aligned with the output coordinate system.")
    print(f"Inertia for RocketPy: \n {I_com}")

    return area, volume, com, mass, I_com

# Rotate part 180 degrees around X-axis
def rotate_stl_mesh_180_x(stl_geom: STLGeom):
    R = np.array([
        [1, 0,  0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    stl_geom.mesh.vectors[:] = np.dot(stl_geom.mesh.vectors - stl_geom.mesh.get_mass_properties()[1], R.T) + stl_geom.mesh.get_mass_properties()[1]

def assemble_and_compute_inertia(stl_file_list: list[str], density: float = 1000):

    # Load all parts
    parts = [STLGeom(file) for file in stl_file_list]

    # Align parts in Z-direction (bottom to top stacking)
    for i in range(1, len(parts)):
        lower = parts[i - 1]
        upper = parts[i]
        
        # Get bounding boxes and COMs
        _, _, _, _, zmin_lower, _ = lower.get_bounding_box()
        _, _, _, _, _, zmax_upper = upper.get_bounding_box()

        _, _, com_lower, _ = lower.compute_mass_properties()
        _, _, com_upper, _ = upper.compute_mass_properties()

        # Compute shift to stack and align COMs in XY
        shift_x = com_lower[0] - com_upper[0]
        shift_y = com_lower[1] - com_upper[1]
        shift_z = zmin_lower - zmax_upper
        shift_vec = np.array([shift_x, shift_y, shift_z])

        upper.mesh.vectors += shift_vec

    # Compute mass properties
    total_mass = 0.0
    total_com = np.zeros(3)
    inertias_shifted = []

    for part in parts:
        area, volume, com, inertia = part.compute_mass_properties()
        mass = density * volume
        I_origin = inertia * density
        I_com = shift_inertia_to_com(I_origin, com, mass)

        total_mass += mass
        total_com += mass * com
        inertias_shifted.append((I_com, mass, com))

    total_com /= total_mass

    # Shift all inertias to total COM
    I_total = sum(shift_inertia_to_new_point(I_com, mass, com, total_com)
                  for I_com, mass, com in inertias_shifted)

    # Print final results
    print(f"Total Mass: {total_mass}")
    print(f"Assembly Center of Mass: {total_com}")
    print("Inertia Tensor at Assembly COM:")
    print(I_total)

    # Visualize
    plotter = pv.Plotter()
    colors = ['red', 'blue', 'green', 'orange', 'yellow']
    for i, part in enumerate(parts):
        plotter.add_mesh(part.get_pyvista_mesh(), color=colors[i % len(colors)], opacity=0.5)
    plotter.show()

def test_2components(stl_file1: str, stl_file2: str):
    
    # Load two parts
    stl_geom1 = STLGeom(stl_file1)  # e.g., NoseCone
    stl_geom2 = STLGeom(stl_file2)  # e.g., PayloadBase

    # Stack part1 on top of part2
    stack_and_align_parts_in_z(stl_geom1, stl_geom2)
    density = 1000
    area1, vol1, com1, mass1, I_com1 = compute_part_mass_and_inertia(stl_geom1, density)
    area2, vol2, com2, mass2, I_com2 = compute_part_mass_and_inertia(stl_geom2, density)

    mass_total = mass1 + mass2
    cg_total = (mass1 * com1 + mass2 * com2) / (mass1 + mass2)

    I1_shifted = shift_inertia_to_new_point(I_com1, mass1, com1, cg_total)
    I2_shifted = shift_inertia_to_new_point(I_com2, mass2, com2, cg_total)
    I_assembly_at_cg = I1_shifted + I2_shifted

    print(f"mass_total: {mass_total}")
    print(f"cg_total: {cg_total}")
    print("Moments of inertia: ( kilograms * square meters (if the stl is in meters) )/" \
    "Taken at the center of mass and aligned with the output coordinate system.")
    print(f"Inertia for RocketPy: \n {I_assembly_at_cg}")

    mesh1 = stl_geom1.get_pyvista_mesh()
    mesh2 = stl_geom2.get_pyvista_mesh()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, color='red', opacity=0.5)
    plotter.add_mesh(mesh2, color='blue', opacity=0.5)
    plotter.show()

def test_3components(stl_file1: str, stl_file2: str, stl_file3: str):
    
    # Load two parts
    stl_geom1 = STLGeom(stl_file1)  # e.g., PayloadBase
    stl_geom2 = STLGeom(stl_file2)  # e.g., MainBody
    stl_geom3 = STLGeom(stl_file3)  # e.g., tail

    # Stack part1 on top of part2 on top of part3
    stack_and_align_three_parts_in_z(stl_geom1, stl_geom2, stl_geom3)
    density = 1000
    area1, vol1, com1, mass1, I_com1 = compute_part_mass_and_inertia(stl_geom1, density)
    area2, vol2, com2, mass2, I_com2 = compute_part_mass_and_inertia(stl_geom2, density)
    area3, vol3, com3, mass3, I_com3 = compute_part_mass_and_inertia(stl_geom3, density)

    mass_total = mass1 + mass2 + mass3
    cg_total = (mass1 * com1 + mass2 * com2 + mass3 * com3) / (mass1 + mass2 + mass3)

    I1_shifted = shift_inertia_to_new_point(I_com1, mass1, com1, cg_total)
    I2_shifted = shift_inertia_to_new_point(I_com2, mass2, com2, cg_total)
    I3_shifted = shift_inertia_to_new_point(I_com3, mass3, com3, cg_total)
    I_assembly_at_cg = I1_shifted + I2_shifted + I3_shifted

    print(f"mass_total: {mass_total}")
    print(f"cg_total: {cg_total}")
    print("Moments of inertia: ( kilograms * square meters (if the stl is in meters) )/" \
    "Taken at the center of mass and aligned with the output coordinate system.")
    print(f"Inertia for RocketPy: \n {I_assembly_at_cg}")

    mesh1 = stl_geom1.get_pyvista_mesh()
    mesh2 = stl_geom2.get_pyvista_mesh()
    mesh3 = stl_geom3.get_pyvista_mesh()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, color='red', opacity=0.5)
    plotter.add_mesh(mesh2, color='blue', opacity=0.5)
    plotter.add_mesh(mesh3, color='blue', opacity=0.5)
    plotter.show()

if __name__ == "__main__":
    import os


    script_dir = os.path.dirname(os.path.abspath(__file__))
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/SolidNoseConeForZAngularSymmetry.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/PayloadBase_meters.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/KnuckleAssembly/KnuckleAssembly.STL')
    stl_file1 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTest/AssemblyInertiaTestPart1.STL')
    stl_file2 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTest/AssemblyInertiaTestPart2_BasePlate.STL')
    stl_file2 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTestMultiComp/AssemblyInertiaTestPart2_MainBody.STL')
    stl_file3 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTestMultiComp/AssemblyInertiaTestPart3_tail.STL')

    # write the assumptions. Document the files used and the approach taken.
    # This script reads two STL files, stacks them in Z-axis and computes their mass and inertia properties.
    # Next is to stack multiple components and compute the mass and inertia properties of the stacked assembly.
    # Assumtions: 
    #   The STL files are in meters.
    #   The density is in kg/m^3.
    #   Each STL geometry is oriented with the Z-axis.

    # Test two parts
    #test_2components(stl_file1, stl_file2)
    #test_3components(stl_file1, stl_file2, stl_file3)
    stl_file_list = [stl_file1, stl_file2, stl_file3]
    assemble_and_compute_inertia(stl_file_list, density=1000)
