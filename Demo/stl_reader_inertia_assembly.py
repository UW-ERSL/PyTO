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

   
if __name__ == "__main__":
    import os


    script_dir = os.path.dirname(os.path.abspath(__file__))
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/SolidNoseConeForZAngularSymmetry.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/PayloadBase_meters.STL')
    stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/KnuckleAssembly/KnuckleAssembly.STL')
    stl_file1 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTest/AssemblyInertiaTestPart1.STL')
    stl_file2 = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/AssemblyInertiaTest/AssemblyInertiaTestPart2_BasePlate.STL')

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
