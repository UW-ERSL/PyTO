import pyvista as pv
import numpy as np

# Function to create the domain with or without a cutout
def make_domain_with_optional_cutout(width, height, make_hole=False, thickness=0.1):
    outer_points = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0]
    ])
    outer_faces = [4, 0, 1, 2, 3]
    outer = pv.PolyData(outer_points, outer_faces).extrude((0, 0, thickness), capping=True)
    
    if not make_hole:
        return outer

    hole_points = np.array([
        [width / 4, height / 4, 0],
        [width * 3 / 4, height / 4, 0],
        [width * 3 / 4, height * 3 / 4, 0],
        [width / 4, height * 3 / 4, 0]
    ])
    hole_faces = [4, 0, 1, 2, 3]
    hole = pv.PolyData(hole_points, hole_faces).extrude((0, 0, thickness), capping=True)

    domain = outer.boolean_difference(hole)
    return domain


def is_point_inside_domain(domain, point):
    #pt3d = pv.PolyData(np.array([[point[0], point[1], point[2]]], dtype=np.float32))
    enclosed = point.select_enclosed_points(domain, tolerance=1e-5)
    return bool(enclosed["SelectedPoints"][0])


def is_segment_inside_domain(domain, line_seg: pv.Line, eps=1e-6):
    # Extract endpoints from the pv.Line object
    if line_seg.n_points != 2:
        raise ValueError("Expected a line segment with exactly two points.")

    pt1 = line_seg.points[0].astype(np.float32)
    pt2 = line_seg.points[1].astype(np.float32)

    # Compute the direction vector
    direction = pt2 - pt1
    length = np.linalg.norm(direction)

    if length == 0:
        raise ValueError("Segment length is zero; pt1 and pt2 are identical.")

    # Normalize and offset both endpoints slightly inward
    unit_dir = direction / length
    pt1_adj = pt1 + eps * unit_dir
    pt2_adj = pt2 - eps * unit_dir

    # Create the line and perform inside test
    line = pv.Line(pt1_adj, pt2_adj)
    enclosed = line.select_enclosed_points(domain, tolerance=1e-5)
    return bool(enclosed["SelectedPoints"].all())

if __name__ =='__main__': 
    # Create domain
    domain = make_domain_with_optional_cutout(20, 10, make_hole=False)
    print('open edges', domain.n_open_edges)

    # Test point and segment
    point = (10, 5, 1e-3)
    segment_start = (0, 0, 1e-3)
    segment_end = (10, 10, 1e-3)
    pv_point = pv.PolyData(np.array([[point[0], point[1], point[2]]], dtype=np.float32))
    print("Point inside:", is_point_inside_domain(domain, pv_point))
    line_seg = pv.Line(np.array(segment_start), np.array(segment_end))
    print("Segment inside:", is_segment_inside_domain(domain, line_seg))

    # Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(domain, show_edges=True, opacity=0.5, color='lightblue')

    # Add test point
    plotter.add_mesh(pv.PolyData([point]), color='red', point_size=15, render_points_as_spheres=True)

    # Add test segment
    segment_line = pv.Line(np.array(segment_start), np.array(segment_end))
    plotter.add_mesh(segment_line, color='green', line_width=4)

    # Show
    plotter.show()

