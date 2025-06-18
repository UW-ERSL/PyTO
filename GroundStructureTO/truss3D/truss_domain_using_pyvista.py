import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk

# Function to create the domain with or without a cutout
def create_domain_with_optional_cutout(width, height, depth=0.1, make_hole=False):
    """
    Create a 3D box domain with optional cutout.
    
    Parameters:
        width (float): Width of the domain in X.
        height (float): Height of the domain in Y.
        depth (float): Depth of the domain in Z.
        make_hole (bool): Whether to create a cutout in the XY plane.

    Returns:
        pv.PolyData: The extruded 3D domain.
    """
    # Define base rectangle
    outer_points = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0]
    ])
    outer_faces = [4, 0, 1, 2, 3]
    outer = pv.PolyData(outer_points, outer_faces).extrude((0, 0, depth), capping=True).triangulate()
    
    if not make_hole:
        return outer

    # Define cutout (to make L-bracket): a block of size 2×3×1
    cutout_width = 2
    cutout_height = 3
    cutout_depth = depth  # same as original depth

    # Position the cutout at top-right corner (XY view), Z remains the same
    cutout_origin = [width - cutout_width, height - cutout_height, 0]
    hole_points = np.array([
        [cutout_origin[0], cutout_origin[1], 0],
        [cutout_origin[0] + cutout_width, cutout_origin[1], 0],
        [cutout_origin[0] + cutout_width, cutout_origin[1] + cutout_height, 0],
        [cutout_origin[0], cutout_origin[1] + cutout_height, 0]
    ])
    hole_faces = [4, 0, 1, 2, 3]
    hole = pv.PolyData(hole_points, hole_faces).extrude((0, 0, cutout_depth), capping=True).triangulate()
    outer.plot()
    hole.plot()
    # Subtract cutout from base block
    domain = outer.boolean_difference(hole)
    domain.plot()
    return domain

def make_lbracket(width=3.0, height=4.0, cut_width=2.0, cut_height=3.0, depth=1.0):
   # Create the first box (vertical part of the L)
    box1 = pv.Box(bounds=(0, height, 0, width-cut_width, 0, depth)).triangulate()
    box2 = pv.Box(bounds=(0, height-cut_height, 0, width, 0, depth)).triangulate()
    lbracket_3d = box1.merge(box2)
    
    return lbracket_3d




def is_point_inside_domain(domain: pv.PolyData, point: np.ndarray, tol: float = 1e-2) -> bool:
    """
    Check if a 3D point is inside or very close to the surface of a PolyData domain.

    Parameters:
        domain : pv.PolyData
            The surface mesh of the domain.
        point : np.ndarray
            A single 3D point as (x, y, z).
        tol : float
            Tolerance threshold to consider near-surface points as 'inside'.

    Returns:
        bool : True if the point is inside or on/near the surface, else False.
    """
    # Wrap point into a single-point PolyData object
    pt_poly = pv.PolyData(np.array((point)).reshape(1, 3))
    
    # First, try enclosed test
    enclosed = pt_poly.select_enclosed_points(domain, tolerance=tol, check_surface=False)

    is_inside = bool(enclosed["SelectedPoints"][0])

    # If not enclosed, check proximity to surface
    if not is_inside:
        is_inside = is_point_inside_vtk(domain, point)

    return is_inside

def convert_pv_to_vtk_polydata(pv_polydata: pv.PolyData) -> vtk.vtkPolyData:
    """
    Converts a PyVista PolyData object to a native VTK vtkPolyData object.
    This is done safely using DeepCopy from PyVista's underlying VTK object.
    """
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(pv_polydata)
    geometry_filter.Update()

    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.DeepCopy(geometry_filter.GetOutput())
    return vtk_polydata


def is_point_inside_vtk(domain_pv: pv.PolyData, point: np.ndarray, tol:float = 1e-3) -> bool:
    """
    Checks whether a 3D point lies inside a closed domain using
    vtkImplicitPolyDataDistance. Returns True if inside, else False.
    """
    # Convert PyVista PolyData to vtkPolyData
    domain_vtk = convert_pv_to_vtk_polydata(domain_pv)

    # Setup implicit distance object
    implicit_dist = vtk.vtkImplicitPolyDataDistance()
    implicit_dist.SetInput(domain_vtk)

    # Compute signed distance (negative if inside)
    signed_dist = implicit_dist.EvaluateFunction(point)
    return signed_dist < tol


def is_segment_inside_domain(domain: pv.PolyData, line_seg, n_samples=10, eps=1e-6):
    """
    Determines whether a line segment is fully enclosed within the 3D domain.
    Uses sampling of points along the segment and checks if all points are inside.
    """
    pt1 = line_seg.points[0].astype(np.float32)
    pt2 = line_seg.points[1].astype(np.float32)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length == 0:
        return False

    unit_dir = direction / length
    pt1_adj = pt1 + eps * unit_dir
    pt2_adj = pt2 - eps * unit_dir

    # Sample intermediate points
    t_vals = np.linspace(0, 1, n_samples)
    sample_pts = np.array([(1 - t) * pt1_adj + t * pt2_adj for t in t_vals])

    # Check if all points lie inside the domain
    return all(is_point_inside_domain(domain, pt) for pt in sample_pts)


def make_mesh_watertight(mesh: pv.PolyData, hole_size=10.0) -> pv.PolyData:
    """
    Fix common issues and try to make the mesh watertight:
    - Clean
    - Triangulate
    - Fill holes
    """
    # Clean and triangulate
    mesh = mesh.clean(tolerance=1e-6)
    mesh = mesh.triangulate()

    # Use VTK to fill small holes
    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputData(mesh)
    geom_filter.Update()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(geom_filter.GetOutputPort())
    fill_holes.SetHoleSize(hole_size)  # Try increasing if holes aren't filled
    fill_holes.Update()

    # Convert back to PyVista
    filled_mesh = pv.wrap(fill_holes.GetOutput())
    return filled_mesh.clean(tolerance=1e-4)

# --- Convex check using SciPy ConvexHull ---
def is_polygon_convex(domain: pv.PolyData) -> bool:
    # Extract outer boundary points from the base (z = 0) surface
    base_z = np.min(domain.points[:, 2])
    base_points = domain.points[np.abs(domain.points[:, 2] - base_z) < 1e-6]
    
    # Drop the z-coordinate to get 2D projection
    points_2d = base_points[:, :2]
    
    # Remove duplicates to avoid issues in ConvexHull
    points_2d = np.unique(points_2d, axis=0)
    
    if len(points_2d) < 3:
        return False  # Not enough points to form a polygon

    try:
        hull = ConvexHull(points_2d)
        return len(hull.vertices) == len(points_2d)
    except:
        return False  # Degenerate case or error
    
def plot_pyvista_domain(domain: pv.PolyData, Nd, dof, f):
    Nd = np.array(Nd)
    dim = Nd.shape[1]  # Automatically detect 2D or 3D

    dof = np.array(dof).reshape((-1, dim))
    f = np.array(f).reshape((-1, dim))

    fully_fixed = Nd[np.all(dof == 0, axis=1)]
    loads = Nd[np.any(f != 0, axis=1)]

    fig = plt.figure(figsize=(10, 5))

    if dim == 2:
        ax = fig.add_subplot(111)
        base_z = np.min(domain.points[:, 2])
        base_points = domain.points[np.abs(domain.points[:, 2] - base_z) < 1e-6][:, :2]
        hull = base_points[np.argsort(np.arctan2(base_points[:,1] - base_points[:,1].mean(),
                                                 base_points[:,0] - base_points[:,0].mean()))]
        ax.plot(*hull.T, 'k', linewidth=2, label='Domain')
        ax.plot(Nd[:, 0], Nd[:, 1], 'o', markersize=3, color='lightgray', label='Nodes')
        if len(fully_fixed):
            ax.plot(fully_fixed[:, 0], fully_fixed[:, 1], 's', color='blue', label='Fully Fixed')
        if len(loads):
            ax.plot(loads[:, 0], loads[:, 1], 'v', color='red', label='Loads')
        ax.set_aspect('equal')
        ax.set_title("Supports and Loads (2D)")
        ax.legend()
        plt.grid(True)
        plt.show()

    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(domain.points[:, 0], domain.points[:, 1], domain.points[:, 2], 'k.', alpha=0.1, label='Domain Mesh')
        ax.scatter(Nd[:, 0], Nd[:, 1], Nd[:, 2], color='lightgray', label='Nodes', s=10)
        if len(fully_fixed):
            ax.scatter(fully_fixed[:, 0], fully_fixed[:, 1], fully_fixed[:, 2], color='blue', label='Fully Fixed', s=20)
        if len(loads):
            ax.scatter(loads[:, 0], loads[:, 1], loads[:, 2], color='red', marker='v', label='Loads', s=20)
        ax.set_title("Supports and Loads (3D)")
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.show()

    else:
        raise ValueError("Unsupported dimension: must be 2D or 3D")

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_truss_with_domain_and_bcs(
    domain: pv.PolyData,
    Nd,
    dof,
    f,
    Cn,
    a=None,
    q=None,
    threshold=1e-9,
    title="Truss and Domain",
    update=True
):
    """
    Plot truss bars, domain, supports, and loads in 3D.
    If a and q are not provided, all members are plotted in gray.

    Parameters:
        domain : pv.PolyData
            The PyVista 3D domain geometry.
        Nd : ndarray
            Node coordinates (N x 3).
        dof : ndarray
            Degrees of freedom (N x 3), 0 = fixed, 1 = free.
        f : ndarray
            Nodal forces (N x 3).
        Cn : ndarray
            Member connectivity (M x 2 or M x >=4 if extra info).
        a : ndarray, optional
            Cross-sectional areas (M,).
        q : list of ndarray, optional
            Internal forces (len(q) x M).
    """

    Nd = np.array(Nd)
    Cn = np.array(Cn)[:, :2].astype(int)
    dof = np.array(dof).reshape((-1, 3))
    f = np.array(f).reshape((-1, 3))

    fully_fixed = Nd[np.all(dof == 0, axis=1)]
    loads = Nd[np.any(f != 0, axis=1)]

    plt.ion() if update else plt.ioff()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract visible surface and triangulate
    surf = domain.extract_geometry().triangulate()
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    mesh_faces = [surf.points[face] for face in faces]
    mesh_collection = Poly3DCollection(mesh_faces, alpha=0.1, facecolor='gray', edgecolor='none', label='Domain Surface')
    ax.add_collection3d(mesh_collection)

    # Plot nodes
    ax.scatter(Nd[:, 0], Nd[:, 1], Nd[:, 2], color='lightgray', s=10, label='Nodes')

    # Plot supports and loads
    if len(fully_fixed):
        ax.scatter(fully_fixed[:, 0], fully_fixed[:, 1], fully_fixed[:, 2], color='blue', s=30, label='Fully Fixed')
    if len(loads):
        ax.scatter(loads[:, 0], loads[:, 1], loads[:, 2], color='red', marker='v', s=30, label='Loads')
        # Draw arrows for loads (increase arrow length/width for visibility)
        for i in np.where(np.any(f != 0, axis=1))[0]:
            start = Nd[i]
            force = f[i]
            norm = np.linalg.norm(force)
            if norm < 1e-8:
                continue
            direction = force / norm
            arrow_length = 0.5  # Increase this value for longer arrows
            ax.quiver(
                start[0], start[1], start[2],
                direction[0], direction[1], direction[2],
                length=arrow_length, color='red', linewidth=2, arrow_length_ratio=0.2, normalize=True
            )

    # Plot truss bars
    if a is None or q is None:
        for i in range(len(Cn)):
            n1, n2 = Cn[i]
            pos = Nd[[n1, n2]]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='gray', linewidth=1.0)
    else:
        a = np.array(a)
        tk = 5 / max(a)
        for i in range(len(a)):
            if a[i] < threshold:
                continue
            n1, n2 = Cn[i]
            pos = Nd[[n1, n2]]
            if all(qk[i] >= 0 for qk in q):
                color = 'r'
            elif all(qk[i] <= 0 for qk in q):
                color = 'b'
            else:
                color = 'gray'
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=a[i] * tk)

    # Add custom legend handles
    custom_lines = [
        Line2D([0], [0], color='r', lw=2, label='Tension'),
        Line2D([0], [0], color='b', lw=2, label='Compression'),
        Line2D([0], [0], color='gray', lw=2, label='Mixed')
    ]

    ax.set_title(title)
    ax.legend(handles=custom_lines + ax.get_legend_handles_labels()[0])

    # Set aspect ratio and labels
    xrange = np.ptp(Nd[:, 0])
    yrange = np.ptp(Nd[:, 1])
    zrange = np.ptp(Nd[:, 2])
    ax.set_box_aspect([xrange, yrange, zrange])
    ax.set_axis_on()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01) if update else plt.show()


def plot_3d_domain_with_bconditions(
    domain: pv.PolyData,
    Nd: np.ndarray,
    dof: np.ndarray,
    f: np.ndarray
) -> None:
    """
    Visualize a 3D truss domain, nodes, supports, and load vectors.

    Parameters:
        domain (pv.PolyData): 3D PyVista geometry of the domain (possibly with cutouts).
        Nd (np.ndarray): Node coordinates, shape (N, 3).
        dof (np.ndarray): Degree of freedom flags for each node, shape (N, 3),
                          where 0 = fixed, 1 = free in x/y/z.
        f (np.ndarray): Flattened force vector of shape (N * 3,), in [fx1, fy1, fz1, fx2, ...] order.
    """
    assert Nd.ndim == 2 and Nd.shape[1] == 3, "Nd must be (N, 3)"
    assert dof.shape == Nd.shape, "dof must be same shape as Nd"
    #assert f.ndim == 1 and f.size == Nd.shape[0] * 3, "f must be flattened with size 3*N"

    N = Nd.shape[0]
    #f = f.reshape((N, 3))

    p = pv.Plotter()
    p.add_mesh(domain, color="lightgray", opacity=0.3, show_edges=True)

    # Plot all nodes
    p.add_points(Nd, color="black", point_size=5)

    # Plot fully fixed nodes
    fully_fixed = Nd[np.all(dof == 0, axis=1)]
    if len(fully_fixed):
        p.add_points(fully_fixed, color="blue", point_size=10, render_points_as_spheres=True)

    # Ensure f is reshaped to (N, 3)
    f = np.array(f).reshape((N, 3))

    # Plot load vectors as arrows (vectorized for nonzero forces)
    nonzero_forces = np.linalg.norm(f, axis=1) > 1e-8
    for i in np.where(nonzero_forces)[0]:
        start = Nd[i]
        force = f[i]
        direction = (force / np.linalg.norm(force)) * 0.5  # scale for visibility
        arrow = pv.Arrow(start=start, direction=direction,
                        tip_length=0.2, tip_radius=0.05, shaft_radius=0.02)
        p.add_mesh(arrow, color="red")

    p.show_grid()
    p.show()

def plot_line_and_domain(domain, line_seg):
    plotter = pv.Plotter()
    plotter.add_mesh(domain, show_edges=True, opacity=0.5, color='lightblue')
    plotter.add_mesh(line_seg, color='green', line_width=4)
    plotter.show()


if __name__ =='__main__': 
    # Create domain
    #domain = create_domain_with_optional_cutout(3, 4, 1., make_hole=True)
    domain = make_lbracket()
    print('open edges', domain.n_open_edges)
    domain = make_mesh_watertight(domain)

    # Visual test
    print("Open edges:", domain.n_open_edges)
    # Test point and segment
    point = (10, 5, 1e-3)
    segment_start = (0, 0, 1e-3)
    segment_end = (1, 1, 1e-3)
    #pv_point = pv.PolyData(np.array([[point[0], point[1], point[2]]], dtype=np.float32))
    print("Point inside:", is_point_inside_domain(domain, point))
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
    plotter.add_axes()
    # Show
    plotter.show()

