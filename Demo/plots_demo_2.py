import pyvista as pv
import numpy as np
import trimesh

def diagnostic_report(mesh, name="mesh"):
    report = {}

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

    report['type'] = type(mesh).__name__
    report['n_cells'] = mesh.n_cells
    report['n_points'] = mesh.n_points

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
    mesh = mesh.compute_normals(consistent_normals=True, auto_orient_normals=True)

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

# === Example usage ===
if __name__ == "__main__":
    orig_stl = "../Models/Saketh/BliskSectionWithBlade2test.STL"
    mesh_vtu = "../Models/Saketh/test1.vtu"
    result = robust_boolean_difference(orig_stl, mesh_vtu)
    if result:
        result.save("boolean_result.vtk")
        print("\nSaved result to boolean_result.vtk")
    else:
        print("\nBoolean operation failed.")
