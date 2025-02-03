import math
from stl import mesh
from collections import defaultdict
from queue import Queue
import numpy as np
import pyvista as pv

class STLGeom:
    TOL = 1e-9

    def __init__(self, file_path):
        self.mesh = mesh.Mesh.from_file(file_path)
        self.stl_n_triangles = len(self.mesh.vectors)
        self.tri_normals = [self.compute_normal(vertices) for vertices in self.mesh.vectors]
        self.tri_areas = [self.get_area_of_triangle(i) for i in range(self.stl_n_triangles)]
        self.tri_neighbors = self.compute_neighbors()
        self.tri_highlight = [False] * self.stl_n_triangles
        self.selected_triangles = set()
        self.file_path = file_path

    def compute_normal(self, vertices):
        v1 = [vertices[1][i] - vertices[0][i] for i in range(3)]
        v2 = [vertices[2][i] - vertices[0][i] for i in range(3)]
        normal = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        norm = math.sqrt(sum(n ** 2 for n in normal)) or self.TOL
        return [n / norm for n in normal]
    
    def compute_neighbors(self):
        edge_map = defaultdict(list)  # Map of edges to triangle indices
        neighbors = [[] for _ in range(self.stl_n_triangles)]

        for i, vertices in enumerate(self.mesh.vectors):
            edges = [
                tuple(sorted((tuple(vertices[0]), tuple(vertices[1])))),
                tuple(sorted((tuple(vertices[1]), tuple(vertices[2])))),
                tuple(sorted((tuple(vertices[2]), tuple(vertices[0])))),
            ]

            for edge in edges:
                edge_map[edge].append(i)

        for edge, tri_list in edge_map.items():
            for t1 in tri_list:
                for t2 in tri_list:
                    if t1 != t2 and t2 not in neighbors[t1]:
                        neighbors[t1].append(t2)

        return neighbors

    def get_area_of_triangle(self, triangle_index):
        vertices = self.mesh.vectors[triangle_index]
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        return self.compute_area_of_triangle(x, y, z)

    def compute_area_of_triangle(self, x, y, z):
        v1 = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
        v2 = [x[2] - x[0], y[2] - y[0], z[2] - z[0]]

        cross_product = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        cross_product_norm = math.sqrt(sum(c ** 2 for c in cross_product))
        return 0.5 * cross_product_norm
    
    def highlight_triangles_recursive(self, seed_triangle, depth, cutoff_angle_degrees):
        """
        Toggle highlight state for triangles recursively based on the angle between normals.
        If the seed triangle is highlighted, recursively deselect; otherwise, highlight.
        """
        cumulative_area = 0

        cos_theta = math.cos(math.radians(cutoff_angle_degrees))
        # Determine the target state (toggle behavior)
        target_state = not self.tri_highlight[seed_triangle]

        # Initialize queue
        q = Queue()
        q.put((seed_triangle, depth))
        self.tri_highlight[seed_triangle] = target_state 
        n1 = self.tri_normals[seed_triangle]

        while not q.empty():
            current_tri, current_depth = q.get()
            if current_depth == 0:
                continue

            cumulative_area += self.get_area_of_triangle(current_tri)
            n1 = self.tri_normals[current_tri]
            for neighbor_tri in self.tri_neighbors[current_tri]:
                n2 = self.tri_normals[neighbor_tri] 
                if self.tri_highlight[neighbor_tri] != target_state and np.dot(n1, n2) > cos_theta:
                    self.tri_highlight[neighbor_tri] = target_state
                    q.put((neighbor_tri, current_depth - 1))

        return cumulative_area
    
    def plotGeometry(self, show_edges=False, show_axes=True, show_bounding_box=True):
         # Create a PyVista mesh from the STL data
        vertices = stl_geom.mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        faces = np.column_stack((np.full(len(faces), 3), faces))
        mesh = pv.PolyData(vertices, faces)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=show_edges)
        if show_axes:
            plotter.add_axes()
        if show_bounding_box:
            plotter.add_bounding_box()
            # Get bounding box
            bounds = mesh.bounds
            lengths = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]

            # Add text labels for dimensions
            plotter.add_point_labels([[bounds[1], bounds[2], bounds[4]], 
                                    [bounds[0], bounds[3], bounds[4]], 
                                    [bounds[0], bounds[2], bounds[5]]], 
                                    [f'X: {lengths[0]:.3f}', 
                                    f'Y: {lengths[1]:.3f}', 
                                    f'Z: {lengths[2]:.3f}'])
            # Add text label for lowest left corner coordinates
            plotter.add_point_labels([[bounds[0], bounds[2], bounds[4]]],
                                    [f'({bounds[0]:.2f}, {bounds[2]:.2f}, {bounds[4]:.2f})'])
        plotter.show()

    def find_points_single_triangle_distances(self, points, triangle_id):
        """
        Vectorized calculation of minimum distances between multiple points and a single triangle.
        Args:
            points: numpy array of shape (n, 3) containing points [x, y, z]
            triangle_id: index of the triangle in the mesh
        Returns:
            distances: array of minimum distances between points and the triangle
        """
        points = np.asarray(points)
        vertices = self.mesh.vectors[triangle_id]
        normal = np.array(self.tri_normals[triangle_id])
        
        # Project points onto triangle plane
        point_to_plane = np.dot(points - vertices[0], normal)
        projected_points = points - point_to_plane[:, np.newaxis] * normal
        
        # Compute barycentric coordinates
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = projected_points - vertices[0]
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Check if points project inside triangle
        inside_triangle = (u >= 0) & (v >= 0) & (w >= 0) & (np.abs(u + v + w - 1.0) < self.TOL)
        
        # Calculate distances to edges where needed
        distances = np.where(inside_triangle,
                            np.linalg.norm(points - projected_points, axis=1),
                            np.minimum.reduce([
                                np.linalg.norm(np.cross(vertices[1] - vertices[0], points - vertices[0]), axis=1) / np.linalg.norm(vertices[1] - vertices[0]),
                                np.linalg.norm(np.cross(vertices[2] - vertices[1], points - vertices[1]), axis=1) / np.linalg.norm(vertices[2] - vertices[1]),
                                np.linalg.norm(np.cross(vertices[0] - vertices[2], points - vertices[2]), axis=1) / np.linalg.norm(vertices[0] - vertices[2])
                            ]))
        
        return distances
   
    
    
    def find_point_triangle_distance(self, point, triangle_id):
        """
        Calculate the minimum distance between a point and a triangle.
        Args:
            point: numpy array [x, y, z]
            triangle_id: index of the triangle in the mesh
        Returns:
            distance: minimum distance between point and triangle
        """
        vertices = self.mesh.vectors[triangle_id]
        
        # Convert to numpy arrays for easier computation
        p = np.array(point)
        a = np.array(vertices[0])
        b = np.array(vertices[1])
        c = np.array(vertices[2])
        
        # Calculate triangle normal
        normal = np.array(self.tri_normals[triangle_id])
        
        # Project point onto triangle plane
        plane_point = p - np.dot(p - a, normal) * normal
        
        # Compute barycentric coordinates
        v0 = b - a
        v1 = c - a
        v2 = plane_point - a
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # If barycentric coordinates are all between 0 and 1,
        # projected point lies within triangle
        if (u >= 0 and v >= 0 and w >= 0 and abs(u + v + w - 1.0) < self.TOL):
            return np.linalg.norm(p - plane_point)
        
        # Otherwise, find minimum distance to triangle edges
        edge_distances = [
            np.linalg.norm(np.cross(b - a, p - a)) / np.linalg.norm(b - a),
            np.linalg.norm(np.cross(c - b, p - b)) / np.linalg.norm(c - b),
            np.linalg.norm(np.cross(a - c, p - c)) / np.linalg.norm(a - c)
        ]
        
        return min(edge_distances)

if __name__ == "__main__":
    import os
    import pyvista as pv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #stl_file = os.path.join(script_dir, '../TOExamples/AlcoaGrabCAD/AlcoaGrabCAD.STL')
    #stl_file =  os.path.join(script_dir, '../TOExamples/CompliantMechanism/CompliantMechanism.STL')
    stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
    stl_geom = STLGeom(stl_file)
    stl_geom.plotGeometry()
    # Create a plotter and add the mesh
    
    # Generate random 3D points
    random_points = np.random.uniform(-1, 1, size=(10, 3))
    triangle_id = 13    
    print(stl_geom.find_points_single_triangle_distances(random_points, triangle_id))