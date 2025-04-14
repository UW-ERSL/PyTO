import math
from stl import mesh #pip install numpy-stl
from collections import defaultdict
from queue import Queue
import numpy as np



class STLGeom:
    TOL = 1e-9

    def __init__(self, file_path):
        self.mesh = mesh.Mesh.from_file(file_path)
        
        self.stl_n_triangles = len(self.mesh.vectors)
        #self.tri_normals = [self.compute_normal(vertices) for vertices in self.mesh.vectors]
        self.tri_normals = self.compute_normals_vectorized()
        #self.tri_areas = [self.get_area_of_triangle(i) for i in range(self.stl_n_triangles)]
        self.tri_areas = self.compute_areas_vectorized()
        self.tri_neighbors = self.compute_neighbors()
        self.tri_highlight = [False] * self.stl_n_triangles
        self.selected_triangles = set()
        self.file_path = file_path

    def get_bounding_box(self):
        """
        Compute the bounding box (min/max coordinates) of the STL geometry.
        Returns:
            tuple: (xmin, xmax, ymin, ymax, zmin, zmax) coordinates of the bounding box
        """
        if not hasattr(self, 'mesh') or self.mesh is None:
            return (0, 0, 0, 0, 0, 0)
        # Get all vertices as a single array
        vertices = self.mesh.vectors.reshape(-1, 3)
        # Calculate min and max for each coordinate
        xmin, ymin, zmin = np.min(vertices, axis=0)
        xmax, ymax, zmax = np.max(vertices, axis=0)
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def compute_areas_vectorized(self):
        """Compute all triangle areas using vectorized operations"""
        vectors = self.mesh.vectors
        v1 = vectors[:, 1] - vectors[:, 0]  # Edge vectors from v0 to v1
        v2 = vectors[:, 2] - vectors[:, 0]  # Edge vectors from v0 to v2
        
        # Cross product of edges 
        cross = np.cross(v1, v2)
        
        # Area = 0.5 * |cross product|
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return areas.tolist()
    def compute_normals_vectorized(self):
        """Compute all triangle normals using vectorized operations"""
        vectors = self.mesh.vectors
        v1 = vectors[:, 1] - vectors[:, 0]  # Edge vectors from v0 to v1 
        v2 = vectors[:, 2] - vectors[:, 0]  # Edge vectors from v0 to v2
        
        # Cross product of edges gives normals 
        normals = np.cross(v1, v2)
        
        # Normalize the normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < self.TOL] = self.TOL # Avoid division by zero
        normals = normals / norms
        
        return normals.tolist()
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
    
    def find_nearest_triangle_normal(self, point):
        """
        Find the outward normal of the nearest triangle to a given point.
        Args:
            point: numpy array [x, y, z] coordinates
        Returns:
            normal: outward normal vector of nearest triangle
            distance: distance to nearest triangle
        """
        point = np.array(point)
        min_dist = float('inf')
        min_normal = None
        
        for i in range(self.stl_n_triangles):
            dist = self.find_point_triangle_distance(point, i)
            if dist < min_dist:
                min_dist = dist
                min_normal = self.tri_normals[i]
        
        return min_normal, min_dist
         
    
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
    
    
    def integrate_polynomial_over_triangle(self, triangle_id, exponents,nGaussPts=1):
        """
        Integrate polynomial x^a * y^b * z^c over a triangle using Gaussian quadrature.
        Args:
            triangle_id: index of triangle
            exponents: tuple (a,b,c) for powers of x,y,z
        Returns:
            integral value
        """
        # Gaussian quadrature points and weights for triangles
        if nGaussPts == 1:
            gauss_points = [[0.333333333333333, 0.333333333333333]]
            gauss_weights = [1.0]
        elif nGaussPts == 3:
            gauss_points = [
            [0.166666666666667, 0.166666666666667],
            [0.166666666666667, 0.666666666666667],
            [0.666666666666667, 0.166666666666667]
            ]
            gauss_weights = [0.333333333333333, 0.333333333333333, 0.333333333333333]
        else:
            raise ValueError(f"Unsupported number of Gauss points: {nGaussPts}")

        vertices = self.mesh.vectors[triangle_id]
        area = self.tri_areas[triangle_id]
        
        result = 0.0
        # Convert to numpy arrays for vectorized operations
        gauss_points = np.array(gauss_points)
        gauss_weights = np.array(gauss_weights)
        vertices = np.array(vertices)

        # Vectorized mapping of points to triangle
        mapped_points = vertices[0] + np.outer(gauss_points[:, 0], vertices[1] - vertices[0]) + \
                   np.outer(gauss_points[:, 1], vertices[2] - vertices[0])

        # Vectorized polynomial evaluation
        result = np.sum(gauss_weights * 
                   mapped_points[:, 0]**exponents[0] * 
                   mapped_points[:, 1]**exponents[1] * 
                   mapped_points[:, 2]**exponents[2])
        return result *  area
    

    def compute_mass_properties(self):
        """
        The computation is based on divergence theorem, converting volume integrals to surface integrals:
        1. Volume is computed by integrating x·n over the surface
        2. Center of mass uses second order moments (x²·n, y²·n, z²·n) 
        3. Inertia tensor uses third order moments and cross terms
        The process:
        1. Iterates through each triangle in the STL mesh
        2. For each triangle:
            - Adds its area contribution
            - Computes volume contribution using divergence theorem
            - Computes center of mass contributions using second moments
            - Computes inertia tensor contributions using third moments and cross terms
        3. Normalizes center of mass by volume
        4. Assembles the final inertia tensor
        Parameters
        ----------
        None
        Returns
        -------
        area : float
             Total surface area of the geometry
        volume : float
             Volume of the geometry
        center_of_mass : numpy.ndarray
             3D vector containing (x,y,z) coordinates of the center of mass
        inertia_tensor : numpy.ndarray
             3x3 matrix representing the inertia tensor about the center of mass
        Notes
        -----
        The computation assumes the STL represents a closed manifold surface.
        The inertia tensor is computed about the origin of the coordinate system.
        Integration is performed using Gaussian quadrature with variable number of points.
        
        """
        area = 0.0
        volume = 0.0
        center_of_mass = np.zeros(3)
        inertia_tensor = np.zeros((3, 3))
        crossTerms = np.zeros((3, 3))
        for t in range(self.stl_n_triangles):
            normal = self.tri_normals[t]
            area += self.tri_areas[t]
            # Compute volume
            volume += normal[0]*self.integrate_polynomial_over_triangle(t, (1,0,0),nGaussPts=1)

            # Compute center of mass
            center_of_mass[0] += 0.5*normal[0]*self.integrate_polynomial_over_triangle(t, (2,0,0),nGaussPts=1)
            center_of_mass[1] += 0.5*normal[1]*self.integrate_polynomial_over_triangle(t, (0,2,0),nGaussPts=1)
            center_of_mass[2] += 0.5*normal[2]*self.integrate_polynomial_over_triangle(t, (0,0,2),nGaussPts=1)

            # Integral of x^2, y^2, z^2, x*y, etc. over the triangle
            crossTerms[0,0] += (1/3.0)*normal[0]*self.integrate_polynomial_over_triangle(t, (3,0,0),nGaussPts=3)
            crossTerms[1,1] += (1/3.0)*normal[1]*self.integrate_polynomial_over_triangle(t, (0,3,0),nGaussPts=3)
            crossTerms[2,2] += (1/3.0)*normal[2]*self.integrate_polynomial_over_triangle(t, (0,0,3),nGaussPts=3)

            crossTerms[0,1] += normal[2]*self.integrate_polynomial_over_triangle(t, (1,1,1),nGaussPts=3)
            crossTerms[0,2] += normal[1]*self.integrate_polynomial_over_triangle(t, (1,1,1),nGaussPts=3)
            crossTerms[1,2] += normal[0]*self.integrate_polynomial_over_triangle(t, (1,1,1),nGaussPts=3)

        center_of_mass /= volume
        inertia_tensor[0,0] = crossTerms[1,1] + crossTerms[2,2]
        inertia_tensor[1,1] = crossTerms[0,0] + crossTerms[2,2]
        inertia_tensor[2,2] = crossTerms[0,0] + crossTerms[1,1]


        inertia_tensor[0,1] = -crossTerms[0,1]
        inertia_tensor[1,0] = -crossTerms[0,1]
        inertia_tensor[0,2] = -crossTerms[0,2]
        inertia_tensor[2,0] = -crossTerms[0,2]
        inertia_tensor[1,2] = -crossTerms[1,2]
        inertia_tensor[2,1] = -crossTerms[1,2]
        
        return area, volume, center_of_mass, inertia_tensor

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
    
    def get_triangle_data(self, index):
        """Get full triangle data for a given index"""
        if not 0 <= index < self.stl_n_triangles:
            return None
        # Create triangle data structure similar to store_selected_triangles
        triangle_data = {
            'index': index,
            'vertices': self.mesh.vectors[index],
            'normal': self.tri_normals[index],
            'area': self.tri_areas[index],
            'center': self.get_triangle_center(index)
        }
        return triangle_data
    
    def plotGeometry(self, show_edges=False, show_axes=True, show_bounding_box=True):
         # Create a PyVista mesh from the STL data
        vertices = self.mesh.vectors.reshape(-1, 3)
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
   
    def find_points_triangle_distances_loop(self, points, triangle_id):
        """
        Calculate the minimum distances between multiple points and a single triangle.
        Args:
            points: list of numpy arrays, each representing [x, y, z] coordinates
            triangle_id: index of the triangle in the mesh
        Returns:
            distances: list of minimum distances between each point and the triangle
        """
        distances = []
        for point in points:
            distance = self.find_point_triangle_distance(point, triangle_id)
            distances.append(distance)
        return np.array(distances)
  
    def find_points_triangle_distances_vectorized(self, points, triangle_id):
        """
        Calculate the minimum distances between multiple points and a single triangle using vectorized operations.
        Args:
            points: numpy array of shape (n, 3), each row representing [x, y, z] coordinates
            triangle_id: index of the triangle in the mesh
        Returns:
            distances: numpy array of minimum distances between each point and the triangle
        """

        vertices = self.mesh.vectors[triangle_id]
        points = np.array(points)
        a, b, c = np.array(vertices[0]), np.array(vertices[1]), np.array(vertices[2])

        # Calculate normal of the triangle
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)

        # Project points onto the plane of the triangle
        plane_points = points - np.dot(points - a, normal)[:, np.newaxis] * normal
 
        # Check if the projected points are inside the triangle using barycentric coordinates
        v0, v1 = b - a, c - a
        d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
        v2 = plane_points - a
        d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        inside_triangle = (u >= 0) & (v >= 0) & (w >= 0)
        distances = np.full(points.shape[0], np.inf)

        # Calculate distances for points inside the triangle
        distances[inside_triangle] = np.linalg.norm(points[inside_triangle] - plane_points[inside_triangle], axis=1)

        # Calculate distances for points outside the triangle
        def point_to_segment_distance(p, a, b):
            ab = b - a
            ap = p - a
            t = np.sum(ap * ab, axis=1) / np.sum(ab * ab)
            t = np.clip(t, 0, 1)
            projection = a + np.outer(t, ab)
            return np.linalg.norm(p - projection, axis=1)

        if not np.all(inside_triangle):
            points_outside = points[~inside_triangle]
            distances_outside = np.min(np.array([
                point_to_segment_distance(points_outside, a, b),
                point_to_segment_distance(points_outside, b, c),
                point_to_segment_distance(points_outside, c, a)
            ]), axis=0)
            distances[~inside_triangle] = distances_outside

        return distances
    
    def find_point_triangle_distance(self, point, triangle_id):
        """
        Find the shortest distance from a point to a triangle.
        Args:
            point: numpy array [x, y, z]
            triangle_id: index of the triangle in the mesh
        Returns:
            distance: shortest distance from point to triangle
        """
        vertices = self.mesh.vectors[triangle_id]
        p = np.array(point)
        a, b, c = np.array(vertices[0]), np.array(vertices[1]), np.array(vertices[2])

        # Calculate normal of the triangle
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)

        # Project point onto the plane of the triangle
        plane_point = p - np.dot(p - a, normal) * normal

        # Check if the projected point is inside the triangle using barycentric coordinates
        v0, v1, v2 = b - a, c - a, plane_point - a
        d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
        d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        if (u >= 0) and (v >= 0) and (w >= 0):
            return np.linalg.norm(p - plane_point)

        # If the projected point is outside the triangle, find the shortest distance to the edges
        def point_to_segment_distance(p, a, b):
            ab = b - a
            t = np.dot(p - a, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            projection = a + t * ab
            return np.linalg.norm(p - projection)

        distances = [
            point_to_segment_distance(p, a, b),
            point_to_segment_distance(p, b, c),
            point_to_segment_distance(p, c, a)
        ]

        return min(distances)

    
    def get_triangle_center(self, triangle_index):
        vertices = self.mesh.vectors[triangle_index]
        center = [(vertices[0][i] + vertices[1][i] + vertices[2][i])/3 for i in range(3)]
        return center

    def highlight_triangles_recursive(self, seed_triangle, depth, cutoff_angle_degrees):
        """
        Toggle highlight state for triangles recursively based on the angle between normals.
        If the seed triangle is highlighted, recursively deselect; otherwise, highlight.
        """

        cumulative_area = 0
        cos_theta = math.cos(math.radians(cutoff_angle_degrees))

        # Always set to True for left click (no more toggle)
        target_state = True

        # Initialize queue
        q = Queue()
        q.put((seed_triangle, depth))
        self.tri_highlight[seed_triangle] = target_state

        # Keep track of processed triangles to avoid cycles
        processed = {seed_triangle}

        while not q.empty():
            current_tri, current_depth = q.get()
            if current_depth == 0:
                continue

            cumulative_area += self.get_area_of_triangle(current_tri)
            n1 = self.tri_normals[current_tri]

            for neighbor_tri in self.tri_neighbors[current_tri]:
                if neighbor_tri not in processed:
                    n2 = self.tri_normals[neighbor_tri]
                if not self.tri_highlight[neighbor_tri] and np.dot(n1, n2) > cos_theta:
                    self.tri_highlight[neighbor_tri] = target_state
                    q.put((neighbor_tri, current_depth - 1))
                    processed.add(neighbor_tri)

        highlighted_count = sum(1 for x in self.tri_highlight if x)
        return highlighted_count, cumulative_area

    def store_selected_triangles(self):
        selected_indices = [i for i, is_highlighted in enumerate(self.tri_highlight) if is_highlighted]
        selected_triangles_data = []
        
        for idx in selected_indices:
            triangle_data = {
                'index': idx,
                'vertices': self.mesh.vectors[idx],
                'normal': self.tri_normals[idx],
                'area': self.tri_areas[idx],
                'center': self.get_triangle_center(idx)
            }
            selected_triangles_data.append(triangle_data)
   
        return selected_triangles_data
   
if __name__ == "__main__":
    import os
    import pyvista as pv
    script_dir = os.path.dirname(os.path.abspath(__file__))

    stl_file = os.path.join(script_dir, '../Models/CantileverBeam/CantileverBeam.STL')
    stl_file =  os.path.join(script_dir, '../Models/CompliantMechanism/CompliantMechanism.STL')
    stl_file =  os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')
    stl_file =  os.path.join(script_dir, '../Models/BliskModel/BliskQuarter.STL')
    
    
    stl_file = os.path.join(script_dir, '../Models/AlcoaGrabCAD/AlcoaGrabCAD.STL')
    stl_file = os.path.join(script_dir, '../Models/Overhang/Overhang.STL')
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    stl_file =  os.path.join(script_dir, '../Models/KnuckleAssembly/KnuckleAssembly.STL')
    stl_file =  os.path.join(script_dir, '../Models/Table/Table.STL')
    stl_file = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
    #stl_file = os.path.join(script_dir, '../Models/Nosecone/HollowNosecone.STL')
    stl_geom = STLGeom(stl_file)

    [area, volume, cg, inertia] = stl_geom.compute_mass_properties()
    print(f"Area: {area}")
    print(f"Volume: {volume}")
    print(f"Center of Mass: {cg}")
    print(f"Inertia: {inertia}")


    stl_geom.plotGeometry(show_edges=True, show_axes=True, show_bounding_box=True)
