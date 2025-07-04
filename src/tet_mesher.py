
import numpy as np
import pyvista as pv # pip install pyvista
from scipy.sparse import coo_matrix
import tetgen #pip install tetgen


class TetMesher:
    def __init__(self):	
        self.num_nodes = 0
        self.num_elems = 0
    
    def createTetMeshFromSTLFile(self, stlFileName: str, mergeFacets = True, nElemsDesired: int = 10000, elemSizeFunction = None):
        """
        Create a tetrahedral mesh from an STL file.
        This function reads an STL file, cleans and repairs the surface, and generates a tetrahedral mesh
        with a target number of elements. It also calculates and stores the nodes, elements, and surface triangles.
        Parameters:
        stlFileName (str): The path to the STL file.
        nElemsDesired (int, optional): The desired number of tetrahedral elements. Default is 10000.
        Attributes:
        self.stlGeom (pyvista.PolyData): The cleaned and repaired STL surface mesh.
        self.node_xyz (numpy.ndarray): The array of node coordinates.
        self.elems (numpy.ndarray): The array of tetrahedral elements.
        self.num_nodes (int): The number of nodes in the mesh.
        self.num_elems (int): The number of elements in the mesh.
        self.surface_triangles (numpy.ndarray): The array of surface triangles.
        self.elem_size (float): The average size of the elements.
        Prints:
        Total volume of the surface mesh.
        Number of nodes, elements, and surface triangles.
        Average element size.
        """

        self.stlGeom = pv.read(stlFileName)
        # Clean and repair the STL surface
        surf = self.stlGeom.clean()

        # Calculate approximate volume constraint based on desired number of elements
        total_volume = surf.volume
        print(f"Total volume: {total_volume}")
  

        # Add a scaling factor to get closer to desired element count
        max_tet_volume = (total_volume / nElemsDesired)  

        # Generate tetrahedral mesh with target number of cells and quality constraints
        tet = tetgen.TetGen(surf)

        # If an element size function is provided, use it to create a background mesh
        if elemSizeFunction is not None:
            resolution = 10
        
            x_min, x_max, y_min, y_max, z_min, z_max = self.stlGeom.bounds

            x_vals = np.linspace(x_min, x_max, resolution)
            y_vals = np.linspace(y_min, y_max, resolution)
            z_vals = np.linspace(z_min, z_max, resolution)
            grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

            # Create structured grid
            bgmesh = pv.StructuredGrid(grid_x, grid_y, grid_z).triangulate()

            # Compute target size field
            sizes = np.zeros(bgmesh.n_points)
            for i in range(bgmesh.n_points):
                pt = bgmesh.points[i, :3]
                sizes[i] = elemSizeFunction(pt)
            bgmesh.point_data["target_size"] = sizes

        else:
            bgmesh = None

        if (mergeFacets):
            if (bgmesh is None):
                nodes, elements = tet.tetrahedralize(switches=f"pq1.5a{max_tet_volume}Q")
            else:
                nodes, elements = tet.tetrahedralize(switches=f"pq1.5a{max_tet_volume}mQ",bgmesh = bgmesh)
        else:
            if (bgmesh is None):
                nodes, elements = tet.tetrahedralize(switches=f"pq1.5a{max_tet_volume}QM")
            else:
                nodes, elements = tet.tetrahedralize(switches=f"pq1.5a{max_tet_volume}mQM",bgmesh = bgmesh)
        self.node_xyz = nodes
        self.origin = np.min(self.node_xyz, axis=0)
        self.elems = elements
        self.num_nodes = len(self.node_xyz)
        self.num_elems = len(self.elems)
        print(f"Tetmesh: Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")
        # Compute total mesh volume by summing volumes of all tetrahedra
        def tet_volume(a, b, c, d):
            return abs(np.dot((a - d), np.cross((b - d), (c - d)))) / 6.0

        volumes = np.array([
            tet_volume(self.node_xyz[e[0]], self.node_xyz[e[1]], self.node_xyz[e[2]], self.node_xyz[e[3]])
            for e in self.elems
        ])
        self.volume = np.sum(volumes)

        self.createSurfaceMesh()
    
    def createSurfaceMesh(self):
        """
        Create a surface mesh from the tetrahedral mesh.
        This function extracts the surface triangles from the tetrahedral mesh and calculates their properties.
        Attributes:
        self.surface_triangles (numpy.ndarray): The array of surface triangles.
        self.elem_size (float): The average size of the elements.
        """
        # Get all faces from tetrahedra
        faces = np.vstack([
            self.elems[:, [0, 2, 1]],
            self.elems[:, [0, 1, 3]],
            self.elems[:, [1, 2, 3]],
            self.elems[:, [3, 2, 0]]
        ])

        # Sort faces for comparison
        sortedfaces = np.sort(faces, axis=1)

        # Find unique faces and their counts
        _, idx, counts = np.unique(sortedfaces, axis=0, return_index=True, return_counts=True)

        # Surface triangles are faces that appear only once
        self.surface_triangles = faces[idx[counts == 1]]
        
        print(f"Number of surface triangles: {len(self.surface_triangles)}")

        element_sizes = np.zeros(self.num_elems)
        for i in range(self.num_elems):
            element_sizes[i] = np.linalg.norm(self.node_xyz[self.elems[i, 0], :3] - self.node_xyz[self.elems[i, 1], :3])
        self.elem_size = np.mean(element_sizes)
        print(f"Element size: {self.elem_size}")


    def read_Abaqus_linear_tetmesh(self, abaqusFileName: str):
        """
        Read an Abaqus input file and extract nodes and elements.
        The files are generated via SolidWorks.
        Parameters:
        abaqusFileName (str): The path to the Abaqus input file.
        Attributes:
        self.node_xyz (numpy.ndarray): The array of node coordinates.
        self.elems (numpy.ndarray): The array of tetrahedral elements.
        self.num_nodes (int): The number of nodes in the mesh.
        self.num_elems (int): The number of elements in the mesh.
        """
        with open(abaqusFileName, 'r') as f:
            lines = f.readlines()
        print(f"Reading Abaqus input file: {abaqusFileName}")
        print(f"Number of lines: {len(lines)}")

        # Extract nodes
        node_start = next(i for i, line in enumerate(lines) if '*NODE' in line.upper()) + 1
        node_end = next(i for i, line in enumerate(lines[node_start:], node_start) 
                       if '*ELEMENT' in line.upper())
        node_lines = lines[node_start:node_end]
        nodes = np.array([list(map(float, line.split(','))) for line in node_lines])
        self.node_xyz = nodes[:, 1:4]
        self.origin = np.min(self.node_xyz, axis=0)
        self.num_nodes = len(self.node_xyz)
        element_start = next(i for i, line in enumerate(lines[node_end:], node_end) if '*ELEMENT' in line.upper()) + 1
        element_end = next(i for i, line in enumerate(lines[element_start:], element_start) if '*SOLID' in line.upper())
        element_lines = lines[element_start:element_end]
        elements = np.array([list(map(int, line.split(','))) for line in element_lines])
        self.elems = elements[:, 1:5]-1  # Convert to zero-based indexing
        self.num_elems = len(self.elems)
        print(f"Tetmesh: Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")
        self.createSurfaceMesh()

    def createQuadraticTetMesh(self):
        # 10 noded quadratic tetrahedral elements
        # Create quadratic 10-noded tet mesh from linear 4-noded tet mesh
        # First copy the nodes and elements
        '''
        The implementation assumes that the 10 nodes are numbered as follows:

            Nodes 0-3: The four vertices of the tetrahedron.
            Nodes 4-9: The mid-edge nodes, arranged so that:
            Node 4 lies on the edge between vertex 0 and vertex 1.
            Node 5 lies on the edge between vertex 1 and vertex 2.
            Node 6 lies on the edge between vertex 2 and vertex 0.
            Node 7 lies on the edge between vertex 0 and vertex 3.
            Node 8 lies on the edge between vertex 1 and vertex 3.
            Node 9 lies on the edge between vertex 2 and vertex 3.
        '''
        node_xyz = self.node_xyz.copy()
        elems = self.elems.copy()

        # Create edges for each element
        edges = []
        for e in elems:
            edges.extend([(e[0], e[1]), (e[1], e[2]), (e[2], e[0]),
                        (e[0], e[3]), (e[1], e[3]), (e[2], e[3])])

        # Remove duplicates and sort edge nodes
        unique_edges = list(set(tuple(sorted(edge)) for edge in edges))

        # Add midpoint nodes
        mid_nodes_xyz = node_xyz[unique_edges].mean(axis=1)
        num_mid_nodes = len(mid_nodes_xyz)

        # Update nodes array with mid-nodes
        node_xyz = np.vstack((node_xyz, mid_nodes_xyz))

        # Create node mapping for edge midpoints
        edge_to_node = {tuple(sorted(edge)): i + self.num_nodes 
                        for i, edge in enumerate(unique_edges)}

        # Update elements with mid-node indices
        quad_elems = np.zeros((self.num_elems, 10), dtype=np.int32)
        quad_elems[:, :4] = elems  # Corner nodes
        for i, e in enumerate(elems):
            # Add mid-nodes
            quad_elems[i, 4] = edge_to_node[tuple(sorted((e[0], e[1])))]
            quad_elems[i, 5] = edge_to_node[tuple(sorted((e[1], e[2])))]
            quad_elems[i, 6] = edge_to_node[tuple(sorted((e[2], e[0])))]
            quad_elems[i, 7] = edge_to_node[tuple(sorted((e[0], e[3])))]
            quad_elems[i, 8] = edge_to_node[tuple(sorted((e[1], e[3])))]
            quad_elems[i, 9] = edge_to_node[tuple(sorted((e[2], e[3])))]

        quadratic_mesh = type('QuadTetMesh', (), {
            'node_xyz': node_xyz,
            'elems': quad_elems,
            'origin': np.min(node_xyz, axis=0),
            'num_nodes': len(node_xyz),
            'num_elems': self.num_elems
        })
        print(f"Quadratic Tetmesh: Number of nodes: {quadratic_mesh.num_nodes}, Number of elements: {quadratic_mesh.num_elems}")
        # Create surface triangles for quadratic mesh
        return quadratic_mesh


    def integrate_over_surface_triangles(self, q, tri_surface_indices):
        """
        Integrates a given quantity over specified surface triangles and distributes the resulting force 
        over the nodes of the triangles.
        Parameters:
        -----------
        q : float
            The quantity to be integrated over the surface triangles.
        tri_surface_indices : array-like
            Indices of the surface triangles over which the integration is to be performed.
        Returns:
        --------
        force_vector : numpy.ndarray
            The force vector distributed over the nodes, normalized by the total area of the triangles.
        Notes:
        ------
        - The method assumes that `self.surface_triangles` is an array where each row represents a triangle 
          by storing the indices of its three nodes.
        - The method assumes that `self.node_xyz` is an array where each row represents the coordinates of a node.
        - The force is distributed equally among the three nodes of each triangle.
        - The total area of the triangles is printed for debugging purposes.
        """

        # Initialize force vector
        force_vector = np.zeros(self.num_nodes)
        surf_triangles = self.surface_triangles[tri_surface_indices, :]
        # Loop over each surface triangle
        total_area = 0.0
        for tri in surf_triangles:
            # Get the nodes of the triangle
            node_coords = self.node_xyz[tri, :]
            # Calculate the area of the triangle
            vec1 = node_coords[1] - node_coords[0]
            vec2 = node_coords[2] - node_coords[0]
            tri_area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            total_area += tri_area
            # Distribute the force over the triangle nodes
            for node in tri:
                force_vector[node] += q * tri_area / 3.0
        # print(f"Total area of triangles: {total_area}")
        return force_vector/total_area
    
    def integrate_function_over_surface_triangles(self, func, tri_surface_indices): 
        """
        Integrates a given function over specified surface triangles and distributes the result as a force vector.
        Parameters:
        -----------
        func : callable
            A function that takes a coordinate (numpy array) as input and returns a scalar value.
        tri_surface_indices : arsray-like
            Indices of the surface triangles over which the function is to be integrated.
        Returns:
        --------
        force_vector : numpy array
            The force vector distributed over the nodes, normalized by the total area of the triangles.
        Notes:
        ------
        - The method calculates the area of each triangle and evaluates the function at the centroid of the triangle.
        - The force is distributed equally among the nodes of each triangle.
        - The total area of the triangles is printed for reference.
        """

        # Initialize force vector
        force_vector = np.zeros(self.num_nodes)
        surf_triangles = self.surface_triangles[tri_surface_indices, :]
        # Loop over each surface triangle
        total_area = 0.0
        for tri in surf_triangles:
            # Get the nodes of the triangle
            node_coords = self.node_xyz[tri, :]
            # Calculate the area of the triangle
            vec1 = node_coords[1] - node_coords[0]
            vec2 = node_coords[2] - node_coords[0]
            tri_area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            total_area += tri_area
            # Distribute the force over the triangle nodes
            center = np.mean(node_coords, axis=0)
            q = func(center)
            for node in tri:
                force_vector[node] += q * tri_area / 3.0
        # print(f"Total area of triangles: {total_area}")
        return force_vector
    
    def compute_surface_triangle_properties(self):
        """
        Computes unit normal vectors for all surface triangles.
        Returns:
        --------
        normals : np.ndarray of shape (n_surface_triangles, 3)
            The unit normal vector for each triangle.
        """
        centers = np.zeros((len(self.surface_triangles), 3))
        normals = np.zeros((len(self.surface_triangles), 3))
        areas = np.zeros(len(self.surface_triangles))
        for i, tri in enumerate(self.surface_triangles):
            p0, p1, p2 = self.node_xyz[tri]
            centers[i] = (p0 + p1 + p2) / 3
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            areas[i] = 0.5*norm
            if norm > 0:
                normal /= norm
            normals[i] = normal
        return centers, normals, areas
    
    def get_nodes_from_locations(self, locations):
        distances = np.linalg.norm(self.node_xyz[:, :3] - locations[:, None], axis=2)
        return np.argmin(distances, axis=1)
    
    def get_nodes_within_annular_region(self, centerPt, axis, innerRadius, outerRadius):
        # Get the nodes within the annular region defined by the center point, axis, and radii
        axis = axis / np.linalg.norm(axis)
        axis = axis.reshape(1, 3)
        nodes = self.node_xyz[:, :3] - centerPt
        proj = np.dot(nodes, axis.T)
        nodes = nodes - proj * axis
        dist = np.linalg.norm(nodes, axis=1)
        return np.where((dist >= innerRadius) & (dist <= outerRadius))[0]
    
    def getNodesOnBoundingBoxPlane(self, axis: int, min_limit: bool): 
        """
        Get the nodes on a bounding box plane for the tetrahedral mesh.

        Args:
            axis (int): The axis of the bounding box plane (0 = x, 1 = y, 2 = z).
            min_limit (bool): Whether to get nodes on the minimum (True) or maximum (False) plane.

        Returns:
            np.ndarray: Indices of nodes on the specified bounding box plane.
        """
        if not hasattr(self, 'node_xyz') or self.node_xyz is None:
            raise ValueError("Node coordinates are not defined. Please ensure the mesh is loaded.")

        # Determine the plane coordinate (min or max along the specified axis)
        plane_coord = np.min(self.node_xyz[:, axis]) if min_limit else np.max(self.node_xyz[:, axis])

        # Find nodes on the specified plane (within a small tolerance to account for floating-point errors)
        tolerance = 1e-6
        nodes_on_plane = np.where(np.abs(self.node_xyz[:, axis] - plane_coord) <= tolerance)[0]

        return nodes_on_plane

    def get_nodes_within_radius(self, pt: np.ndarray, r: float) -> np.ndarray: 
            """Find nodes within a given radius from a point.
            
            Args:
                pt: Array of shape (3,) containing x, y, z coordinates of the point
                r: Radius within which to find nodes
                
            Returns:
                np.ndarray: Indices of nodes within the given radius
            """
            # Calculate squared distances from the point to all nodes
            distances_sq = np.sum((self.node_xyz - pt)**2, axis=1)
            
            # Find nodes within the radius (compare squared distances to squared radius)
            nodes_within_radius = np.where(distances_sq <= r**2)[0]
            return nodes_within_radius
    
    def get_boundary_nodes(self) -> np.ndarray: 
        """
        Find nodes that lie on the boundary of a tetrahedral mesh.

        Returns:
            np.ndarray: Array of unique node indices that are on the boundary.
        """
        if not hasattr(self, 'surface_triangles') or self.surface_triangles is None:
            raise ValueError("Surface triangles are not defined. Please ensure the mesh is created.")

        # Extract unique node indices from surface triangles
        boundary_nodes = np.unique(self.surface_triangles.flatten())
        return boundary_nodes

    # def get_element_containing_point(self, point: np.ndarray) -> tuple: 
    #     """Find the element that contains the given point in a tetrahedral mesh and compute its shape functions.
        
    #     Args:
    #         point: Array of shape (3,) containing x, y, z coordinates.
            
    #     Returns:
    #         tuple: (Index of the element containing the point, Shape function values at the point),
    #             or (-1, None) if no element is found.
    #     """
    #     # Iterate through all elements to find the one containing the point
    #     for elem_idx, elem_nodes in enumerate(self.elems):
    #         # Get vertices of the tetrahedron
    #         vertices = self.node_xyz[elem_nodes]  # Shape: (4, 3)

    #         # Compute the matrix for barycentric coordinates
    #         bary_matrix = np.vstack([
    #             vertices.T,
    #             np.ones((1, 4))
    #         ])  # Shape: (4, 4)

    #         # Add the point to compute barycentric coordinates
    #         point_extended = np.append(point, 1)
            
    #         # Solve for barycentric coordinates
    #         try:
    #             bary_coords = np.linalg.solve(bary_matrix, point_extended)
    #         except np.linalg.LinAlgError:
    #             continue  # Skip elements with degenerate tetrahedra

    #         # Check if all barycentric coordinates are between 0 and 1 (inclusive)
    #         if np.all(bary_coords >= 0) and np.all(bary_coords <= 1):
    #             # Compute shape function values (they are equal to barycentric coordinates for tet4)
    #             shape_functions = bary_coords
    #             return elem_idx, shape_functions

    #     return -1, None  # No containing element foun
    
    def get_element_containing_point(self, point: np.ndarray) -> tuple: 
        """Efficiently find the element containing a point in a tetrahedral mesh and compute its shape functions.

        Args:
            point: Array of shape (3,) containing x, y, z coordinates.

        Returns:
            tuple: (Index of the element containing the point, Shape function values at the point),
                or (-1, None) if no element is found.
        """
        # Use a bounding box filter to reduce candidate elements
        point = np.asarray(point)
        bbox_min = np.min(self.node_xyz[self.elems], axis=1)
        bbox_max = np.max(self.node_xyz[self.elems], axis=1)
        in_bbox = np.all((point >= bbox_min) & (point <= bbox_max), axis=1)
        candidate_indices = np.where(in_bbox)[0]

        for elem_idx in candidate_indices:
            elem_nodes = self.elems[elem_idx]
            vertices = self.node_xyz[elem_nodes]  # Shape: (4, 3)
            # Compute barycentric coordinates
            T = np.hstack((vertices[1:] - vertices[0], np.eye(3)))
            try:
                # Solve for barycentric coordinates (lambda1, lambda2, lambda3)
                lambdas = np.linalg.solve((vertices[1:] - vertices[0]).T, (point - vertices[0]))
                bary_coords = np.empty(4)
                bary_coords[0] = 1 - np.sum(lambdas)
                bary_coords[1:] = lambdas
            except np.linalg.LinAlgError:
                continue  # Skip degenerate tets
            if np.all(bary_coords >= -1e-10) and np.all(bary_coords <= 1 + 1e-10):
                return elem_idx, bary_coords

        return -1, None  # No containing element found
    

    def get_surface_triangles_within_annular_region(self, centerPt, axis, innerRadius, outerRadius):
        # Get the surface triangles within the annular region defined by the center point, axis, and radii
        axis = axis / np.linalg.norm(axis)
        axis = axis.reshape(1, 3)
        nodes = self.node_xyz[:, :3] - centerPt
        proj = np.dot(nodes, axis.T)
        nodes = nodes - proj * axis
        dist = np.linalg.norm(nodes, axis=1)
        
        # Find surface triangles where all three nodes are within the annular region
        surface_tri_indices = []
        for i, tri in enumerate(self.surface_triangles):
            if all((dist[node] >= innerRadius) & (dist[node] <= outerRadius) for node in tri):
                surface_tri_indices.append(i)
        
        return surface_tri_indices
    
    def get_surface_triangles_on_bounding_box(self,  axis_dir = 0, min_plane = True):
        
        nodes_xyz = self.node_xyz
        
        if (min_plane):
            ref_val = np.min(nodes_xyz[:,axis_dir])
        else:
            ref_val = np.max(nodes_xyz[:,axis_dir])

        surface_tri_indices = []
        for i, tri in enumerate(self.surface_triangles):
            if all(abs(nodes_xyz[node,axis_dir]-ref_val)<1e-10 for node in tri):
                surface_tri_indices.append(i)
        return surface_tri_indices
    
    def get_surface_triangles_with_all_nodes_in_node_set(self, node_set):
        # Find surface triangles where all three nodes are in given node_set
        surface_tri_indices = []
        for i, tri in enumerate(self.surface_triangles):
            if all(node in node_set for node in tri):
                surface_tri_indices.append(i)
        return surface_tri_indices

    def plot(self, title = 'Tet Mesh'):
        plotter = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.node_xyz)
        plotter.plot(show_edges=True, show_scalar_bar=False, show_grid=True)
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")   

    def plotField(self, field, show_edges =  True, show_scalar_bar = True, show_grid = False):
        plotter = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.node_xyz)
        plotter.point_data["field"] = field
        # Some common alternatives:
        plotter.plot(show_edges=show_edges, show_scalar_bar=show_scalar_bar, show_grid=show_grid, cmap="jet",
                       scalar_bar_args={ 
                      'title': '',
                      'vertical': True,
                      'position_x': 0.8,
                      'position_y': 0.3,
                      'width': 0.1
                      })      # Classic rainbow colormap
        
        
if __name__ == "__main__":
    import os

    def cube_size_function_constant(pt):
        # Example size function that returns a constant size
        return 0.1
    
    def cube_size_function_linear(pt):
        # Example size function that returns a constant size
        z_min = 0.0
        z_max = 1.0
        z = pt[2]
        size = 0.1 + (0.02 - 0.1) * (z - z_min) / (z_max - z_min)
        size = np.clip(size, 0.02, 0.1)
        return size

    
    tetmesh = TetMesher()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/Cube/Cube.STL')
    tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=20000,mergeFacets=False, elemSizeFunction=cube_size_function_linear)
    tetmesh.plot()
   

