
import numpy as np
import pyvista as pv # pip install pyvista
from scipy.sparse import coo_matrix
import tetgen #pip install tetgen
import jax.numpy as jnp

class TetMesher:
    def __init__(self):	
        self.num_nodes = 0
        self.num_elems = 0
    
    def createTetMeshFromSTLFile(self, stlFileName: str, nElemsDesired: int = 10000):
        self.stlMesh = pv.read(stlFileName)
        # Clean and repair the STL surface
        surf = self.stlMesh.clean()

        # Calculate approximate volume constraint based on desired number of elements
        total_volume = surf.volume
        print(f"Total volume: {total_volume}")
  

        # Add a scaling factor to get closer to desired element count
        max_tet_volume = (total_volume / nElemsDesired)  

        # Generate tetrahedral mesh with target number of cells and quality constraints
        tet = tetgen.TetGen(surf)
        nodes, elements = tet.tetrahedralize(switches=f"pq1.5a{max_tet_volume}Q")
        self.nodes = nodes
        self.elems = elements
        self.num_nodes = len(self.nodes)
        self.num_elems = len(self.elems)
        
        # Get all faces from tetrahedra
        faces = np.vstack([
            self.elems[:, [0, 1, 2]],
            self.elems[:, [0, 1, 3]],
            self.elems[:, [1, 2, 3]],
            self.elems[:, [0, 2, 3]]
        ])

        # Sort faces for comparison
        faces = np.sort(faces, axis=1)

        # Find unique faces and their counts
        _, idx, counts = np.unique(faces, axis=0, return_index=True, return_counts=True)

        # Surface triangles are faces that appear only once
        self.surface_triangles = faces[idx[counts == 1]]
        
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")
        print(f"Number of surface triangles: {len(self.surface_triangles)}")

        element_sizes = np.zeros(self.num_elems)
        for i in range(self.num_elems):
            element_sizes[i] = np.linalg.norm(self.nodes[self.elems[i, 0], :3] - self.nodes[self.elems[i, 1], :3])
        self.elem_size = np.mean(element_sizes)
        print(f"Element size: {self.elem_size}")


    def integrate_surface_force(self, q, tri_surface_indices):
        # Initialize force vector
        force_vector = np.zeros(self.num_nodes)
        surf_triangles = self.surface_triangles[tri_surface_indices, :]
        # Loop over each surface triangle
        total_area = 0.0
        for tri in surf_triangles:
            # Get the nodes of the triangle
            node_coords = self.nodes[tri, :]
            # Calculate the area of the triangle
            vec1 = node_coords[1] - node_coords[0]
            vec2 = node_coords[2] - node_coords[0]
            tri_area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            total_area += tri_area
            # Distribute the force over the triangle nodes
            for node in tri:
                force_vector[node] += q * tri_area / 3.0
        print(f"Total area of triangles: {total_area}")
        return force_vector/total_area
    
    def createEdofMatThermal(self):
        self.edofMat = np.array(self.elems[:, :4], dtype=int)
    
    def get_nodes_from_locations(self, locations):
        distances = np.linalg.norm(self.nodes[:, :3] - locations[:, None], axis=2)
        return np.argmin(distances, axis=1)
    
    def get_nodes_within_annular_region(self, centerPt, axis, innerRadius, outerRadius):
        # Get the nodes within the annular region defined by the center point, axis, and radii
        axis = axis / np.linalg.norm(axis)
        axis = axis.reshape(1, 3)
        nodes = self.nodes[:, :3] - centerPt
        proj = np.dot(nodes, axis.T)
        nodes = nodes - proj * axis
        dist = np.linalg.norm(nodes, axis=1)
        return np.where((dist >= innerRadius) & (dist <= outerRadius))[0]
    
    def get_surface_triangles_within_annular_region(self, centerPt, axis, innerRadius, outerRadius):
        # Get the surface triangles within the annular region defined by the center point, axis, and radii
        axis = axis / np.linalg.norm(axis)
        axis = axis.reshape(1, 3)
        nodes = self.nodes[:, :3] - centerPt
        proj = np.dot(nodes, axis.T)
        nodes = nodes - proj * axis
        dist = np.linalg.norm(nodes, axis=1)
        
        # Find surface triangles where all three nodes are within the annular region
        surface_tri_indices = []
        for i, tri in enumerate(self.surface_triangles):
            if all((dist[node] >= innerRadius) & (dist[node] <= outerRadius) for node in tri):
                surface_tri_indices.append(i)
        
        return surface_tri_indices
    def plot(self, title = 'Tet Mesh'):
        plotter = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.nodes)
        plotter.plot(show_edges=True, show_scalar_bar=False, show_grid=True)
        
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")   

    def plotField(self, field, show_edges =  True, show_scalar_bar = True, show_grid = False):
        plotter = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.nodes)
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
    tetmesh = TetMesher()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
    tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=50000)
    tetmesh.plot()
