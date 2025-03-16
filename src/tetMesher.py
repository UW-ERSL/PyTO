
import numpy as np
import pyvista as pv # pip install pyvista
from scipy.sparse import coo_matrix
import tetgen #pip install tetgen
import jax.numpy as jnp

class TetMesher:
    def __init__(self):	
        self.num_nodes = 0
        self.num_elems = 0
    
    def createTetMeshFromSTLFile(self, stlFileName: str, elemSizeDesired):
        self.stlMesh = pv.read(stlFileName)
        # Clean and repair the STL surface
        surf = self.stlMesh.clean()

        # Calculate approximate volume constraint based on desired number of elements
        total_volume = surf.volume
        print(f"Total volume: {total_volume}")
        tetVolume = (elemSizeDesired**3) / 6
        nElemsDesired = total_volume/tetVolume
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

 

    def createEdofMatThermal(self):
        self.edofMat = np.array(self.elems[:, :4], dtype=int)
    
    def get_nodes_from_locations(self, locations):
        distances = np.linalg.norm(self.nodes[:, :3] - locations[:, None], axis=2)
        return np.argmin(distances, axis=1)
    
    def plot(self):
        tet_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.nodes)
        tet_mesh.plot(show_edges=True, show_scalar_bar=False, show_grid=True)
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")   

    def plotField(self, field, show_edges =  True, show_scalar_bar = True, show_grid = True):
        tet_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.nodes)
        tet_mesh.point_data["field"] = field
        tet_mesh.plot(show_edges=show_edges, show_scalar_bar=True, show_grid=True)

if __name__ == "__main__":
    import os
    import time
    tetmesh = TetMesher()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
    tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=50000)
    tetmesh.plot()
