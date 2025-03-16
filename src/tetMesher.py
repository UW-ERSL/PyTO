
import numpy as np
import pyvista as pv # pip install pyvista
from scipy.sparse import coo_matrix
import tetgen #pip install tetgen


class TetMesher:
    def __init__(self):	
        self.num_nodes = 0
        self.num_elems = 0
    
    def createTetMeshFromSTLFile(self, stlFileName: str, nElemsDesired=20000):
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
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")

    def plot(self):
        tet_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: self.elems}, self.nodes)
        tet_mesh.plot(show_edges=True, show_scalar_bar=False, show_grid=False)
        print(f"Number of nodes: {self.num_nodes}, Number of elements: {self.num_elems}")   


if __name__ == "__main__":
    import os
    import time
    tetmesh = TetMesher()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
    stlFileName = os.path.join(script_dir, '../Models/Overhang/Overhang.STL')
    #stlFileName = os.path.join(script_dir, '../Models/AlcoaGrabCAD/AlcoaGrabCAD.STL')
    #stlFileName = os.path.join(script_dir, '../Models/Knuckle/Knuckle.STL')
    #stlFileName = os.path.join(script_dir, '../Models/SwingArmAssembly/SwingArmAssembly.STL')
    tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=25000)
    tetmesh.plot()
