import numpy as np
import pyvista as pv

# Read the file
fp_stl_folder = "../Models/"
fp_msh = fp_stl_folder + "EdgeCantilever/EdgeCantilever_mshForCADRecovery_50Elems.mesh"

with open(fp_msh, 'r') as file:
    lines = file.readlines()

# Parse nodes
node_lines = lines[1:133]
nodes = np.array([[float(x), float(y), float(z)] for _, x, y, z in (line.strip().split() for line in node_lines)])

# Parse hex elements
element_lines = lines[134:194]
hex_elements = [list(map(int, line.strip().split()[1:])) for line in element_lines]
hex_elements = np.array(hex_elements)

# Pseudo density
pseudo_density = np.array([float(val) for val in lines[197].strip().split()])

# Build PyVista grid
cells = np.hstack([[8] + list(elem) for elem in hex_elements])
offset = np.arange(0, len(cells), 9)
cell_types = np.full(len(hex_elements), pv.CellType.HEXAHEDRON)
# VTK-compatible cells format
cells_vtk = np.hstack([[8] + list(cell) for cell in hex_elements])

# Number of cells
n_cells = len(hex_elements)

# Build the UnstructuredGrid
grid = pv.UnstructuredGrid(cells_vtk, cell_types, nodes)
grid.cell_data["PseudoDensity"] = pseudo_density[:n_cells]

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="PseudoDensity", cmap="viridis", show_edges=True)
plotter.show()

