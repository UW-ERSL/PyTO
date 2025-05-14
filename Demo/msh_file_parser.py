import numpy as np
import pyvista as pv

# File path
fp_stl_folder = "../Models/"
fp_msh = fp_stl_folder + "EdgeCantilever/EdgeCantilever_mshForCADRecovery_50Elems.mesh"

# Read all lines
with open(fp_msh, 'r') as file:
    lines = file.readlines()

# 1. Get number of nodes from first line
num_nodes = int(lines[0].strip())
node_lines = lines[1:1 + num_nodes]

# Parse nodes
nodes = np.array([
    list(map(float, line.strip().split()[1:]))  # Skip the node ID
    for line in node_lines
])

# 2. Find section indices
section_map = {line.strip(): i for i, line in enumerate(lines) if line.strip().startswith("#")}

# Extract relevant sections
pseudo_density_line_idx = section_map.get("#PseudoDensity", None)
elem_conn_start_idx = num_nodes + 2  # Comes just after node count and node lines and count of elements
elem_conn_end_idx = section_map.get("#NodeLabel", pseudo_density_line_idx)  # Until next section or PseudoDensity

# Parse elements
element_lines = lines[elem_conn_start_idx:elem_conn_end_idx]
hex_elements = [list(map(int, line.strip().split()[1:])) for line in element_lines]
hex_elements = np.array(hex_elements)

# Parse pseudo density
pseudo_density = np.array([float(x) for x in lines[pseudo_density_line_idx + 1].strip().split()])

# 3. Construct PyVista UnstructuredGrid
cells_vtk = np.hstack([[8] + list(elem) for elem in hex_elements])
cell_types = np.full(len(hex_elements), pv.CellType.HEXAHEDRON)

grid = pv.UnstructuredGrid(cells_vtk, cell_types, nodes)
grid.cell_data["PseudoDensity"] = pseudo_density[:len(hex_elements)]

# 4. Plot
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="PseudoDensity", cmap="viridis", show_edges=True)
plotter.show()
