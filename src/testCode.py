import pyvista as pv
from tkinter import filedialog

stl_file = filedialog.askopenfilename(
        title="Select STL file",
        filetypes=[("STL files", "*.stl *.STL")]
    )
if not stl_file:
        print("No file selected")
        exit()
# Load STL model
mesh = pv.read(stl_file)

# Define a uniform voxel grid
voxels = pv.voxelize(mesh, density=0.02)
print(voxels)
p = pv.Plotter()
p.add_mesh(voxels, opacity=0.25)
p.show()