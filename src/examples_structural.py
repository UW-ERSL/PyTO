
import mesher
import numpy as np 
import bound_cond
import mat_lib
import os
import yaml 
import voxelizer as vx
from tkinter import filedialog

# Load settings from YAML file
script_dir = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(script_dir, 'settings.yaml')
with open(settings_path, 'r') as file:
  settings = yaml.safe_load(file)

cfg_mat = settings['MATERIAL']
cfg_opt = settings['OPTIMIZATION']
cfg_defl = settings['DEFLATION']


def createCantileverProblem(nDOFDesired: int = 10000, L: float = [1.0, 1.0, 1.0]):
	# This is an example where a grid mesh is created, and a structural problem is posed on it.
	# For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
	nVoxelsDesired = nDOFDesired/3	
	# Let the number of voxels be proportional to the length in each direction
	alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
	nelx = round(alpha*L[0])
	nely = round(alpha*L[1])
	nelz = round(alpha*L[2])
	print(f'nelx: {nelx}, nely: {nely}, nelz: {nelz}')
	mesh = mesher.Mesher()
	mesh.grid_mesh(num_elems = (nelx, nely, nelz),
								 elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))


	node_array = mesh.node_array

	fixed_nodes = np.where(node_array[:, 0] == 0)[0] # x = 0 plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
	dirichlet_values[0:3] = 0

	mesh.node_array[fixed_nodes, 3] = 1

	# line defined by x = xMax, and z = 0
	load_nodes = np.where((node_array[:, 0] == mesh.grid[0]) & 
												(node_array[:, 2] == 0) )[0]	 
	load_dofs = 3 * load_nodes + 2  # z direction

	mesh.node_array[load_nodes, 3] = 2
	load_per_dof = -1.e5/(nelz+1)

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = load_per_dof

	bc = bound_cond.BC(force = force,
						fixed_dofs = fixed_dofs,
						dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=cfg_mat['youngs_modulus'],
											poissons_ratio=cfg_mat['poissons_ratio'])
	return mesh, mat_prop, bc

# %%
def createLBracketProblem(nDOFDesired: int = 10000):
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
	stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
	nElemsDesired = nDOFDesired/3	# estimate
	mesh = mesher.Mesher()
	
	mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
	node_array = mesh.node_array
	node_pts = node_array[:, :3]*mesh.elem_size +mesh.origin
	
	fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
	dirichlet_values[0:3] = 0
	mesh.node_array[fixed_nodes, 3] = 1 # for plotting

	load_nodes = np.where((node_pts[:, 1] > 0.039) & (node_pts[:, 0] > 0.09))[0] # hard coded	
	load_dofs = 3 * load_nodes + 1  # z direction
	mesh.node_array[load_nodes, 3] = 2 # for plotting
	totalLoad = 1000

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = -totalLoad/len(load_nodes)

	bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=2.1e5,
										poissons_ratio=0.3)
	return mesh, mat_prop, bc
        

# %%
def createAlcoaProblem():
	# This is an example where an existing mesh is read, and a structural problem is posed on it.
	mesh = mesher.Mesher()
	mesh.read_pareto_mesh("../meshFiles/AlcoaGrabCAD.msh")
	node_array = mesh.node_array

	fixed_nodes = np.where(node_array[:, 3] == 1)[0]
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							 3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

	load_nodes = np.where(node_array[:, 3] == 2)[0]
	load_dofs = 3 * load_nodes + 1  # y direction
	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = -1000.

	bc = bound_cond.BC(force = force,
						fixed_dofs = fixed_dofs,
						dirichlet_values = dirichlet_values)

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=cfg_mat['youngs_modulus'],
										poissons_ratio=cfg_mat['poissons_ratio'])
	return mesh, mat_prop, bc

# %%
def createFilletedBeamProblem(nElemsDesired=50000):
	stl_file = os.path.join(script_dir, '../TOExamples/FilletedBeam/FilletedBeam.STL')

	mesh = mesher.Mesher()
	mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
	node_array = mesh.node_array
	fixed_nodes = np.where(node_array[:, 0] == 0)[0] # x = 0 plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
	dirichlet_values[0:3] = 0

	mesh.node_array[fixed_nodes, 3] = 1 # for plotting

	# line defined by x = xMax, and z = 0
	load_nodes = np.where(node_array[:, 0] == mesh.grid[0])[0] # x = xMax plane	 

	load_dofs = 3 * load_nodes + 2  # z direction

	mesh.node_array[load_nodes, 3] = 2 # for plotting
	load_per_dof = -1000

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = load_per_dof

	bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=2.1e5,
										poissons_ratio=0.3)
	return mesh, mat_prop, bc
        
	