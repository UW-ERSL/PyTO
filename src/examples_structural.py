
import mesher
import numpy as np 
import bound_cond
import mat_lib
import os
import yaml 

# Load settings from YAML file
script_dir = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(script_dir, 'settings.yaml')
with open(settings_path, 'r') as file:
  settings = yaml.safe_load(file)

cfg_mat = settings['MATERIAL']
cfg_opt = settings['OPTIMIZATION']
cfg_defl = settings['DEFLATION']

def createEdofMatStructural(mesh):
	edofMat = np.zeros((mesh.num_elems, 24), dtype = int)
	elemArray= mesh.elemArray
	for el in range(mesh.num_elems):
		edofMat[el, :] = np.array([
			3*elemArray[el][0], 3*elemArray[el][0]+1, 3*elemArray[el][0]+2,
			3*elemArray[el][1], 3*elemArray[el][1]+1, 3*elemArray[el][1]+2,
			3*elemArray[el][2], 3*elemArray[el][2]+1, 3*elemArray[el][2]+2,
			3*elemArray[el][3], 3*elemArray[el][3]+1, 3*elemArray[el][3]+2,
			3*elemArray[el][4], 3*elemArray[el][4]+1, 3*elemArray[el][4]+2,
			3*elemArray[el][5], 3*elemArray[el][5]+1, 3*elemArray[el][5]+2,
			3*elemArray[el][6], 3*elemArray[el][6]+1, 3*elemArray[el][6]+2,
			3*elemArray[el][7], 3*elemArray[el][7]+1, 3*elemArray[el][7]+2]
		)
	return edofMat

def createCantileverProblem(nDOFDesired: int = 10000, L: float = [0.1, 0.1, 0.1]):
	# This is an example where a grid mesh is created, and a structural problem is posed on it.
	# For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
	nVoxelsDesired = nDOFDesired/3	
	# Let the number of voxels be proportional to the length in each direction
	alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
	nelx = round(alpha*L[0])
	nely = round(alpha*L[1])
	nelz = round(alpha*L[2])
	mesh = mesher.Mesher()
	mesh.grid_mesh(num_elems = (nelx, nely, nelz),
								 elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)

	node_indices = mesh.node_indices

	fixed_nodes = np.where(node_indices[:, 0] == 0)[0] # x = 0 plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

	mesh.node_indices[fixed_nodes, 3] = 1

	# line defined by x = xMax, and z = 0
	load_nodes = np.where((node_indices[:, 0] == mesh.grid[0]) & 
												(node_indices[:, 2] == 0) )[0]	 
	load_dofs = 3 * load_nodes + 2  # z direction

	mesh.node_indices[load_nodes, 3] = 2
	load_per_dof = -10000/len(load_nodes)

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
	"""Creates a structural problem setup for an L-bracket topology optimization.
	This function sets up a finite element mesh and boundary conditions for an L-bracket
	structural problem from an STL file. The mesh is created with approximately the desired
	number of degrees of freedom. The problem includes fixed boundary conditions on the top
	surface and a distributed load on a portion of the right surface.
	Args:
		nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
									Defaults to 10000.
	Returns:
		tuple: A tuple containing:
			- mesh (Mesher): Mesh object with the L-bracket discretization
			- mat_prop (StructuralMaterial): Material properties object with structural parameters
			- bc (BC): Boundary conditions object with forces and constraints
	Notes:
		- The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
		- Fixed boundary conditions are applied at y = yMax
		- Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
		- Total applied load is 1000 units distributed equally among loaded nodes
		- Material properties are set to E = 2.1e5 and ν = 0.3
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
	stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
	nElemsDesired = nDOFDesired/3	# estimate
	mesh = mesher.Mesher()
	
	mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)

	node_pts = mesh.node_indices[:, :3]*mesh.elem_size +mesh.origin
	
	fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

	mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

	load_nodes = np.where((node_pts[:, 1] > 0.039) & (node_pts[:, 0] > 0.09))[0] # hard coded	
	load_dofs = 3 * load_nodes + 1  # z direction
	mesh.node_indices[load_nodes, 3] = 2 # for plotting
	totalLoad = 1000

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = -totalLoad/len(load_nodes)

	bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=2.1e11,
										poissons_ratio=0.3)
	return mesh, mat_prop, bc
        
def createCompliantMechanismProblem(nDOFDesired: int = 10000):
	"""Creates a structural problem setup for an Compliant Mechanism 
	This function sets up a finite element mesh and boundary conditions for an Compliant Mechanism
	structural problem from an STL file. The mesh is created with approximately the desired
	number of degrees of freedom. 
	Args:
		nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
									Defaults to 10000.
	Returns:
		tuple: A tuple containing:
			- mesh (Mesher): Mesh object with the L-bracket discretization
			- mat_prop (StructuralMaterial): Material properties object with structural parameters
			- bc (BC): Boundary conditions object with forces and constraints
	Notes:
		- The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
		- Fixed boundary conditions are applied at y = yMax
		- Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
		- Total applied load is 1000 units distributed equally among loaded nodes
		- Material properties are set to E = 2.1e5 and ν = 0.3
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
	stl_file = os.path.join(script_dir, '../TOExamples/CompliantMechanism/CompliantMechanism.STL')
	nElemsDesired = nDOFDesired/3	# estimate
	mesh = mesher.Mesher()
	
	mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)

	node_pts =mesh.node_indices[:, :3]*mesh.elem_size + mesh.origin
	
	fixed_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) > 20))[0] # the two end faces of the mechanism
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

	mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

	load_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) < 20))[0] # the middle face of the mechanism	
	load_dofs = 3 * load_nodes  
	mesh.node_indices[load_nodes, 3] = 2 # for plotting
	totalLoad = 1e6

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
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)
	node_indices = mesh.node_indices

	fixed_nodes = np.where(node_indices[:, 3] == 1)[0]
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							 3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

	load_nodes = np.where(node_indices[:, 3] == 2)[0]
	load_dofs = 3 * load_nodes + 1  # y direction
	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = -1000.

	bc = bound_cond.BC(force = force,
						fixed_dofs = fixed_dofs,
						dirichlet_values = dirichlet_values)

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=cfg_mat['youngs_modulus'],
										poissons_ratio=cfg_mat['poissons_ratio'])
	return mesh, mat_prop, bc

def createBeamSurfaceLoadProblem(nDOFDesired: int = 20000, L: float = [0.1, 0.01, 0.01]):
	# This is for large deformation
	nVoxelsDesired = int(nDOFDesired/5)
	print(nVoxelsDesired)

	# Let the number of voxels be proportional to the length in each direction
	alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
	nelx = round(alpha*L[0])+1
	nely = round(alpha*L[1])
	nelz = round(alpha*L[2])
	mesh = mesher.Mesher()
	mesh.grid_mesh(num_elems = (nelx, nely, nelz),
								 elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)
	
	node_indices = mesh.node_indices

	fixed_nodes = np.where(node_indices[:, 0] == 0)[0] # x = 0 plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

	mesh.node_indices[fixed_nodes, 3] = 1 # this is for plotting

	# line defined by x = xMax
	load_nodes = np.where(node_indices[:, 0] == mesh.grid[0] )[0]	 
	load_dofs = 3 * load_nodes + 2  # z direction

	mesh.node_indices[load_nodes, 3] = 2
	load_per_dof = -30000/len(load_nodes)

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = load_per_dof

	bc = bound_cond.BC(force = force,
						fixed_dofs = fixed_dofs,
						dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=3e7,
											poissons_ratio=0.3)
	return mesh, mat_prop, bc

# %%
def createFilletedBeamProblem(nDOFDesired=50000):
	stl_file = os.path.join(script_dir, '../TOExamples/FilletedBeam/FilletedBeam.STL')

	mesh = mesher.Mesher()
	nElemsDesired = round(nDOFDesired/3)	# estimate
	mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
	mesh.dofs_per_node = 3 # structural
	mesh.edofMat = createEdofMatStructural(mesh)

	node_indices = mesh.node_indices
	fixed_nodes = np.where(node_indices[:, 0] == 0)[0] # x = 0 plane
	fixed_dofs = np.array([3 * fixed_nodes,
							3 * fixed_nodes + 1,
							3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)


	mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

	# line defined by x = xMax, and z = 0
	load_nodes = np.where(node_indices[:, 0] == mesh.grid[0])[0] # x = xMax plane	 

	load_dofs = 3 * load_nodes + 2  # z direction

	mesh.node_indices[load_nodes, 3] = 2 # for plotting
	load_per_dof = -1000

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = load_per_dof

	bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=2.1e5,
										poissons_ratio=0.3)
	return mesh, mat_prop, bc
        
if __name__ == "__main__":
    import plots
    import struct_fea as fea
    import linear_solvers as lin_solv
    import jax # import jax to enable 64 bit precision
    import time	
    jax.config.update("jax_enable_x64", True)

    mesh, mat_prop, bc = createCantileverProblem(nDOFDesired=10000)
    fe_solver = fea.StructFEA(mesh = mesh,
                          mat_prop = mat_prop,
                          bc = bc,
                          solver = lin_solv.Solvers.PARDISO)

    youngs_modulus = np.ones((fe_solver.mesh.num_elems,)) * fe_solver.mat_prop.youngs_modulus
    startTime = time.time()
    u = np.asarray(fe_solver.solve(elem_youngs_modulus= youngs_modulus))
    delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
    deltaMax = np.max(delta)
    nDOF = 3*fe_solver.mesh.num_nodes
    print("nDof: ", nDOF)
    print('-----------------------------')
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max displacement: ', deltaMax)
    print('-----------------------------')
	
    plots.plotMesh(fe_solver.mesh, fe_solver.bc, title=f'Cantilever; dof = {nDOF}')
    plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
								title=f'Max deformation: {deltaMax:.3e}')
