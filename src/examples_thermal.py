
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

def createEdofMatThermal(mesh):
	edofMat = np.zeros((mesh.num_elems, 8), dtype = int)
	elemArray= mesh.elemArray
	for el in range(mesh.num_elems):
		edofMat[el, :] = np.array([
			elemArray[el][0], 
			elemArray[el][1],
			elemArray[el][2], 
			elemArray[el][3], 
			elemArray[el][4],
			elemArray[el][5],
			elemArray[el][6],
			elemArray[el][7]]
		)
	return edofMat

def createLBracketThermalProblem(nDOFDesired: int = 10000):
    """Creates a thermal problem setup for an L-bracket topology optimization.
    This function sets up a finite element mesh and boundary conditions for an L-bracket
    thermal problem from an STL file. The mesh is created with approximately the desired
    number of degrees of freedom. The problem includes fixed temperature boundary conditions
    on the top surface and a heat load on a portion of the right surface.

    Args:
        nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                                    Defaults to 10000.
    Returns:
        tuple: A tuple containing:
            - mesh (Mesher): Mesh object with the L-bracket discretization
            - mat_prop (ThermalMaterial): Material properties object with thermal parameters
            - bc (BC): Boundary conditions object with heat loads and temperature constraints

    Notes:
        - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
        - Fixed temperature boundary condition (T=0) applied at y = yMax
        - Heat load is applied where y > 0.039 and x > 0.09
        - Total heat load is 1000 W distributed equally among loaded nodes
        - Material properties are set to k = 45 W/mK
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
    nElemsDesired = nDOFDesired/3	# estimate
    mesh = mesher.Mesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.dofs_per_node = 1
    mesh.edofMat = createEdofMatThermal(mesh)

    node_pts = mesh.node_indices[:, :3]*mesh.elem_size +mesh.origin
	
    fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 23*np.ones_like(fixed_dofs, dtype = float)
  
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    load_nodes = np.where((node_pts[:, 1] > 0.039) & (node_pts[:, 0] > 0.09))[0] # hard coded	
    load_dofs = load_nodes
    mesh.node_indices[load_nodes, 3] = 2 # for plotting
    totalHeat= 1000

    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = totalHeat/len(load_nodes)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial( thermal_conductivity=50)
    return mesh, mat_prop, bc
    
if __name__ == "__main__":
    import plots
    import thermal_fea as fea
    import linear_solvers as lin_solv
    import jax # import jax to enable 64 bit precision
    import time	
    jax.config.update("jax_enable_x64", True)

    mesh, mat_prop, bc = createLBracketThermalProblem(nDOFDesired=10000)
    fe_solver = fea.ThermalFEA(mesh = mesh,
                          mat_prop = mat_prop,
                          bc = bc,
                          solver = lin_solv.Solvers.PARDISO)

    thermal_conductivity = np.ones((fe_solver.mesh.num_elems,)) * fe_solver.mat_prop.thermal_conductivity
    startTime = time.time()
    u = np.asarray(fe_solver.solve(elem_conductivity= thermal_conductivity))
    
    uMax = np.max(np.abs(u))
    nDOF = fe_solver.mesh.num_nodes
    print("nDof: ", nDOF)
    print('-----------------------------')
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max displaucement: ', uMax)
    print('-----------------------------')
	
    plots.plotMesh(fe_solver.mesh, fe_solver.bc, title=f'Cantilever; dof = {nDOF}')
    plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
								title=f'Max u: {uMax:.3e}')
