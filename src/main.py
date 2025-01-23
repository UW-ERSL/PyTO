# %%
import yaml # pip install pyyaml
import time

import jax # pip install jax jaxlib
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib

import mat_lib
import struct_fea as fea
import deflation
import linear_solvers as lin_solv
import topopt as topopt
import os
import mesher
import bound_cond
import plots

jax.config.update("jax_enable_x64", True)

# %%
# Load settings from YAML file


script_dir = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(script_dir, 'settings.yaml')
with open(settings_path, 'r') as file:
  settings = yaml.safe_load(file)

cfg_mat = settings['MATERIAL']
cfg_opt = settings['OPTIMIZATION']
cfg_defl = settings['DEFLATION']

# %%
cfg_plot = settings['PLOT']
plt.rcParams.update(cfg_plot)

# %%
def createCantileverProblem(nelz: int = 10):
	mesh = mesher.Mesher()
	mesh.grid_mesh(num_elems = (2*nelz, nelz, nelz),
								 elem_size = (1.0/nelz, 1.0/nelz, 1.0/nelz))

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
def createLBracketProblem():
	mesh = mesher.Mesher()
	mesh.read_pareto_mesh("../meshFiles/LBracket.msh")
	node_array = mesh.node_array
	fixed_nodes = np.where(node_array[:, 3] == 1)[0]	 # use node label 1
	fixed_dofs = np.array([3 * fixed_nodes,
															 3 * fixed_nodes + 1,
															 3 * fixed_nodes + 2]).flatten().astype(int)
	dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

	load_nodes = np.where(node_array[:, 3] == 2)[0]
	load_dofs = 3 * load_nodes + 1  # y direction

	force = np.zeros(3*mesh.num_nodes)
	force[load_dofs] = -10.

	bc = bound_cond.BC(force = force,
										fixed_dofs = fixed_dofs,
										dirichlet_values = dirichlet_values)

	mat_prop = mat_lib.StructuralMaterial(youngs_modulus=cfg_mat['youngs_modulus'],
																				poissons_ratio=cfg_mat['poissons_ratio'])
	return mesh, mat_prop, bc

# %%
def createAlcoaProblem():
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
def run_fea(fe_solver: fea.StructFEA,
						plot: bool = True,
						verbose: bool = True):

	nDOF = 3*fe_solver.mesh.num_nodes

	startTime = time.time()
	youngs_modulus = np.ones((fe_solver.mesh.num_elems,)) * fe_solver.mat_prop.youngs_modulus
	u = np.asarray(fe_solver.solve(elem_youngs_modulus= youngs_modulus))
	delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
	deltaMax = np.max(delta)

	if verbose:
		print("nDof: ", nDOF)
		print('-----------------------------')
		print('Solver: ', fe_solver.solver.name)
		print("FEA time: ", time.time() - startTime)
		print('Max displacement: ', deltaMax)
		print('-----------------------------')
	if plot:
		plots.plotMesh(fe_solver.mesh, title=f'Cantilever; dof = {nDOF}')
		plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
									title=f'Max deformation: {deltaMax:.3e}')

# %%
def run_topopt(fe_solver: fea.StructFEA,
							volfrac: float,
							optimizationMethod = topopt.Optimizers):

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("optimizationMethod: ", optimizationMethod.name)

	if optimizationMethod == topopt.Optimizers.MMA:
		u, history = topopt.topopt_mma(fe_solver = fe_solver,
																	maxMMAIterations = cfg_opt['num_iter'],
																	volfrac = volfrac
																	)
		J = fe_solver.bc.force.T @ u
		title = f'MMA: vol: {volfrac}, J: {J:.2e}'

	elif optimizationMethod == topopt.Optimizers.OC:
		u, history = topopt.topopt_optimality_criteria(
																					fe_solver = fe_solver,
																					maxIterations= cfg_opt['num_iter'],
																					volfrac = volfrac
																					)
		J = fe_solver.bc.force.T @ u
		title = f'OC: vol: {volfrac}, J: {J:.2e}'

	elif optimizationMethod == topopt.Optimizers.PARETO:
		u, history = topopt.topopt_pareto(fe_solver = fe_solver,
												 							desiredVolFrac =  volfrac
												 							)
		J = fe_solver.bc.force.T @ u
		title = f'Pareto: vol: {volfrac}, J: {J:.3e}'

	plots.plotMesh(fe_solver.mesh, fe_solver.bc, u, title = title)
	plots.plotIsocontour(fe_solver.mesh, u, title = title)

	for key in history:
		plt.plot(history[key], label=key)
		plt.xlabel('iter')
		plt.ylabel(key)
		plt.show()

# %%
nelz = 15
mesh, mat_prop, bc = createCantileverProblem(nelz)

# %%
num_deflation_groups =  cfg_defl['num_groups']
dsolver = deflation.DeflationSolver()
dsolver.create_deflation_groups(mesh, cfg_defl['num_groups'])
dsolver.create_delfation_matrix(mesh)
dsolver.W = dsolver.W[bc.free_dofs, :]

# %%
fe_solver = fea.StructFEA(mesh = mesh,
                          mat_prop = mat_prop,
                          bc = bc,
                          solver = lin_solv.Solvers.DPCG,
                          dsolver = dsolver,
                          rtol = 1e-6,
                          verbose = False)

# %%
run_fea(fe_solver = fe_solver, plot = True)

# %%
#run_topopt(fe_solver, volfrac=0.5, optimizationMethod = topopt.Optimizers.PARETO)

