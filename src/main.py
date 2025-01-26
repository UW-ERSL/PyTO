# %%
import itertools
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

import bound_cond
import plots
import examples_structural as examplesStructural

jax.config.update("jax_enable_x64", True)
dsolver = deflation.DeflationSolver()
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
# Load fea and topopt routines
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
		plots.plotMesh(fe_solver.mesh, fe_solver.bc, title=f'Cantilever; dof = {nDOF}')
		plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
									title=f'Max deformation: {deltaMax:.3e}')

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
		u, history = topopt.topopt_optimality_criteria(fe_solver = fe_solver,
														maxIterations= cfg_opt['num_iter'],
														volfrac = volfrac
														)
		J = fe_solver.bc.force.T @ u
		title = f'OC: vol: {volfrac}, J: {J:.2e}'

	elif optimizationMethod == topopt.Optimizers.PARETO:
		u, history = topopt.topopt_pareto(fe_solver = fe_solver,
										desiredVolFrac =  volfrac)
		J = fe_solver.bc.force.T @ u
		title = f'Pareto: vol: {volfrac}, J: {J:.3e}'

	plots.plotMesh(fe_solver.mesh, fe_solver.bc, u, title = title)
	plots.plotIsocontour(fe_solver.mesh, u, title = title)

	for key in history:
		plt.plot(history[key], label=key)
		plt.xlabel('iter')
		plt.ylabel(key)
		plt.show()

def compareSolvers(linearSolvers = ['spsolve','pyamg','pardiso','dpcg'],
								dofs = [1000,5000,10000,50000,100000,250000,500000,10**6,2*10**6,3*10**6,5*10**6]):
	
	dofList = []
	solverTime = dict(zip(linearSolvers, [None]*len(linearSolvers)))
	for linearSolver in linearSolvers:
		solverTime[linearSolver] = []

	timeLimit = 60 # seconds	
	example = 1
	for dofDesired in dofs:
		print('-----------------------------')
		print("dofDesired: ",dofDesired)
		if (example == 1):
			mesh, mat_prop, bc = examplesStructural.createCantileverProblem(nDOFDesired=dofDesired,L=[2, 1, 1])
			title = 'Cantilever: Time for single FEA'
		elif (example == 2):	
			mesh, mat_prop, bc = examplesStructural.createLBracketProblem(nDOFDesired=dofDesired)
			title = 'LBracket: Time for single FEA'
		else:
			mesh, mat_prop, bc = examplesStructural.createFilletedBeamProblem(nDOFDesired=dofDesired)
			title = 'FilletedBeam: Time for single FEA'	
		
		
		dofActual = 3*mesh.num_nodes
		print("dofActual: ",dofActual)
		dofList.append(dofActual)
		
		for linearSolver in linearSolvers:
			# assuming increasing time with increasing DOF, skip if previous time was too long
			if len(solverTime[linearSolver]) > 0 and solverTime[linearSolver][-1] > timeLimit: 
				print('Solver: ', linearSolver, ' -')
				continue
			if (linearSolver == 'pardiso')  and (dofActual > 500000): # skip Pardiso for large problems, eats up memory, stalls computer
				print('Solver: ', linearSolver, ' -')
				continue
			startTime = time.time()
			if (linearSolver == 'spsolve'):
				solver = lin_solv.Solvers.SPSOLVE
			elif (linearSolver == 'pyamg'):
				solver = lin_solv.Solvers.PYAMG
			elif (linearSolver == 'pardiso'):
				solver = lin_solv.Solvers.PARDISO
			elif (linearSolver == 'dpcg'):
				solver = lin_solv.Solvers.DPCG
				nGroups =  min(2000,max(5,round(3*mesh.num_nodes/500)));
				dsolver.create_deflation_groups(mesh, nGroups)
				dsolver.create_delfation_matrix(mesh)
				dsolver.W = dsolver.W[bc.free_dofs, :]

			fe_solver = fea.StructFEA(mesh = mesh,
									mat_prop = mat_prop,
									bc = bc,
									solver = solver,
									dsolver = dsolver,
									rtol = 1e-8,
									verbose = False)
			run_fea(fe_solver = fe_solver, plot = False, verbose = False)
			totalTime = time.time() - startTime
			print('Solver: ', linearSolver, ' time: {:.2f}'.format(totalTime))
			solverTime[linearSolver].append(totalTime)
		
		
	
	marker = itertools.cycle(('dk', '+b', 'og', '*r','xm')) 
	colors = itertools.cycle(('k', 'b', 'g', 'r','m')) 
	for linearSolver in linearSolvers:
		timing = solverTime[linearSolver]
		plt.loglog(dofList[0:len(timing)],timing,next(marker))
	plt.legend(linearSolvers,loc = 'upper left')
	
	for linearSolver in linearSolvers:
		timing = solverTime[linearSolver]
		plt.loglog(dofList[0:len(timing)],timing,next(colors))

	plt.title(title)
	plt.xlabel('DOF')
	plt.ylabel('Time (secs)')
	plt.grid(True)
	plt.show()

compareSolvers()



exit()

# %%
mesh, mat_prop, bc = examplesStructural.createCantileverProblem(nDOFDesired=20000,L=[2, 1, 1])	
#mesh, mat_prop, bc = examplesStructural.createLBracketProblem(nDOFDesired=20000)	
#mesh, mat_prop, bc = examplesStructural.createFilletedBeamProblem(nDOFDesired=20000)
# %%
num_deflation_groups =  cfg_defl['num_groups']
dsolver = deflation.DeflationSolver()
dsolver.create_deflation_groups(mesh, cfg_defl['num_groups'])
dsolver.create_delfation_matrix(mesh)
dsolver.W = dsolver.W[bc.free_dofs, :]

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
#run_topopt(fe_solver, volfrac=0.5, optimizationMethod = topopt.Optimizers.MMA)

