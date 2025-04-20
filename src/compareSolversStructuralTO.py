# %%
import time
import jax # pip install jax jaxlib
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import struct_fea as fea
import deflation
import linear_solvers as lin_solv
import os
from topopt_density_mma import topopt_mma
from topopt_density_oc import topopt_optimality_criteria	
from topopt_pareto import topopt_pareto
from examples_structural import *
from topopt_common import *
from topopt_benchmarks import *
import itertools

jax.config.update("jax_enable_x64", True)
dsolver = deflation.DeflationSolver()

script_dir = os.path.dirname(os.path.abspath(__file__))
to_params = TOParams()
kwargs=	{}
structural_problem = StructuralExamples.DistributedLoad
kwargs['topload'] = 1.5e4
kwargs['midload'] = 0
to_params.ExtrudeZ = True
to_params.nDOFDesired = 25000
to_params.DesiredVolFraction = 0.5


linearSolvers = [lin_solv.Solvers.CG, lin_solv.Solvers.PARDISO, lin_solv.Solvers.DPCG]

optimizationMethod = TO_METHODS.PARETO
dofs = [1000,5000,10000,25000,50000,100000,250000,500000,1e6,1.5e6,2e6,3e6]
# Set the time limit
timeLimit = 60*5 # seconds
dofList = []
solverTime = dict(zip(linearSolvers, [None]*len(linearSolvers)))
for linearSolver in linearSolvers:
	solverTime[linearSolver] = []

continueMeshing = True # set to false to skip to solving the FEA problems
dsolver = deflation.DeflationSolver()
title = f"{structural_problem.name} - {optimizationMethod.name}"
print_progress = False
for dofDesired in dofs:
	print("    ")
	if (not continueMeshing):
		break
	print('**************************')
	print("dofDesired: ", dofDesired)
	mesh, mat_prop, bc,elem_body_force = getStructuralProblem(structural_problem,nDOFDesired = dofDesired)
	dofActual = 3 * mesh.num_nodes
	print("dofActual: ",dofActual)
	dofList.append(dofActual)
	continueMeshing = False
	
	for linearSolver in linearSolvers:
		print('-----------------------')
		print(linearSolver)
		# assuming increasing time with increasing DOF, skip if previous time was too long
		if len(solverTime[linearSolver]) > 0 and solverTime[linearSolver][-1] > timeLimit: 
			print( ' ... skipping')
			continue
		continueMeshing = True
		startTime = time.time()
		if (linearSolver == lin_solv.Solvers.DPCG):
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_delfation_matrix(mesh)
			dsolver.W = dsolver.W[bc.free_dofs, :]

		fe_solver = fea.StructFEA(mesh = mesh,
								mat_prop = mat_prop,
								bc = bc,
								solver = linearSolver,
								dsolver = dsolver,
								rtol = 1e-8,
								verbose = False)
		nDOF = 3*fe_solver.mesh.num_nodes
		if optimizationMethod == TO_METHODS.DENSITYMMA:
			topopt_mma(fe_solver = fe_solver,to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.DENSITYOC:
			topopt_optimality_criteria(fe_solver = fe_solver,
										to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.PARETO:
			topopt_pareto(fe_solver = fe_solver,to_params = to_params,print_progress = print_progress)

		totalTime = time.time() - startTime
		
		print('Time: {:.2f}'.format(totalTime))
		solverTime[linearSolver].append(totalTime)
		
marker = itertools.cycle(('dk', '+b','xm', '*g', 'or')) 
colors = itertools.cycle(('k', 'b','m', 'g', 'r')) 
for linearSolver in linearSolvers:
	timing = solverTime[linearSolver]
	plt.loglog(dofList[0:len(timing)],timing,next(marker))
plt.legend(linearSolvers,loc = 'upper left')

for linearSolver in linearSolvers:
	timing = solverTime[linearSolver]
	plt.loglog(dofList[0:len(timing)],timing,next(colors))

plt.axhline(y=timeLimit, color='black', linestyle=':', label='Time limit')
plt.title(title)
plt.xlabel('DOF')
plt.ylabel('Time (secs)')
plt.grid(True)
plt.show()

