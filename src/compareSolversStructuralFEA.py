# %%
"""
This script compares the performance of different linear solvers for finite element analysis (FEA) problems.

Functions:
	createCantileverProblem: Creates a cantilever problem for FEA.
	createLBracketProblem: Creates an L-bracket problem for FEA.
	createCompliantMechanismProblem: Creates a compliant mechanism problem for FEA.
	createFilletedBeamProblem: Creates a filleted beam problem for FEA.
Global Variables:
	linearSolvers: List of linear solvers to be compared.
	dofs: List of degrees of freedom (DOF) for the problems.
	timeLimit: Time limit for each solver in seconds.
	dofList: List to store the actual DOF for each problem.
	solverTime: Dictionary to store the time taken by each solver.
	example: Integer to select the type of problem to solve.
	continueMeshing: Boolean to control the meshing process.
	dsolver: Instance of the DeflationSolver class.
Main Code:
	The script iterates over the list of DOFs and solves the selected FEA problem using each linear solver.
	It measures the time taken by each solver and stores the results.
	Finally, it plots the time taken by each solver against the DOF.
For Loop:
	The outer for loop iterates over the desired DOFs.
	The inner for loop iterates over the list of linear solvers.
	It checks if the previous solver time exceeded the time limit and skips if true.
	It creates the appropriate solver instance and solves the FEA problem.
	It measures and prints the time taken by each solver.
	It stores the time taken by each solver in the solverTime dictionary.
"""
import itertools
import time
import jax # pip install jax jaxlib
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import struct_fea as fea
import deflation
import linear_solvers as lin_solv
import topopt_common as topopt_common
import os
from examples_structural import *
jax.config.update("jax_enable_x64", True)


# Choose the linear solvers to compare
linearSolvers = ['spsolve','pyamg','pycg','pypardiso','pydpcg']
# Set the DOF for the problems to run through
dofs = [1000,5000,10000,25000,50000,100000,250000,500000,1e6,1.5e6,2e6,3e6]
# Set the time limit for each solver
timeLimit = 20 # seconds
dofList = []
solverTime = dict(zip(linearSolvers, [None]*len(linearSolvers)))
for linearSolver in linearSolvers:
	solverTime[linearSolver] = []

continueMeshing = True # set to false to skip to solving the FEA problems
dsolver = deflation.DeflationSolver()
problem = StructuralExamples.EdgeCantilever
for dofDesired in dofs:
	if (not continueMeshing):
		break
	print('-----------------------------')
	print("dofDesired: ", dofDesired)
	mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = dofDesired)
	dofActual = 3 * mesh.num_nodes
	print("dofActual: ",dofActual)
	dofList.append(dofActual)
	continueMeshing = False
	title = f"{problem.name}"
	for linearSolver in linearSolvers:
		# assuming increasing time with increasing DOF, skip if previous time was too long
		if len(solverTime[linearSolver]) > 0 and solverTime[linearSolver][-1] > timeLimit: 
			print('Solver: ', linearSolver, ' -')
			continue
		continueMeshing = True
		startTime = time.time()
		if (linearSolver == 'spsolve'):
			solver = lin_solv.Solvers.SPSOLVE
		elif (linearSolver == 'pyamg'):
			solver = lin_solv.Solvers.PYAMG
		elif (linearSolver == 'pypardiso'):
			solver = lin_solv.Solvers.PARDISO
		elif (linearSolver == 'pydpcg'):
			solver = lin_solv.Solvers.DPCG
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_delfation_matrix(mesh)
			dsolver.W = dsolver.W[bc.free_dofs, :]
		elif (linearSolver == 'pycg'):
			solver = lin_solv.Solvers.CG

		fe_solver = fea.StructFEA(mesh = mesh,
								mat_prop = mat_prop,
								bc = bc,
								solver = solver,
								dsolver = dsolver,
								rtol = 1e-8,
								verbose = False)
		nDOF = 3*fe_solver.mesh.num_nodes
		u = np.asarray(fe_solver.solve())
		totalTime = time.time() - startTime
		delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
		deltaMax = np.max(delta)
		nDOF = 3*fe_solver.mesh.num_nodes
		
		print('Solver: ', linearSolver, ' time: {:.2f}'.format(totalTime))
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


