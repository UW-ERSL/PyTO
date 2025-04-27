
"""Optimization routines for topology optimization."""
import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
from topopt_common import *
from topopt_pareto import *
from topopt_levelset import *
from topopt_density_oc import *
from topopt_density_mma import *
import plots_demo	

import jax
	
if __name__ == "__main__":    
	from topopt_benchmarks import *
	from examples_topology_optimization_demo import *
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time
	import matplotlib.pyplot as plt
	import deflation
	import plots	
	
	jax.config.update("jax_enable_x64", True)
	optimizationMethod = TO_METHODS.PARETO # DENSITYMMA, DENSITYOC, PARETO, LEVELSET

	#runTOTests(); exit(0) # Run all tests for each example in the StructuralTOExamples enum
	
	# Choose the TO problem
	print("-" * 50)
	#to_problem = StructuralTOExamplesDemo.BasePlateAssembly
	to_problem = StructuralTOExamplesDemo.EdgeCantileverDemo
	print(f"Running {to_problem.name}...")
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	fe_solver = fea.StructFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = solver,
				dsolver = dsolver,
				rtol = 1e-8,
        		elem_body_force = elem_body_force)
	

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#mesh.plot()
	plots.plotMesh(mesh, bc,title = title)
	#plots.plotIsocontour(fe_solver.mesh, title = title, save_path = None)

	startTime = time.time()
	if optimizationMethod == TO_METHODS.DENSITYMMA:
		print("OptimizationMethod: MMA")
		u, history,success,errorMsg = topopt_mma(fe_solver = fe_solver,
						  			to_params = to_params,
									debug = debug)
		timeTaken = time.time() - startTime
		fig, ax1 = plt.subplots()

		# Plot compliance on left y-axis
		ax1.set_xlabel('Iterations')
		ax1.set_ylabel('Compliance', color='tab:blue')
		ax1.plot(history['compliance'], color='tab:blue', label='Compliance')
		ax1.tick_params(axis='y', labelcolor='tab:blue')

		# Plot volume fraction on right y-axis with dotted line
		ax2 = ax1.twinx()
		ax2.set_ylabel('Volume Fraction', color='tab:orange')
		ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
		ax2.tick_params(axis='y', labelcolor='tab:orange')
		ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

		plt.title('MMA: Volume and Compliance vs. Iterations')

		# Add legend
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2)

		plt.grid(True)
		plt.show(block=False)

		title = f"MMA: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

	elif optimizationMethod == TO_METHODS.DENSITYOC:
		print("OptimizationMethod: OC")
		u, history, success,errorMsg = topopt_optimality_criteria(fe_solver = fe_solver,
										  		to_params = to_params,
												debug = debug)
		timeTaken = time.time() - startTime
		title = f"OC: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

		fig, ax1 = plt.subplots()

		# Plot compliance on left y-axis
		ax1.set_xlabel('Iterations')
		ax1.set_ylabel('Compliance', color='tab:blue')
		ax1.plot(history['compliance'], color='tab:blue', label='Compliance')
		ax1.tick_params(axis='y', labelcolor='tab:blue')

		# Plot volume fraction on right y-axis with dotted line
		ax2 = ax1.twinx()
		ax2.set_ylabel('Volume Fraction', color='tab:orange')
		ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
		ax2.tick_params(axis='y', labelcolor='tab:orange')
		ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

		plt.title('OC: Volume and Compliance vs. Iterations')

		# Add legend
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2)

		plt.grid(True)
		plt.show(block=False)
	
	elif optimizationMethod == TO_METHODS.PARETO:
		print("OptimizationMethod: Pareto")
		u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver = fe_solver,
										to_params = to_params,
										debug = debug)
		
		timeTaken = time.time() - startTime
		title = f"Pareto: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
		
		# Plot volume vs compliance history
		plt.figure()
		plt.plot(history['volume'], history['compliance'], marker='o')
		plt.xlabel('Volume Fraction')
		plt.ylabel('Compliance')
		plt.title('Pareto: Volume vs Compliance History')
		plt.grid(True)
		plt.show(block=False)
	elif optimizationMethod == TO_METHODS.LEVELSET:
		print("OptimizationMethod: Level Set")
		u, history, success,errorMsg = topopt_levelset(fe_solver = fe_solver,
										to_params = to_params,
										maxIterations = 100,
										time_step = 0.1,
										epsilon = 1.0,
										rel_conv_tol = 1e-4,
										debug = debug)
		timeTaken = time.time() - startTime
		title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"	

	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")
	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title, show_edges = False)

	plots.plotIsocontour(fe_solver.mesh, title = title, save_path = None)
	# Save the mesh and results
	# fp_original_stl = '../Models/EdgeCantilever/EdgeCantilever.STL'
	# fp_original_stl = '../Models/Rocket/HollowNoseConeWithSolidBaseNew.STL'
	# if fp_original_stl is not None and os.path.isfile(fp_original_stl):
	# 	print(f'{fp_original_stl} File exists.')
	# else:
	# 	print(f'{fp_original_stl} File does not exists.')
	# plots_demo.retainOuterGeom(fe_solver.mesh, fp_original_stl)
	#plots_demo.retainOuterGeomUsingIsoSurf(fe_solver.mesh, fp_original_stl, u, isovalue=0.5)