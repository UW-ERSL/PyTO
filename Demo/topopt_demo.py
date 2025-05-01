
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
	#from examples_topology_optimization_demo import *
	import matplotlib.pyplot as plt
	
	from hex_structural_fea import StructFEA
	from linear_solvers import Solvers

	jax.config.update("jax_enable_x64", True)
	#optimizationMethod = TO_METHODS.PARETO # DENSITYMMA, DENSITYOC, PARETO, LEVELSET
	
	# Choose the TO problem
	print("-" * 50)
	#to_problem = StructuralTOExamplesDemo.BasePlateAssembly
	to_problem = StructuralTOExamples.LBracketMidLoad
	title = f'{to_problem.name}'
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
	print(f"Running {to_problem.name}...")
	print("-" * 50)
	
	
	fe_solver = StructFEA(mesh=mesh, mat_prop=mat_prop, solver= Solvers.PARDISO, bc=bc)
	fe_solver.plot_mesh(title = title)
	u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver=fe_solver,to_params=to_params,plot_progress=False)
	plt.figure()
	plt.plot(history['volume'], history['compliance'], marker='o')
	plt.xlabel('Volume Fraction')
	plt.ylabel('Compliance')
	plt.title('Pareto: Volume vs Compliance History')
	plt.grid(True)
	plt.show()
	if not success:
		print(f"Error: {errorMsg}")
	fe_solver.plot_mesh(title = title)
	
	plots_demo.export_vtu_mesh(fe_solver.mesh, title = title)
	# Save the mesh and results
	# fp_original_stl = '../Models/EdgeCantilever/EdgeCantilever.STL'
	# fp_original_stl = '../Models/Rocket/HollowNoseConeWithSolidBaseNew.STL'
	# if fp_original_stl is not None and os.path.isfile(fp_original_stl):
	# 	print(f'{fp_original_stl} File exists.')
	# else:
	# 	print(f'{fp_original_stl} File does not exists.')
	# plots_demo.retainOuterGeom(fe_solver.mesh, fp_original_stl)
	#plots_demo.retainOuterGeomUsingIsoSurf(fe_solver.mesh, fp_original_stl, u, isovalue=0.5)