# Topopt demo

import sys
sys.path.append('../src') #assuming the pyTO src files is in the parent directory
from topopt_common import *
from topopt_benchmarks import *	
from topopt_pareto import topopt_pareto
jax.config.update("jax_enable_x64", True)


to_problem = StructuralTOExamples.CantileverTipLoad # Choose the TO problem
mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem) # fetch the structural problem

plots.plotMesh(mesh, bc) # plot the femesh and boundary conditions

fe_solver = fea.StructFEA(mesh = mesh,
				mat_prop = mat_prop,
                solver = lin_solv.Solvers.PARDISO,
				bc = bc) # create the finite element solver

# Call the topopt solver
u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver = fe_solver,to_params = to_params) # solve TO


# Plot volume vs compliance history
plt.figure()
plt.plot(history['volume'], history['compliance'], marker='o')
plt.xlabel('Volume Fraction')
plt.ylabel('Compliance')
plt.title('Pareto: Volume vs Compliance History')
plt.grid(True)
plt.show(block=False)

if not success:
    print(f"Error: {errorMsg}")
plots.plotMesh(fe_solver.mesh) # plot the final mesh

plots.plotIsocontour(fe_solver.mesh) # plot the isocontour of the final mesh