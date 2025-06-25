from topopt_common import *
from topopt_structural_benchmarks import *
from hex_structural_examples import StructuralExamples,getStructuralProblem

import time



problem = StructuralExamples.BeamBending
nDOFDesired = 10000
mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)

solver = lin_solv.Solvers.DPCG # # Choose solver
dsolver = deflation.DeflationSolver()
nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
dsolver.create_deflation_groups(mesh, nGroups)
dsolver.create_deflation_matrix(mesh)
dsolver.plot_deflation_groups(mesh)
dsolver.W = dsolver.W[bc.free_dofs, :]

debug = False

fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
            mat_prop = mat_prop,
            bc = bc,
            solver = solver,
            dsolver = dsolver,
            rtol = 1e-8,
            elem_body_force = elem_body_force)

print('Solver: ', fe_solver.solver.name)
print("nNodes: ", fe_solver.mesh.num_nodes)
print("nElem: ", fe_solver.mesh.num_elems)	

title = f'nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
#fe_solver.plot_mesh(title = title, save_path = None)

startTime = time.time()
fe_solver.solve()
timeTaken = time.time() - startTime
print(f"Time taken to solve: {timeTaken:.2f} seconds")
fe_solver.postprocess()

fe_solver.plot_elem_field(fe_solver.elemStrainEnergy, title=title, save_path=None)
# mappingMatrix  = mesh.elem_to_node_field_mapping
# nodalStrainEnergy = mappingMatrix*fe_solver.elemStrainEnergy

dsolver.create_deflation_groups_manual(mesh,nGroups)
dsolver.plot_deflation_groups(mesh)
dsolver.create_deflation_matrix(mesh)
dsolver.W = dsolver.W[bc.free_dofs, :]

startTime = time.time()
fe_solver.solve()
timeTaken = time.time() - startTime
print(f"Time taken to solve: {timeTaken:.2f} seconds")