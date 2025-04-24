'''
This script is a demo for the pyTO package. 
It includes examples of how to use the package for various tasks, including loading STL files, 
creating voxel meshes, and performing finite element analysis (FEA) and topology optimization (TO).
'''


import sys
import jax
import enum
import time
import numpy as np  
import matplotlib.pyplot as plt

# PyTO files
sys.path.append('./src') #assumes PyTO src files are in this directory
from stl_reader import STLGeom
from hex_mesher import Mesher
from linear_solvers import Solvers
from deflation import DeflationSolver
from hex_structural_fea import StructFEA
from hex_modal_fea import ModalFEA
from thermal_fea import ThermalFEA
from hex_structural_examples import StructuralExamples, getStructuralProblem
from hex_thermal_examples import ThermalExamples, getThermalProblem
from topopt_benchmarks import StructuralTOExamples, getStructuralTOProblem
from topopt_density_mma import topopt_mma
from topopt_pareto import topopt_pareto
from topopt_density_oc import topopt_optimality_criteria
from tet_mesher import TetMesher
from tet_thermal_examples import  createAnnularPlateThermalProblemTet
from tet_thermal_fea import ThermalFEATet



class pyTODemos(enum.Enum):
	Load_STL = enum.auto() # Load an STL file and compute mass properties

    # The following demos use a non-conforming voxel mesh
	Create_Voxelmesh = enum.auto() # Create a voxel mesh from an STL file  
	ThermalFEA_Voxel_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver
	StructuralFEA_Voxel_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver
	StructuralFEA_Voxel_DPCG = enum.auto() # Create and solve a predefined structural FEA problem using the DPCG solver
	ModalFEA_Voxel_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver    
	StructuralTO_Voxel_DensityMMA = enum.auto() # Create and solve a structural topology optimization problem using Density-method and MMA solver
	StructuralTO_Voxel_DensityOC = enum.auto() # Create and solve a structural topology optimization problem using Density-method and OC solver
	StructuralTO_Voxel_Pareto = enum.auto() # Create and solve a structural topology optimization problem using Pareto method
    
    # The following demos use a conforming tet mesh
	Create_Tetmesh = enum.auto() # Create a tet mesh from an STL file 
	ThermalFEA_Tet_Pardiso = enum.auto() # Create and solve a thermal FEA problem using the PARDISO solver 


#Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

demo = pyTODemos.Load_STL  # Initialize with first demo
while True:
    print(50*'-')
    print(f"\nCurrent demo: {demo.name}")
    choice = input("Run this demo? (r=run, s=skip, q=quit): ").lower()
    
    if choice == 'q':
        break
    elif choice == 's':
        # Get next demo in enum
        demo_list = list(pyTODemos)
        current_index = demo_list.index(demo)
        if current_index < len(demo_list) - 1:
            demo = demo_list[current_index + 1]
        else:
            print("No more demos to run")
            break
        continue
    elif choice != 'r':
        print("Invalid choice. Please enter 'r', 's', or 'q'")
        continue

    if demo == pyTODemos.Load_STL:
        # Load an STL file and compute mass properties
        stl_file = './Models/AlcoaGrabCAD/AlcoaGrabCAD.STL'
        stl_geom = STLGeom(stl_file)
        [area, volume, cg, inertia] = stl_geom.compute_mass_properties()
        print(f"Area: {area}, Volume: {volume}, Center of Mass: {cg}, Inertia: {inertia}")
        stl_geom.plotGeometry(show_edges=False, show_axes=True, show_bounding_box=True)

    elif demo == pyTODemos.Create_Voxelmesh:
        # Create a voxel mesh from an STL file and display it
        mesh = Mesher()
        stlFileName = './Models/LBracket/LBracket.STL'
        mesh.createMeshFromSTLFile(stlFileName, nElemsDesired=10000)
        mesh.plot()
    elif demo == pyTODemos.ThermalFEA_Voxel_Pardiso:
        problem = ThermalExamples.ThickPlate
        nDOFDesired = 10000
        solver = Solvers.PARDISO
        mesh, mat_prop, bc = getThermalProblem(problem, nDOFDesired=nDOFDesired)
        thermal_fe_solver = ThermalFEA(mesh=mesh,
                    mat_prop=mat_prop,
                    bc=bc,
                    solver=solver)
        
        startTime = time.time()
        u = thermal_fe_solver.solve()
        uMax = np.max(np.abs(u))
        print("FEA time: ", time.time() - startTime)
        thermal_fe_solver.plot_temperature()
       
    elif demo == pyTODemos.StructuralFEA_Voxel_Pardiso:
        # This example uses the Beam Bending problem from the StructuralExamples module.
        problem = StructuralExamples.BeamBending 
        nDOFDesired = 10000 
        mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
        solver = Solvers.PARDISO
        fe_solver = StructFEA(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, elem_body_force=elem_body_force)
        fe_solver.plot_mesh()
        startTime = time.time()
        u = fe_solver.solve()
        print('Solver time: ', time.time() - startTime)
        fe_solver.postprocess()
        fe_solver.plot_deformation()
        fe_solver.plot_vonMisesStress()
        fe_solver.plot_stress_component(0)
        
    elif demo == pyTODemos.StructuralFEA_Voxel_DPCG:# DPCG solver for large scale problems
        # This example uses the Knuckle assembly problem from the StructuralExamples module.
        problem = StructuralExamples.KnuckleAssembly 
        nDOFDesired = 500000 
        mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
        solver = Solvers.DPCG
        dsolver = DeflationSolver()
        startTime = time.time()
  
        # for DPCG solver, create deflation groups and matrix
        nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_delfation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]

        fe_solver = StructFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)

        u = fe_solver.solve()
        print('Solver time: ', time.time() - startTime)
        fe_solver.plot_deformation()
    elif demo == pyTODemos.ModalFEA_Voxel_Pardiso:
        problem = StructuralExamples.LBracket
        nDOFDesired = 50000
        mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
        solver = Solvers.PARDISO 

        startTime = time.time()

        modal_solver = ModalFEA(mesh = mesh,
                mat_prop = mat_prop,
                bc = bc,
                solver = solver,
                elem_body_force = elem_body_force)

        nEigenModes = 3
        eigenvals, eigenvecs = modal_solver.computeEigenModes(nEigenModes = nEigenModes)
        
        print('-----------------------------')
        print("FEA time: ", time.time() - startTime)
        print('Eigenvalues: ', eigenvals)
        print('-----------------------------')
        for i in range(nEigenModes):
            modal_solver.plot_eigenmode(i)
    elif demo == pyTODemos.StructuralTO_Voxel_DensityMMA:
        to_problem = StructuralTOExamples.DistributedLoad # Choose the TO problem
        solver = Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
        # Get the structural problem
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        fe_solver = StructFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
   
        title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        fe_solver.plot_mesh(title = title)

        startTime = time.time()
        u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,plot_progress=True,
                                    to_params = to_params)
        timeTaken = time.time() - startTime

        print(f"Time taken: {timeTaken:.0f} s")
        title = f"MMA: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
    
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh(title = title,plot_bc= False)

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
        plt.show()

        # Save the mesh and results
    elif demo == pyTODemos.StructuralTO_Voxel_DensityOC:
        to_problem = StructuralTOExamples.EdgeCantilever # Choose the TO problem
        solver = Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
        # Get the structural problem
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        fe_solver = StructFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
        
        title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        fe_solver.plot_mesh(title = title)
        startTime = time.time()
        u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,plot_progress=True,
                                                to_params = to_params)
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

        print(f"Time taken: {timeTaken:.0f} s")
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh(title = title)

    elif demo == pyTODemos.StructuralTO_Voxel_Pareto:
        to_problem = StructuralTOExamples.CantileverTipLoad
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        fe_solver.plot_mesh(title = title)
        fe_solver = StructFEA(mesh=mesh, mat_prop=mat_prop, solver= Solvers.PARDISO, bc=bc)
        u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver=fe_solver,to_params=to_params,plot_progress=True)
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
    elif demo == pyTODemos.Create_Tetmesh:
        # Create a tet mesh from an STL file and display it
        tetmesh = TetMesher()
        stlFileName = './Models/BicycleCrank/BicycleCrank.STL'
        tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=20000)
        tetmesh.plot()
    elif demo == pyTODemos.ThermalFEA_Tet_Pardiso:
        # This example uses the ThermalExamples module.
        nDOFDesired = 10000
        solver = Solvers.PARDISO
        tetmesh, mat_prop, bc = createAnnularPlateThermalProblemTet(nDOFDesired=nDOFDesired)
        fe_solver = ThermalFEATet(mesh=tetmesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)

        startTime = time.time()
        fe_solver.assemble_global_stiffness_matrix()
        u = fe_solver.solve()
        uMax = np.max(np.abs(u))
        startTime = time.time()
        u = fe_solver.solve()
        uMax = np.max(np.abs(u))
        print("FEA time: ", time.time() - startTime)
        tetmesh.plotField(u,show_edges=False) # plot the solution field

    # Move to next demo
    demo_list = list(pyTODemos)
    current_index = demo_list.index(demo)
    if current_index < len(demo_list) - 1:
        demo = demo_list[current_index + 1]
    else:
        print("All demos completed")
        break
