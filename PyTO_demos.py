'''
This script is a demo for the pyTO package. 
It includes examples of how to use the package for various tasks, including loading STL files, 
creating hex meshes, and performing finite element analysis (FEA) and topology optimization (TO).
'''


import sys
import enum
import time
import numpy as np  
import matplotlib.pyplot as plt

# PyTO files
sys.path.append('./src') #assumes PyTO src files are in this directory
from stl_reader import STLGeom
from hex_mesher import HexMesher
from linear_solvers import Solvers
from deflation import DeflationSolver
from hex_structural_fea import HexStructuralFEA
from hex_modal_fea import ModalFEA
from hex_thermal_fea import HexThermalFEA
from topopt_mma import topopt_mma
from topopt_pareto import topopt_pareto
from topopt_oc import topopt_optimality_criteria
from tet_mesher import TetMesher
from hex_structural_examples import *
from hex_thermal_examples import *

from topopt_structural_benchmarks import *
from topopt_thermal_benchmarks import *
from tet_thermal_examples import  *
from tet_thermal_fea import TetThermalFEA
from tet_structural_examples import *
from tet_structural_fea import TetStructuralFEA


class pyTODemos(enum.Enum):
	Load_STL = enum.auto() # Load an STL file and compute mass properties

    # The following demos are non-conforming voxel mesh FEA
	HexCreateMesh = enum.auto() # Create a hex mesh from an STL file  
	HexThermalFEA_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver
	HexStructuralFEA_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver
	HexStructuralFEA_DPCG = enum.auto() # Create and solve a predefined structural FEA problem using the DPCG solver
	HexModalFEA_Pardiso = enum.auto() # Create and solve a predefined structural FEA problem using the Pardiso solver    
	
     # The following demos are non-conforming voxel mesh Structural topology optimization
	HexStructuralTO_DensityMMA = enum.auto() # Create and solve a structural topology optimization problem using Density-method and MMA solver
	HexStructuralTO_DensityOC = enum.auto() # Create and solve a structural topology optimization problem using Density-method and OC solver
	HexStructuralTO_Pareto = enum.auto() # Create and solve a structural topology optimization problem using Pareto method
    
    # The following demos are non-conforming voxel mesh thermal topology optimization
	HexThermalTO_DensityMMA = enum.auto() # Create and solve a thermal topology optimization problem using Density-method and MMA solver
	HexThermalTO_DensityOC = enum.auto() # Create and solve a thermal topology optimization problem using Density-method and OC solver
	HexThermalTO_Pareto = enum.auto() # Create and solve a thermal topology optimization problem using Pareto method
    

    # The following demos use a conforming tet mesh
	TetCreateMesh = enum.auto() # Create a tet mesh from an STL file 
	TetThermalFEA_Pardiso = enum.auto() # Create and solve a thermal FEA problem using the PARDISO solver 
	TetStructuralFEA_Pardiso = enum.auto() # Create and solve a thermal FEA problem using the PARDISO solver 

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

    elif demo == pyTODemos.HexCreateMesh:
        # Create a hex mesh from an STL file and display it
        mesh = HexMesher()
        stlFileName = './Models/LBracket/LBracket.STL'
        mesh.createMeshFromSTLFile(stlFileName, nElemsDesired=10000)
        mesh.plot()
    elif demo == pyTODemos.HexThermalFEA_Pardiso:
        problem = HexThermalExamples.ThickPlate
        nDOFDesired = 10000
        solver = Solvers.PARDISO
        mesh, mat_prop, bc = getThermalProblem(problem, nDOFDesired=nDOFDesired)
        thermal_fe_solver = HexThermalFEA(mesh=mesh,
                    mat_prop=mat_prop,
                    bc=bc,
                    solver=solver)
        
        startTime = time.time()
        u = thermal_fe_solver.solve()
        uMax = np.max(np.abs(u))
        print("FEA time: ", time.time() - startTime)
        thermal_fe_solver.plot_temperature()
       
    elif demo == pyTODemos.HexStructuralFEA_Pardiso:
        # This example uses the Beam Bending problem from the StructuralExamples module.
        problem = StructuralExamples.BeamBending 
        nDOFDesired = 10000 
        mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
        solver = Solvers.PARDISO
        fe_solver = HexStructuralFEA(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, elem_body_force=elem_body_force)
        fe_solver.plot_mesh()
        startTime = time.time()
        u = fe_solver.solve()
        print('Solver time: ', time.time() - startTime)
        fe_solver.postprocess()
        fe_solver.plot_deformation()
        fe_solver.plot_vonMisesStress()
        fe_solver.plot_stress_component(0)
        
    elif demo == pyTODemos.HexStructuralFEA_DPCG:# DPCG solver for large scale problems
        # This example uses the Knuckle assembly problem from the StructuralExamples module.
        problem = StructuralExamples.ThreeHoleBracket 
        nDOFDesired = 150000 
        mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
        solver = Solvers.DPCG
        dsolver = DeflationSolver()
        startTime = time.time()
  
        # for DPCG solver, create deflation groups and matrix
        nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]

        fe_solver = HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)

        u = fe_solver.solve()
        print('Solver time: ', time.time() - startTime)
        fe_solver.plot_deformation()
    elif demo == pyTODemos.HexModalFEA_Pardiso:
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
    elif demo == pyTODemos.HexStructuralTO_DensityMMA:
        to_problem = StructuralTOExamples.Mitchell_1 # Choose the TO problem
        solver = Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
        # Get the structural problem
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        fe_solver = HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
   
        title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        startTime = time.time()
        u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,plot_progress=True,
                                    to_params = to_params)
        timeTaken = time.time() - startTime

        print(f"Time taken: {timeTaken:.0f} s")
        title = f"MMA: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"
    
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh(title = title,plot_bc= False)

        fig, ax1 = plt.subplots()

        # Plot compliance on left y-axis
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('objective', color='tab:blue')
        ax1.plot(history['objective'], color='tab:blue', label='objective')
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
    elif demo == pyTODemos.HexStructuralTO_DensityOC:
        to_problem = StructuralTOExamples.Mitchell_2 # Choose the TO problem
        solver = Solvers.PARDISO #
        # Get the structural problem
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem,nDOFDesired = 50000)
        fe_solver = HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
        
        title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        startTime = time.time()
        u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,plot_progress=True,
                                                to_params = to_params)
        timeTaken = time.time() - startTime
        title = f"OC: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"

        fig, ax1 = plt.subplots()

        # Plot compliance on left y-axis
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('objective', color='tab:blue')
        ax1.plot(history['objective'], color='tab:blue', label='objective')
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

    elif demo == pyTODemos.HexStructuralTO_Pareto:
        to_problem = StructuralTOExamples.Mitchell_3
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        
        fe_solver = HexStructuralFEA(mesh=mesh, mat_prop=mat_prop, solver= Solvers.PARDISO, bc=bc)
        u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver=fe_solver,to_params=to_params,plot_progress=True)
        plt.figure()
        plt.plot(history['volume'], history['objective'], marker='o')
        plt.xlabel('Volume Fraction')
        plt.ylabel('objective')
        plt.title('Pareto: Volume vs Compliance History')
        plt.grid(True)
        plt.show()
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh()
    elif demo == pyTODemos.HexThermalTO_DensityMMA:
        to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
        solver = Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
  
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
        fe_solver = HexThermalFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
   
        title = f'nDOF: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        startTime = time.time()
        u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,plot_progress=True,
                                    to_params = to_params)
        timeTaken = time.time() - startTime

        print(f"Time taken: {timeTaken:.0f} s")
        title = f"MMA: nDOF: {fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"
    
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh(title = title,plot_bc= False)

        fig, ax1 = plt.subplots()

        # Plot compliance on left y-axis
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('objective', color='tab:blue')
        ax1.plot(history['objective'], color='tab:blue', label='objective')
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
    elif demo == pyTODemos.HexThermalTO_DensityOC:
        to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
        solver = Solvers.PARDISO #
        # Get the structural problem
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem,nDOFDesired = 50000)
        fe_solver = HexThermalFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    elem_body_force = elem_body_force)
        
        title = f'nDOF: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
        startTime = time.time()
        u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,plot_progress=True,
                                                to_params = to_params)
        timeTaken = time.time() - startTime
        title = f"OC: nDOF: {fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"

        fig, ax1 = plt.subplots()

        # Plot compliance on left y-axis
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('objective', color='tab:blue')
        ax1.plot(history['objective'], color='tab:blue', label='objective')
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

    elif demo == pyTODemos.HexThermalTO_Pareto:
        to_problem = ThermalTOExamples.FourCornersThermal
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
        
        fe_solver = HexThermalFEA(mesh=mesh, mat_prop=mat_prop, solver= Solvers.PARDISO, bc=bc)
        u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver=fe_solver,to_params=to_params,plot_progress=True)
        plt.figure()
        plt.plot(history['volume'], history['objective'], marker='o')
        plt.xlabel('Volume Fraction')
        plt.ylabel('objective')
        plt.title('Pareto: Volume vs Compliance History')
        plt.grid(True)
        plt.show()
        if not success:
            print(f"Error: {errorMsg}")
        fe_solver.plot_mesh(title = title)
    elif demo == pyTODemos.TetCreateMesh:
        # Create a tet mesh from an STL file and display it
        tetmesh = TetMesher()
        stlFileName = './Models/BicycleCrank/BicycleCrank.STL'
        tetmesh.createTetMeshFromSTLFile(stlFileName, nElemsDesired=20000)
        tetmesh.plot()
    elif demo == pyTODemos.TetThermalFEA_Pardiso:
        # This example uses the ThermalExamples module.
        nDOFDesired = 10000
        solver = Solvers.PARDISO
        tetmesh, mat_prop, bc = createAnnularPlateThermalProblemTet(nDOFDesired=nDOFDesired)
        fe_solver = TetThermalFEA(mesh=tetmesh,
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
    elif demo == pyTODemos.TetStructuralFEA_Pardiso:
        nDOFDesired = 1000
        solver = Solvers.PARDISO
        problem = TetStructuralExamples.BeamBending # CubeCompression, TensileBar, TorsionBar, BeamBending
        quadratic_tet_mesh, mat_prop, bc, elem_body_force  = getTetStructuralProblem(problem,nDOFDesired = 1000)
        solver = Solvers.PARDISO # typically DPCG or PARDISO
        fe_solver = TetStructuralFEA(quadratic_tet_mesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)

        startTime = time.time()
        fe_solver.assemble_global_stiffness_matrix()
        fe_solver.solve()
        delta = np.max(np.abs(fe_solver.deformation))
    
        nDOF = fe_solver.mesh.num_nodes
    
        print('-----------------------------')
        print("nDof: ", nDOF)
        print('Solver: ', fe_solver.solver.name)
        print("FEA time: ", time.time() - startTime)
        print('Max deformation: ', delta)
        print('-----------------------------')
        fe_solver.plot_deformation()
    # Move to next demo
    demo_list = list(pyTODemos)
    current_index = demo_list.index(demo)
    if current_index < len(demo_list) - 1:
        demo = demo_list[current_index + 1]
    else:
        print("All demos completed")
        break
