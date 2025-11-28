from topopt_common import *
from topopt_obj_cons_sensitivities import *
from topopt_material_model import *
import time
import matplotlib.pyplot as plt
from mmaWrapper import runMMA
from topopt_thermostructural_sensitivity import ThermoElasticSensitivity



def topopt_mma(feaMode: FEA_MODE, fe_structural_solver, fe_thermal_solver, 
                           to_params,
                            maxMMAIterations: int = 150, 
                            timeLimitSecs: float = 36000, #10 hour
                            move_limit: float = 0.05,
                            kkt_tol: float = 1.e-6,
                            objective_tol: float = 1.e-4,
                            constraint_tol: float = 1.e-4,
                            print_progress: bool = True,
                            plot_progress: bool = False,
                            binarize_topology: bool = True,   
                            progress_callback=None, 
                            plotter=None  
                             ) -> tuple[np.ndarray, dict]:
    """Density-MMA based topology optimization 

    """
    def log_message(msg): # This is a helper function to log messages in GUI or console
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)   

    # We set fe_solver as the primary solver based on the FEA mode. Simplifies code below.
    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_solver = fe_structural_solver
        mesh = fe_structural_solver.mesh
        mat_prop = fe_structural_solver.mat_prop
        nDOFPerNode = 3
    elif (feaMode == FEA_MODE.THERMAL):
        fe_solver = fe_thermal_solver
        mesh = fe_thermal_solver.mesh
        mat_prop = fe_thermal_solver.mat_prop
        nDOFPerNode = 1
    elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
        fe_solver = fe_structural_solver  # primary solver is structural
        thermoElasticSensitivity = ThermoElasticSensitivity(fe_thermal_solver, fe_structural_solver)
        if (fe_structural_solver.mesh.num_elems != fe_thermal_solver.mesh.num_elems):
            raise ValueError("Structural and thermal meshes must have the same number of elements.")
        mesh = fe_structural_solver.mesh
        mat_prop = fe_structural_solver.mat_prop
        nDOFPerNode = 3
    else:
        raise ValueError("Either fe_structural_solver or fe_thermal_solver must be provided.")

    material_model = MaterialModel.SIMP 

    tStart = time.time()
    num_elems= mesh.num_elems
    
    history = {'objective': [] }
    for idx, constraint in enumerate(to_params.Constraints):
        history[f'constraint_{idx+1}'] = []

    history['volfrac'] = []
    if (print_progress):
        log_message("Computing Filters ...")
    [H,Hs] = createFilters(fe_solver, to_params)

    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force,nDOFPerNode)
   
    fe_solver.set_material(mat_prop)
    KE = fe_solver.elem_stiff[0]  # assuming all elements have same material properties
   
    if (fe_solver.elem_body_force is not None):
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None

    success = True
    errorMsg = "No errors."
    nFEAs = 0
    obj0 = None
    mmaIterations = 0
    
    def optimizationFunction(x):
        nonlocal nFEAs, obj0,mmaIterations
        x = np.asarray(x).flatten()
        grey_elements = np.sum((x > 0.1) & (x < 0.9))
        fraction_grey = (grey_elements / num_elems)
        if (print_progress):
            print(f"Percentange grey elements:", f"{fraction_grey*100:.2f}%")
        if (to_params.APPLY_FILTER_TO_DENSITY):
            x = H*x/Hs
        fe_solver.mesh.setPseudoDensity(x)
     
        if (plot_progress):
           if progress_callback is not None:
               progress_callback()
           fe_solver.plot_pseudo_density(
                   plotter=plotter,
                   auto_close=False,
                   title=f"Iter {len(history['objective']) + 1}"
               )
        
        if (feaMode == FEA_MODE.STRUCTURAL) or (feaMode == FEA_MODE.THERMAL):
            sol = fe_solver.solve(x, material_model)
            fe_solver.postprocess()

            obj, grad_obj = compute_objective_and_gradient(feaMode,to_params,sol,x, fe_solver,KE, material_model)
            c, dcdx = compute_constraint_and_gradient(feaMode,to_params,sol,x, fe_solver,KE, material_model)
        elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
            # Solve thermal problem first
            temperature = fe_thermal_solver.solve(x,material_model)
            thermo_elastic_force = fe_thermal_solver.get_thermoelastic_force(x,material_model)
            fe_structural_solver.set_thermal_forces(thermo_elastic_force)
            displacement = fe_structural_solver.solve(x, material_model) # structural solve
            sol = displacement.copy() # for use later
            fe_structural_solver.postprocess()
            obj, grad_obj = compute_thermoelastic_compliance_and_gradient(x, temperature, displacement,
												  fe_thermal_solver, fe_structural_solver)
            c, dcdx = compute_constraint_and_gradient(feaMode,to_params,displacement,x, fe_solver,KE, material_model)
  
        if (obj0 is None):
            obj0 = obj

        if any(c0 > 0.5 for c0 in c.flatten()): # if any constraint is significantly violated, zero out objective gradient
            grad_obj *= 0 # MMA step will try to reduce constraint violation first

        obj = obj/obj0 # normalize objective
        grad_obj = grad_obj/obj0 # normalize gradient

        if (nodal_body_force is not None): # additional body force term. Allowed for structural and thermo-structural problems only
            ce_body_force = (sol[mesh.edofMatStructural].reshape(num_elems, 24) * nodal_body_force[mesh.edofMatStructural].reshape(num_elems, 24)).sum(1)
            grad_obj +=  2*ce_body_force*get_material_model_rho_sensitivity(x,material_model)
        if (to_params.APPLY_FILTER_TO_SENSITIVITY) and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            grad_obj = (H *(x*grad_obj))/Hs/x # apply weighted filter
        elif (to_params.APPLY_FILTER_TO_SENSITIVITY) and (to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION):
            grad_obj = (H *(grad_obj))/Hs # apply regular filter
        if (elemsWithForces.size > 0):
            grad_obj[elemsWithForces] = min(grad_obj) # retain elements that have nodes with external forces

        if (to_params.ElemsToKeep is not None):
            grad_obj[to_params.ElemsToKeep] = min(grad_obj) # also retain elements that are in the keep list

        
        if (to_params.APPLY_FILTER_TO_SENSITIVITY):
            for m in range(len(to_params.Constraints)):
                if (to_params.Constraints[m][0] is TO_QOI.COMPLIANCE):
                    dcdx[m] = ((H *(x*dcdx[m]))/Hs/x) # apply weighted filter
                elif (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
                    dcdx[m] = ((H * dcdx[m])/Hs)# apply regular filter
    
        history['objective'].append(obj*obj0)
        history['volfrac'].append(np.mean(x))
        for idx, val in enumerate(c.flatten()):
            history[f'constraint_{idx+1}'].append(val)

        grad_obj = grad_obj.reshape(-1, 1)

        # Print objective and constraints for this iteration
        if (print_progress):
            print(50* '-')
            print(f"Iteration: {mmaIterations}")
            print(f"Min. Objective ({objective_name}): {obj*obj0:.3g}")
            inequality = '<='
            for idx, val in enumerate(c.flatten()):
                print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} {inequality} {to_params.Constraints[idx][2]:.3g}?")
        mmaIterations += 1
        nFEAs += 1
        return obj, grad_obj, c, dcdx


    objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
    constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]
    
    # Check if there's a volume fraction constraint and set initial density accordingly
    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION:
            initialDensity = constraint[2]  # Use the constraint value as initial density
            break
    
    x0 = initialDensity * np.ones(num_elems, dtype = float).reshape(-1, 1)
    lowerBound = np.zeros(num_elems, dtype = float).reshape(-1, 1)
    upperBound = np.ones(num_elems, dtype = float).reshape(-1, 1)
    nVariables = num_elems
    nConstraints = len(to_params.Constraints)
   
    [xOptimal,f0val, df0dx, gval, dgdx,nFEAs] = runMMA(nVariables,nConstraints,optimizationFunction,x0,lowerBound,
			 upperBound, maxIterations = maxMMAIterations,timeLimitSecs= timeLimitSecs, move_limit = move_limit,
             fTolerance= objective_tol,gTolerance= constraint_tol,kktTol = kkt_tol, verbose = False, 
             progress_callback= progress_callback)
    
    x = np.asarray(xOptimal).flatten()

    if (to_params.Eliminate_Hanging_Elements):
        # we must binarize for hanging element removal
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1-np.mean(x))*len(x))]
        x = np.where(x < threshold, 0.0, 1.0)
        fe_solver.mesh.setPseudoDensity(x)
        meshComponents = fe_solver.mesh.find_connected_components()
        if (len(meshComponents) > 1):
            if (print_progress):
                log_message(50* '-')
                log_message("Removing hanging elements.")
            # Find the largest connected component and its size
            largest_component = max(meshComponents, key=len)
            # Set density to 1 for elements in largest component
            x[:] = 0.0
            x[list(largest_component)] = 1.0
            fe_solver.mesh.setPseudoDensity(x.flatten())
    elif (binarize_topology):
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1-np.mean(x))*len(x))]
        x = np.where(x < threshold, 0.0, 1.0)

    # Get objective and gradient computation once again
    if (feaMode == FEA_MODE.THERMO_STRUCTURAL):
        # Solve thermal problem first
        temperature = fe_thermal_solver.solve(x,material_model)
        thermo_elastic_force = fe_thermal_solver.get_thermoelastic_force(x,material_model)
        fe_solver.set_thermal_forces(thermo_elastic_force)
        sol = fe_solver.solve(x, material_model) # structural solve
        fe_solver.postprocess()
        obj = sol.T @ fe_solver.stiff_mtrx @ sol
        grad_obj = thermoElasticSensitivity.compute_compliance_sensitivity(x, temperature, sol,
                                                                                p=3.0,  # structural penalty
                                                                                q=1.0,  # thermal penalty
                                                                                material_model=material_model)
        c, dcdx = compute_constraint_and_gradient(feaMode,to_params,sol,x, fe_solver,KE, material_model)
    else:
        fe_solver.mesh.setPseudoDensity(x)  
        sol = fe_solver.solve(x, material_model)  
        obj, grad_obj = compute_objective_and_gradient(feaMode,to_params,sol,x, fe_solver,KE, material_model)
        c, dcdx = compute_constraint_and_gradient(feaMode,to_params,sol,x, fe_solver,KE, material_model)

    grey_elements = np.sum((x > 0.1) & (x < 0.9))
    fraction_grey = (grey_elements / num_elems) 

    history['objective'].append(obj)
    history['volfrac'].append(np.mean(x))
    for idx, val in enumerate(c.flatten()):
        history[f'constraint_{idx+1}'].append(val)

    grey_elements = np.sum((x > 0.1) & (x < 0.9))
    fraction_grey = (grey_elements / num_elems) 
    print("-" * 50)
    log_message(f"Final objective: {obj:.4g}, vf: {np.mean(x):.3f}, grey: {fraction_grey:.3f}")

    log_message(f"Total Time: {time.time() - tStart:.2f} s")
    log_message(f"Error: {errorMsg}")
    return np.asarray(sol), history,success,errorMsg,nFEAs
    
if __name__ == "__main__":    
    from topopt_structural_benchmarks import *
    from topopt_thermal_benchmarks import *
    from topopt_thermostructural_examples import *
    print("-" * 50)

    # Choose the TO problem
    to_problem = StructuralTOExamples.LBracketMidLoad 
    #to_problem = ThermalTOExamples.FourCornersThermal
    to_problem = ThermoStructuralExamples.BiClamp

    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
        feaMode = FEA_MODE.STRUCTURAL
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
        feaMode = FEA_MODE.THERMAL
    elif (to_problem in ThermoStructuralExamples):
        structuralTO_problem = StructuralTOExamples.BiClamp
        thermalTO_problem = ThermalTOExamples.BiClamp
        mesh, mat_prop, structural_bc,elem_body_force, to_params = getStructuralTOProblem(structuralTO_problem)
        thermal_mesh, _, thermal_bc,_, thermal_to_params = getThermalTOProblem(thermalTO_problem)
        feaMode = FEA_MODE.THERMO_STRUCTURAL # or FEA_MODE.STRUCTURAL depending on the problem setup

    print(f"Running {to_problem.name}...") 
    print("-" * 50)
    
    solver = lin_solv.Solvers.PARDISO # default, see below
    dsolver = deflation.DeflationSolver(use_gpu=False)
    if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):# Typically PARDISO, but DPCG for large DOF problems
        solver = lin_solv.Solvers.DPCG
        nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]

    print('Solver: ', solver.name)
    fe_structural_solver = None
    fe_thermal_solver = None

    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_structural_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
        nNodes = fe_structural_solver.mesh.num_nodes
        nElems = fe_structural_solver.mesh.num_elems
    elif (feaMode == FEA_MODE.THERMAL):
        fe_thermal_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
        nNodes = fe_thermal_solver.mesh.num_nodes
        nElems = fe_thermal_solver.mesh.num_elems
    elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
        fe_structural_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = structural_bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
        fe_thermal_solver = hex_thermal_fea.HexThermalFEA(mesh = thermal_mesh,
                    mat_prop = mat_prop,
                    bc = thermal_bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8)

        nNodes = fe_structural_solver.mesh.num_nodes
        nElems = fe_structural_solver.mesh.num_elems    
    
    
    print("nNodes: ", nNodes )
    print("nElem: ", nElems)    
    title = f'nNodes: {nNodes}, nElem: {nElems}'

    plot_progress = True
    print_progress = True
    startTime = time.time()
    print("OptimizationMethod: MMA")
    
    u, history,success,errorMsg,nFEAs = topopt_mma(feaMode,fe_structural_solver,
                                                   fe_thermal_solver,
                                                    to_params = to_params,
                                                    plot_progress= plot_progress,
                                                    print_progress= print_progress,
                                                    maxMMAIterations= to_params.MaxIterations,)
    timeTaken = time.time() - startTime
    

    title = f"MMA: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    
    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_structural_solver.plot_mesh(title = title, plot_bc = True, save_path = None)
        fe_structural_solver.postprocess()
        fe_structural_solver.plot_vonMisesStress()
    elif (feaMode == FEA_MODE.THERMAL):
        fe_thermal_solver.plot_mesh(title = title, plot_bc = True, save_path = None)
        fe_thermal_solver.postprocess()
        fe_thermal_solver.plot_temperature()
   
    plt.figure()
    plt.plot(history['objective'], label='Objective')
    plt.xlabel('Iterations')
    plt.ylabel(f'Objective ({to_params.Objective[0].name})')
    plt.grid(True)
    plt.show()

    for idx, constraint_name in enumerate([getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]):
        plt.figure()
        val = np.array(history[f'constraint_{idx+1}'])
        constraint_val = (val+1)*to_params.Constraints[idx][2]
        plt.plot(constraint_val, label=f'Constraint {idx+1} ({constraint_name})')
        plt.xlabel('Iterations')
        plt.ylabel(f'Constraint {idx+1} ({constraint_name})')
        plt.axhline(y=to_params.Constraints[idx][2], color='r', linestyle='--')
        plt.grid(True)
        plt.show()
    
