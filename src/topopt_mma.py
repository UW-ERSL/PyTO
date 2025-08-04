from topopt_common import *
from topopt_material_model import *
import time
import matplotlib.pyplot as plt
from mmaWrapper import runMMA
def topopt_mma(fe_solver, #hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
                           to_params,
                            maxMMAIterations: int = 250, 
                            timeLimitSecs: float =3600, #1 hour
                             move_limit: float = 0.2,
                             kkt_tol: float = 1.e-6,
                             objective_tol: float = 1.e-4,
                             constraint_tol: float = 1.e-4,
                             print_progress: bool = True,
                            plot_progress: bool = False,
                            binarize_topology: bool = True,   
                            progress_callback=None, 
                            plotter=None  
                             ) -> tuple[np.ndarray, dict]:
    """MMA based topology optimization for minimum compliance.

    Args:
        fe_solver: The structural FEA solver object.
        maxMMAIterations: Maximum number of MMA iterations.
        volfrac: The target volume fraction.
        penal: The penalization factor for the SIMP method.
        move_limit: The maximum change allowed for the design variables in each
            iteration.
        kkt_tol: The tolerance for the KKT conditions.
        step_tol: The tolerance for the step size.

    Returns: The displacement field of the optimized structure.
    """
    def log_message(msg): # This is a helper function to log messages in GUI or console
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)   

    nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
    material_model = MaterialModel.SIMP 


    tStart = time.time()
    num_elems= fe_solver.mesh.num_elems
    history = {'objective': [], 'volume': []}
    if (print_progress):
        log_message("Computing Filters ...")
    [H,Hs] = createFilters(fe_solver, to_params)

    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,nDOFPerNode)
   
    
    if isinstance(fe_solver.mat_prop, list): # multiple materials
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp.youngs_modulus,mp.poissons_ratio,fe_solver.mesh.elem_size)
                for mp in fe_solver.mat_prop]
            KE = KE_list[0]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal( mp.thermal_conductivity,fe_solver.mesh.elem_size)
                for mp in fe_solver.mat_prop]
            KE = KE_list[0]    
        log_message("Assuming all elements have the same material properties")
    else: # single material
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
                                                                fe_solver.mat_prop.poissons_ratio,
                                                                fe_solver.mesh.elem_size)
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
    
    
    constraintType = to_params.Constraints[0][0] # assume this is the first constraint
    if (constraintType == TO_QOI.VOLUME_FRACTION):
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        volFractionConstraint =1 # default value

    if (fe_solver.elem_body_force is not None):
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None

    success = True
    errorMsg = "None"
    nFEAs = 0
  
    def optimizationFunction(x):
        x = np.asarray(x).flatten()
        if (to_params.APPLY_FILTER_TO_DENSITY):
            x = H*x/Hs
        fe_solver.mesh.setPseudoDensity(x)
        if (plot_progress):
            fe_solver.plot_pseudo_density(auto_close = False, title = f"Iter {len(history['objective']) + 1} - Density", save_path = None)
        sol = fe_solver.solve(x, material_model)
        fe_solver.postprocess()

        obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
        history['objective'].append(obj)
        history['volume'].append(np.mean(x))
		
       
        if (nodal_body_force is not None): # additional body force term
            ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
            grad_obj +=  2*ce_body_force*get_material_model_rho_sensitivity(x,material_model)

        if (to_params.APPLY_FILTER_TO_SENSITIVITY) and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            grad_obj = (H *(x*grad_obj))/Hs/x # apply weighted filter
        elif (to_params.APPLY_FILTER_TO_SENSITIVITY) and (to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION):
            grad_obj = (H *(grad_obj))/Hs # apply regular filter
        if (elemsWithForces.size > 0):
            grad_obj[elemsWithForces] = min(grad_obj) # retain elements that have nodes with external forces

        if (to_params.ElemsToKeep is not None):
            grad_obj[to_params.ElemsToKeep] = min(grad_obj) # also retain elements that are in the keep list

        c, dcdx = compute_constraint_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
        if (to_params.APPLY_FILTER_TO_SENSITIVITY):
            for m in range(len(to_params.Constraints)):
                if (to_params.Constraints[m][0] is TO_QOI.COMPLIANCE):
                    dcdx[m] = ((H *(x*dcdx[m]))/Hs/x) # apply weighted filter
                elif (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
                    dcdx[m] = ((H * dcdx[m])/Hs)# apply regular filter
    
        grad_obj = grad_obj.reshape(-1, 1)
        return obj, grad_obj, c, dcdx

    x0 = volFractionConstraint * np.ones(num_elems, dtype = float).reshape(-1, 1)
    lowerBound = np.zeros(num_elems, dtype = float).reshape(-1, 1)
    upperBound = np.ones(num_elems, dtype = float).reshape(-1, 1)
    nVariables = num_elems
    nConstraints = len(to_params.Constraints)
   

    [xOptimal,f0val, df0dx, gval, dgdx,nFEAs] = runMMA(nVariables,nConstraints,optimizationFunction,x0,lowerBound,
			 upperBound, maxIterations = maxMMAIterations,timeLimitSecs= timeLimitSecs, move_limit = move_limit,
             fTolerance= objective_tol,gTolerance= constraint_tol,kktTol = kkt_tol, verbose = print_progress, 
             progress_callback= progress_callback)

    x = np.asarray(xOptimal).flatten()
    fe_solver.mesh.setPseudoDensity(x)  
    sol = fe_solver.solve(x, material_model)  
    obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
    volfrac = np.mean(x)
    # Find threshold that preserves volume fraction
    if (binarize_topology):
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1-np.mean(x))*len(x))]
        x = np.where(x < threshold, 0.0, 1.0)
        volfrac = np.mean(x)
        fe_solver.mesh.setPseudoDensity(x)
        meshComponents = fe_solver.mesh.find_connected_components()
        if (len(meshComponents) > 1):
            if (print_progress):
                log_message("Disconnected topology detected. Removing hanging elements.")
            # Find the largest connected component and its size
            largest_component = max(meshComponents, key=len)
            # Set density to 1 for elements in largest component
            x[:] = 0.0
            x[list(largest_component)] = 1.0
            fe_solver.mesh.setPseudoDensity(x.flatten())
            volfrac = np.mean(x)
        sol = fe_solver.solve(x, material_model)
        obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
        history['objective'].append(obj)
        history['volume'].append(volfrac)

    
    if (volfrac > 1.1*volFractionConstraint):
        errorMsg = f"vf {volFractionConstraint:0.3f} not reached"
        success = False 
    grey_elements = np.sum((x > 0.1) & (x < 0.9))
    fraction_grey = (grey_elements / num_elems) 

    log_message(f"Final objective: {obj:.4g}, vf: {np.mean(x):.3f}, grey: {fraction_grey:.3f}")
   
    log_message(f"Total Time: {time.time() - tStart:.2f} s")
    log_message(f"Error: {errorMsg}")
    return np.asarray(sol), history,success,errorMsg,nFEAs
    
if __name__ == "__main__":    
    from topopt_structural_benchmarks import *
    from topopt_thermal_benchmarks import *
 
    print("-" * 50)
 
    to_problem = StructuralTOExamples.LBracketMidLoadStressObjective# Choose the TO problem
    nDOFDesired = 25000

    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem,nDOFDesired=nDOFDesired)
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem,nDOFDesired=nDOFDesired)

    
    print(f"Running {to_problem.name}...") 
    print("-" * 50)
    solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
   

    dsolver = deflation.DeflationSolver()
    if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):# Typically PARDISO, but DPCG for large DOF problems
        solver = lin_solv.Solvers.DPCG
        nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]

    if (to_problem in StructuralTOExamples):
        fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
    elif (to_problem in ThermalTOExamples):
        fe_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
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
    print("OptimizationMethod: MMA")
    
    u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,
                                to_params = to_params,
                                plot_progress = True)
    timeTaken = time.time() - startTime
    

    title = f"MMA: vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    fe_solver.plot_mesh(title = title, plot_bc = False, save_path = None)
    fe_solver.postprocess()
    fe_solver.plot_vonMisesStress()
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

    plt.title('MMA: Volume and Objective vs. Iterations')


    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.grid(True)
    plt.show()