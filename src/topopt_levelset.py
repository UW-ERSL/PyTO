"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from scipy.ndimage import distance_transform_edt
import time


def topopt_levelset(fe_solver,
                    to_params,
                    maxIterations: int = 250,
                    numReinit: int = 10000,
                    plot_progress: bool = False,
                    print_progress : bool = True,
                    debug: bool = False) -> tuple[np.ndarray, dict]:
    """Level Set Method for Topology Optimization using Hamilton-Jacobi equation in 3D.

    Args:
        fe_solver: The structural FEA solver object.
        to_params: Topology optimization constraints.
        maxIterations: Maximum number of iterations.
        volfrac: The target volume fraction.
        time_step: Time step for the Hamilton-Jacobi update.
        numReinit: Reinitialization step performed at every numReinit^th iteration.
        topWeight: Topological sensitivity weight in modified Hamilton-Jacobi equation.
        debug: If True, prints debug information.

    Returns:
        A tuple containing the displacement field of the optimized structure
        and a dictionary containing the optimization history.
    """
    if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
        nDOFPerNode = 3
    else:
        nDOFPerNode = 1
            
    tStart = time.time()
    material_model = MaterialModel.SIMP 
    mesh=fe_solver.mesh


    if (print_progress):
        print("Computing Derivative Filters ...")
    HXD = createXDerivativeFilter(mesh)
    HYD = createYDerivativeFilter(mesh)
    HZD = createZDerivativeFilter(mesh)

    if (print_progress):
        print("Computing Smoothing Filters ...")
    [H,Hs] = createFilters(fe_solver, to_params)

    # Initialize level set function and design variables
    rho = np.ones((fe_solver.mesh.num_elems))
    lsf = fe_solver.mesh.compute_signed_distance_function(rho)
    lsf /= np.max(np.abs(lsf))

    shapeSens = np.zeros((fe_solver.mesh.num_elems))
   
    history = {'compliance': [], 'volume': []}

    # Material properties
    mat_prop = fe_solver.mat_prop
    # if (print_progress):
    
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
        print("Assuming all elements have the same material properties")
    else: # single material
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
	
    volCurr = 1.0
    volDecrementWeight = 0.1
    success = True
    errorMsg = ""    
    constraintType = to_params.Constraints[0][0] # assume this is the first constraint
    if (constraintType == TO_QOI.VOLUME_FRACTION):
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        volFractionConstraint =1 # default value
    beta  = 0.35
    for iterNum in range(maxIterations):
        if (plot_progress):
            fe_solver.plot_mesh(plot_bc = False,auto_close = False, title = f'Volfrac: {volCurr:0.3f}')

        sol = fe_solver.solve(rho, material_model)
   
        obj, grad_obj = compute_objective_and_gradient(to_params,sol,rho, fe_solver,KE, material_model)
		
        shapeSens =(-rho)* (np.dot(sol[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 8*nDOFPerNode), KE) * sol[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 8*nDOFPerNode)).sum(1)
 
        shapeSens = (H * shapeSens)/Hs
        shapeSens /= np.max(np.abs(shapeSens))
        
        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)


        volCurr = np.mean(rho)
        history['compliance'].append(obj)
        history['volume'].append(volCurr)
        
       
	
        if (print_progress):
            print(f"Iter: {iterNum}, Compliance: {obj:.4f}, Volume: {volCurr:.3f}")
        if (abs(volCurr - volFractionConstraint) < 0.001):
             break
        
        adjustedWeight = volDecrementWeight * (1 - abs(volCurr - volFractionConstraint))
        shapeSens = shapeSens + adjustedWeight * (volCurr - volFractionConstraint+beta)

        gradMag = GradientMagnitude(lsf,HXD,HYD,HZD)      
        gradMagSmooth = H*gradMag/Hs        
        
        lsf += (shapeSens*gradMagSmooth)
        rho = (lsf < 0).astype(np.float64)
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
        if (iterNum % numReinit == 0):
            print("Reinitializing level set function")
            lsf = fe_solver.mesh.compute_signed_distance_function(rho)
    
        lsf /= np.max(np.abs(lsf))
        lsf = H*lsf/Hs
        if(volCurr - volFractionConstraint > 0.001):
            volDecrementWeight += 0.0025
        else:
            volDecrementWeight -= 0.005
        
    sol = fe_solver.solve(rho, material_model)
    obj, _ = compute_objective_and_gradient(to_params,sol,rho, fe_solver,KE, material_model)
		
    history['compliance'].append(obj)
    history['volume'].append(volCurr)
    if iterNum == maxIterations - 1:
        errorMsg = "Maximum iterations reached"
        print(errorMsg)
        success = False
    if (obj > 2*history['compliance'][-2]):
        errorMsg = "Disconnected topology"
        success = False
    volfrac = history['volume'][-1]  # Define volfrac based on the last volume in history
    if volfrac > 1.1 * volFractionConstraint:
        errorMsg = f"vf {volFractionConstraint:0.3f} not reached"
        success = False

    nFEAs = iterNum + 1
    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['compliance'][-1]:.4f}, Final Volume: {history['volume'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s")

    return sol, history, success, errorMsg, nFEAs

def GradientMagnitude(lsf: np.ndarray,HXD,HYD,HZD):
		"""Create a filter matrix to compute the magnitude of the gradient of a scalar field.
		
		Args:
			mesh: The mesh object.
		
		Returns:
			tuple containing:
				HGM: Sparse matrix that computes the gradient magnitude.
				HGMs: Array of row sums of HGM matrix.
		"""
		# Compute derivative filters in x, y, and z directions
		lsfdx=HXD@lsf.T
		lsfdy=HYD@lsf.T
		lsfdz=HZD@lsf.T
		gradient_magnitude = np.sqrt(lsfdx**2 + lsfdy**2 + lsfdz**2)
		return gradient_magnitude
    
if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.CantileverMidLoad # Choose the TO problem
	#to_problem = ThermalTOExamples.BridgeThermal # Choose the TO problem
     

	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
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
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#plots.plotMesh(mesh, bc,title = title)


	startTime = time.time()
	
	print("OptimizationMethod: Level Set")
	u, history, success,errorMsg,nFEAs = topopt_levelset(fe_solver=fe_solver,
                                                    to_params=to_params,
                                                    maxIterations = 300,
                                                    plot_progress = True,
                                                    debug = False)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"	

	print(f"Time taken: {timeTaken:.0f} s")
	# if not success:
	# 	print(f"Error: {errorMsg}")
	fe_solver.plot_mesh(title = title, save_path = None)

	#plots.plotIsocontour(fe_solver.mesh, title = title, save_path = None)
	# Save the mesh and results