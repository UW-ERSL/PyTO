"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from topopt_obj_cons_sensitivities import *
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from hex_mesher import DISTANCE_TYPE

def compute_compliance_and_sensitivity(feaMode, sol, rho, fe_solver, KE):
    """
    Compute objective and shape sensitivity for level set optimization.
    
    
    Shape sensitivity is negative strain energy density (NO SIMP penalization).
    """
    if feaMode == FEA_MODE.STRUCTURAL:
        dofMat = fe_solver.mesh.edofMatStructural
    elif feaMode == FEA_MODE.THERMAL:
        dofMat = fe_solver.mesh.edofMatThermal
    else:
        raise ValueError(f"Invalid FEA mode: {feaMode}")
    
    num_elems = fe_solver.mesh.num_elems
    nRows = KE.shape[0]
    
    # Element strain energy: u_e^T * K_e * u_e
    ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * 
          sol[dofMat].reshape(num_elems, nRows)).sum(1)
    
    # Small stiffness for void 
    rho_clipped = np.maximum(rho, 0.0001)
    
    element_volume = np.prod(fe_solver.mesh.elem_size)
    # Shape sensitivity: -ρ * ce/element_volume
    # We divide by element volume to get sensitivity per unit volume
    # This will be consistent with topological sensitivity units
    shapeSens = -rho_clipped * ce/element_volume
    
    # Compliance 
    obj = np.sum(rho_clipped * ce)
    
    return obj, shapeSens
    
def run_topopt_levelset(to_problem):
     
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

    
	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
		feaMode = FEA_MODE.STRUCTURAL         
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
		feaMode = FEA_MODE.THERMAL
	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	if (feaMode == FEA_MODE.STRUCTURAL):
		fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					rtol = 1e-8,
					elem_body_force = elem_body_force)
	elif (feaMode == FEA_MODE.THERMAL):
		print("Thermal not fully supported for level set method yet.")
	

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#plots.plotMesh(mesh, bc,title = title)

	startTime = time.time()
	print("OptimizationMethod: Level Set")
	u, history, success,errorMsg,nFEAs = topopt_levelset(feaMode,fe_solver=fe_solver,
                                                    to_params=to_params,
                                                    plot_progress = True,
                                                    print_progress = True,
                                                    maxIterations=250,
                                                    debug = False)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"	


	fe_solver.plot_mesh(title = title, save_path = None)


def topopt_levelset(feaMode,fe_solver,
                    to_params,
                    maxIterations: int = 250,
                    numReinit: int = 10000,
                    objective_tol: float = 1.e-3,
                    constraint_tol: float = 1.e-3,
                    plot_progress: bool = False,
                    print_progress : bool = False,
                    plotter = None,
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

    def lsevol(phi, shapeSens, gradMagSmooth, la, dt):
        phi += ((shapeSens*gradMagSmooth)) * dt/la 
        phi = H @ phi.T/Hs
        rho = (phi < 0).astype(np.float64)
        rho = np.maximum(rho, void)
        return rho
    
    if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
        nDOFPerNode = 3
    else:
        nDOFPerNode = 1
            
    tStart = time.time()
    material_model = MaterialModel.SIMP 
    mesh=fe_solver.mesh
    
    constraintType = to_params.Constraints[0][0]  # assume this is the first constraint
    if (constraintType == TO_QOI.VOLUME_FRACTION):
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        raise ValueError(f"Unsupported constraint type: {constraintType}")


  
    HXD = createXDerivativeFilter(mesh)
    HYD = createYDerivativeFilter(mesh)
    HZD = createZDerivativeFilter(mesh)  
    [H,Hs] = createFilters(fe_solver, to_params)

    # Initialize level set function and design variables
    rho = np.ones((fe_solver.mesh.num_elems))

      # Determine distance type based on extrusion
    distanceType = DISTANCE_TYPE.DISTANCE_3D
    if to_params.ExtrudeX:
        distanceType = DISTANCE_TYPE.DISTANCE_YZ
    elif to_params.ExtrudeY:
        distanceType = DISTANCE_TYPE.DISTANCE_XZ
    elif to_params.ExtrudeZ:
        distanceType = DISTANCE_TYPE.DISTANCE_XY
    lsf = fe_solver.mesh.compute_signed_distance_function(rho, distance_type = distanceType)
    lsf /= np.max(np.abs(lsf))

    shapeSens = np.zeros((fe_solver.mesh.num_elems))
    history = {'objective': [], 'volfrac': []}
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,nDOFPerNode)
    if isinstance(fe_solver.mat_prop, list): # multiple materials
        print("Not tested")
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp.youngs_modulus,mp.poissons_ratio,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
            KE = KE_list[0]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal( mp.thermal_conductivity,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
            KE = KE_list[0]	
    else: # single material
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
    volCurr = 1.0
    void = 0
    vol_tol = 4/fe_solver.mesh.num_elems
    volFracDecrement = 0.05
    success = True
    errorMsg = "No errors."    
    constraintType = to_params.Constraints[0][0] # assume this is the first constraint
    if (constraintType == TO_QOI.VOLUME_FRACTION):
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        raise ValueError(f"Unsupported constraint type: {constraintType}")

    obj0 = None
    dt = 0.1
    for iteration in range(maxIterations):
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
        if plot_progress:
            fe_solver.plot_pseudo_density_realtime(
                title=f"Iter {iteration }",
                iteration = iteration,
                external_plotter=plotter
            )
        
        sol = fe_solver.solve(rho, material_model)
        obj, shapeSens = compute_compliance_and_sensitivity(feaMode, sol, rho, fe_solver, KE)
        if (obj0 is None):
            obj0 = obj
        c, _ = compute_constraint_and_gradient(feaMode, to_params,sol,rho, fe_solver,KE)
       
        shapeSens = (H * shapeSens)/Hs
        shapeSens /= np.max(np.abs(shapeSens))
        
        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)

        volCurr = np.mean(rho)
        history['objective'].append(obj)
        history['volfrac'].append(volCurr)
        
         # Extract names for printing
        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]
        if (print_progress):
            print(50* '-')
            print(f"Iteration: {iteration}")
            print(f"Min. Objective ({objective_name}): {obj:.3g}")
            for idx, val in enumerate(c.flatten()):
                print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} {'<='} {to_params.Constraints[idx][2]:.3g}?")
       
        
        if (iteration > 5):
            obj_err = (abs(history['objective'][-1] - history['objective'][-3]) / history['objective'][-3]) 
            vol_err = abs(history['volfrac'][-1] - volFractionConstraint)
            #print(obj_err,vol_err)
            if  (obj_err < objective_tol) and (vol_err < vol_tol):
                    break

        gradMag = GradientMagnitude(lsf,HXD,HYD,HZD)      
        gradMagSmooth = H*gradMag/Hs 
        Vlm = max(volFractionConstraint,volCurr - volFracDecrement)
        #bisection to find la
        lMin = 1e-6
        lMax = 1e9
        
        # find volume fraction for lMin and lMax
        rho = lsevol(-lsf, shapeSens, gradMagSmooth, lMin, dt)
        rho = lsevol(-lsf, shapeSens, gradMagSmooth, lMax, dt)

        while (lMax - lMin) > 1e-6:
            la = 0.5 * (lMin + lMax)
            rho = lsevol(-lsf, shapeSens, gradMagSmooth, la, dt)
            volNew = np.mean(rho)
            if volNew > Vlm:
                lMin = la
            else:
                lMax = la
                
        if ((iteration+1) % 5 == 0) and (abs(history['volfrac'][-1] - history['volfrac'][-2]) < constraint_tol):
            print("Reinitializing level set function")
            lsf = fe_solver.mesh.compute_signed_distance_function(rho)
            lsf /= -np.max(np.abs(lsf))


    fe_solver.mesh.setPseudoDensity(np.asarray(rho))
    if (iteration > 0 and iteration % numReinit == 0):
        print("Reinitializing level set function")
        lsf = fe_solver.mesh.compute_signed_distance_function(rho)
        lsf /= np.max(np.abs(lsf))
        lsf = H*lsf/Hs
        
    sol = fe_solver.solve(rho, material_model)
    obj, _ = compute_objective_and_gradient(to_params,sol,rho, fe_solver,KE, material_model)
		
    history['objective'].append(obj)
    history['volfrac'].append(volCurr)
    if iteration == maxIterations - 1:
        errorMsg = "Maximum iterations reached"
        print(errorMsg)
        success = False
    if (obj > 2*history['objective'][-2]):
        errorMsg = "Disconnected topology"
        success = False
    volfrac = history['volfrac'][-1]  # Define volfrac based on the last volume in history
    if volfrac > 1.1 * volFractionConstraint:
        errorMsg = f"vf {volFractionConstraint:0.3f} not reached"
        success = False

    nFEAs = iteration + 1
    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['objective'][-1]:.4f}, Final Volume: {history['volfrac'][-1]:.3f}")
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
	
	to_problem = StructuralTOExamples.MBBBeam # Choose the TO problem
	run_topopt_levelset(to_problem)
