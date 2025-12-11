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
    

def  compute_topological_sensitivity(fe_solver):
     
    strains = fe_solver.strainComponents
    stresses = fe_solver.stressComponents
    stress_tensor = np.array([
		[stresses[:, 0], stresses[:, 3], stresses[:, 4]],
		[stresses[:, 3], stresses[:, 1], stresses[:, 5]],
		[stresses[:, 4], stresses[:, 5], stresses[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
    strain_tensor = np.array([
		[strains[:, 0], strains[:, 3]/2, strains[:, 4]/2],
		[strains[:, 3]/2, strains[:, 1], strains[:, 5]/2],
		[strains[:, 4]/2, strains[:, 5]/2, strains[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
    trace_stress = np.trace(stress_tensor, axis1=1, axis2=2)
    trace_strain = np.trace(strain_tensor, axis1=1, axis2=2)
	
    nu = fe_solver.mat_prop.poissons_ratio
    T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1, 2)) -
			(1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)

    return T

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


def topopt_levelset(feaMode,
                    fe_solver,
                    to_params,
                    maxIterations: int = 250,
                    stepLength: int = 3,
                    numReinit: int = 5,
                    topWeight: float = 200,
                    objective_tol: float = 0.01,
                    constraint_tol: float = 0.01,
                    plot_progress: bool = False,
                    print_progress: bool = False,
                    plotter=None,
                    debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    Level Set Method for Topology Optimization.

    References:
        Challis, V. J. (2010). A discrete level-set topology optimization code 
        written in Matlab. Structural and Multidisciplinary Optimization, 41(3), 453-464.
    """
    
    if (feaMode == FEA_MODE.STRUCTURAL):
        nDOFPerNode = 3
    else:
        print("Thermal FEA not fully supported for level set method yet.")
        return
    
    tStart = time.time()
    material_model = MaterialModel.SIMP  # Not really used since rho is 0/1
    mesh = fe_solver.mesh
    
    objectiveType = to_params.Objective[0]
    if objectiveType != TO_QOI.COMPLIANCE:
        raise ValueError(f"Unsupported objective type: {objectiveType}")
    # Extract volume constraint
    constraintType = to_params.Constraints[0][0]
    if constraintType == TO_QOI.VOLUME_FRACTION:
        volFractionConstraint = to_params.Constraints[0][2]
    else:
        raise ValueError(f"Unsupported constraint type: {constraintType}")
    
    # Create filters
    [H, Hs] = createFilters(fe_solver, to_params)
    
    # Determine distance type based on extrusion
    distanceType = DISTANCE_TYPE.DISTANCE_3D
    if to_params.ExtrudeX:
        distanceType = DISTANCE_TYPE.DISTANCE_YZ
    elif to_params.ExtrudeY:
        distanceType = DISTANCE_TYPE.DISTANCE_XZ
    elif to_params.ExtrudeZ:
        distanceType = DISTANCE_TYPE.DISTANCE_XY


    # Initialize level set function (start with full domain)
    rho = np.ones(mesh.num_elems)
    lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)
    mesh.setPseudoDensity(np.asarray(rho))
    
    # Get element stiffness matrix
    KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            fe_solver.mat_prop.youngs_modulus,
            fe_solver.mat_prop.poissons_ratio,
            mesh.elem_size )
    
    # Identify load-bearing elements
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)
    
    # History tracking
    history = {'objective': [], 'volfrac': []}
    success = True
    errorMsg = "No errors."
    
    # Check one element's neighbors
    elem_id = 4534  # Middle element
    elem_center = mesh.elem_centers[elem_id]
    neighbors = mesh.elemNeighborsArray[elem_id]

    print(f"Element {elem_id} center: {elem_center}")
    print(f"Number of neighbors: {np.sum(neighbors >= 0)}")

    for i, n in enumerate(neighbors):
        if n >= 0:
            neighbor_center = mesh.elem_centers[n]
            delta = neighbor_center - elem_center
            dist = np.linalg.norm(delta)
            print(f"Neighbor {i:2d}: delta={delta}, dist={dist:.4f}")

    input("Press Enter to continue...")
    # Main optimization loop 
    for iteration in range(maxIterations):
        mesh.setPseudoDensity(np.asarray(rho))
        
        if plot_progress:
            fe_solver.plot_pseudo_density_realtime(
                title=f"Iter {iteration }",
                iteration = iteration,
                external_plotter=plotter
            )
        #input("Press Enter to continue...")
        sol = fe_solver.solve(rho, material_model)
        fe_solver.postprocess()
        obj, shapeSens = compute_compliance_and_sensitivity(feaMode, sol, rho, fe_solver, KE)

        T = compute_topological_sensitivity(fe_solver)
        topSens = rho * T # zero out void elements 

        # Scaling of sensitivities; this will alow us to set stepLength and topWeight independent of problem scale
        if iteration == 0:
            sensitivityScaling =  max(np.max(np.abs(shapeSens)), 
                                np.max(np.abs(topSens))) + 1e-12 # Use same scaling for all iterations


        shapeSens = shapeSens / sensitivityScaling
        topSens = topSens / sensitivityScaling
        # Print max and min of shapeSens
 
        # 3. Load bearing elements must remain solid 
        if elemsWithForces is not None and len(elemsWithForces) > 0:
            shapeSens[elemsWithForces] = min(shapeSens)
            topSens[elemsWithForces] = max(topSens)
        if (to_params.ElemsToKeep is not None):
            shapeSens[to_params.ElemsToKeep] = min(shapeSens) # also retain elements that are in the keep list
            topSens[to_params.ElemsToKeep] = max(topSens)
        # 4. Store history 
        volCurr = np.mean(rho)
        history['objective'].append(obj)
        history['volfrac'].append(volCurr)
        
        # 5. Print progress 
        if print_progress:
            objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
            print('-' * 50)
            print(f"Iteration: {iteration}")
            print(f"Objective ({objective_name}): {obj:.3g}")
            print(f"Volume fraction: {volCurr:.3f} (target: {volFractionConstraint:.3f})")
        
        # 6. Check convergence 
        if iteration > 5:
            obj_err = abs(history['objective'][-1] - history['objective'][-2]) / abs(history['objective'][-2])
            vol_err = abs(volCurr - volFractionConstraint)  
            if obj_err < objective_tol and vol_err < constraint_tol:
                print("Convergence achieved!")
                break
        
        # 7.  Lagrangian parameters 
        if iteration == 0:
            lambda_lag = -0.01
            Lambda = 2000.0
            alpha = 0.9
        else:
            lambda_lag = lambda_lag - (1 / Lambda) * (volCurr - volFractionConstraint)
            Lambda = alpha * Lambda

        # 8. Add volume constraint sensitivities 
        shapeSens = shapeSens - lambda_lag + (1 / Lambda) * (volCurr - volFractionConstraint)
        topSens = topSens + 4*np.pi/3*(lambda_lag - (1 / Lambda) * (volCurr - volFractionConstraint))


        #input("Press Enter to continue...")
        # 1. Smooth the sensitivities 
        shapeSens_smooth = (H @ shapeSens) / Hs
        topSens_smooth = (H @ topSens) / Hs
        
  
        # 4. Design update via evolution 
        rho, lsf = evolveUpWind(
                mesh=mesh,
                lsf=lsf,
                v=-shapeSens_smooth,  # Velocity is negative of shape sensitivity
                g=topSens_smooth,     # Topological sensitivity (masking happens in evolve)
                stepLength=stepLength,
                topWeight=topWeight
        )
 

        # 10. Periodic reinitialization 
        if (iteration + 1) % numReinit == 0:
            if print_progress:
                print("  Reinitializing level set function...")
            lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)

    
    # Final solve
    mesh.setPseudoDensity(np.asarray(rho))
    sol = fe_solver.solve(rho, material_model)
    obj, _ = compute_compliance_and_sensitivity(feaMode, sol, rho, fe_solver, KE)
    
    history['objective'].append(obj)
    history['volfrac'].append(volCurr)
    

    # Check final state
    if iteration == maxIterations - 1:
        errorMsg = "Maximum iterations reached"
        print(errorMsg)
        success = False
    
    if len(history['objective']) > 1 and obj > 2 * history['objective'][0]:
        errorMsg = "Disconnected topology"
        success = False
    
    if volCurr > 1.1 * volFractionConstraint:
        errorMsg = f"Volume fraction {volFractionConstraint:.3f} not reached"
        success = False
    
    nFEAs = iteration + 1
    totalTime = time.time() - tStart
    
    print(f"Final Compliance: {history['objective'][-1]:.4f}, Final Volume: {history['volfrac'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s, FEAs: {nFEAs}")
    
    return sol, history, success, errorMsg, nFEAs

def evolveUpWind(mesh, lsf, v, g, stepLength, topWeight):
    """
    Evolution of level set function.
    Implements Hamilton-Jacobi equation:
        ∂ψ/∂t = -v|∇ψ| - ω*g
    """
    # Forcing term only in solid region 
    g_masked = g * (lsf < 0)
    
    # CFL time step 
    h = np.min(mesh.elem_size)
    max_v = np.abs(v).max()
    if max_v < 1e-10:
        rho = (lsf < 0).astype(float)
        return rho, lsf
    
    increment = 0.1
    dt = increment * h / max_v
 
    # Evolve for total time stepLength * CFL value 
    num_steps = int( stepLength/increment)

    for _ in range(num_steps):
        lsf = upwind_step(mesh, lsf, v, g_masked, dt, topWeight)
  
    
    rho = (lsf < 0).astype(float)
    return rho, lsf


def upwind_step(mesh, lsf, v, g, dt, topWeight):
    """
    Single upwind time step for arbitrary hex mesh (vectorized).
    Implements one time step of Hamilton-Jacobi equation.
    """
    num_elems = mesh.num_elems
    eps = 1e-12
    
    # Get all neighbor information at once
    neighbors = mesh.elemNeighborsArray  # Shape: (num_elems, max_neighbors)
    
    # Create mask for valid neighbors (excluding self and -1)
    elem_indices = np.arange(num_elems)[:, None]  # Shape: (num_elems, 1)
    valid_mask = (neighbors != -1) & (neighbors != elem_indices)
    
    # Compute all element center differences
    elem_centers = mesh.elem_centers  # Shape: (num_elems, 3)
    
    # Get neighbor centers (invalid neighbors point to element 0, will be masked)
    neighbor_indices = np.where(neighbors >= 0, neighbors, 0)
    neighbor_centers = elem_centers[neighbor_indices]  # Shape: (num_elems, max_neighbors, 3)
    
    # Compute deltas and distances
    deltas = neighbor_centers - elem_centers[:, None, :]  # (num_elems, max_neighbors, 3)
    distances = np.linalg.norm(deltas, axis=2)  # (num_elems, max_neighbors)
    
    # Update valid_mask to exclude zero/near-zero distances
    valid_mask = valid_mask & (distances > eps)
    
    # Set invalid distances to 1 to avoid division by zero (will be masked out)
    distances = np.where(valid_mask, distances, 1.0)
    
    # Get level set values at neighbors
    lsf_neighbors = lsf[neighbor_indices]  # Shape: (num_elems, max_neighbors)
    lsf_diffs = lsf_neighbors - lsf[:, None]  # (num_elems, max_neighbors)
    
    # Compute gradients based on upwind scheme
    # Forward differences (for v < 0, shrinking)
    grads_forward = np.maximum(lsf_diffs / distances, 0)
    
    # Backward differences (for v >= 0, expanding)
    grads_backward = np.maximum(-lsf_diffs / distances, 0)
    
    # Select based on velocity sign
    v_expanded = v[:, None]  # (num_elems, 1)
    grads = np.where(v_expanded < 0, grads_forward, grads_backward)
    
    # Apply valid mask (set invalid gradients to 0)
    grads = np.where(valid_mask, grads, 0)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(np.sum(grads**2, axis=1))  # (num_elems,)
    
    # Update level set (Challis eq. 3: ∂ψ/∂t = -v|∇ψ| - ω*g)
    lsf_new = lsf - dt * v * grad_mag - topWeight * dt * g
    
    return lsf_new

if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.EdgeCantilever # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
     
	run_topopt_levelset(to_problem)