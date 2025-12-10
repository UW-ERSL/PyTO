"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from topopt_obj_cons_sensitivities import *
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from hex_mesher import DISTANCE_TYPE

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
                    topWeight: float = 2.0,
                    objective_tol: float = 1.e-3,
                    constraint_tol: float = 1.e-3,
                    plot_progress: bool = False,
                    print_progress: bool = False,
                    plotter=None,
                    debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    Level Set Method for Topology Optimization.
    
    Implements Challis (2010) algorithm adapted for 3D arbitrary hex meshes.
    
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
    # if to_params.ExtrudeX:
    #     distanceType = DISTANCE_TYPE.DISTANCE_YZ
    # elif to_params.ExtrudeY:
    #     distanceType = DISTANCE_TYPE.DISTANCE_XZ
    # elif to_params.ExtrudeZ:
    #     distanceType = DISTANCE_TYPE.DISTANCE_XY
    #     print("Using 2D distance transform in XY plane for level set initialization.")
    
    # Initialize level set function (start with full domain)

    rho = np.ones(mesh.num_elems)
    lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)
    mesh.setPseudoDensity(np.asarray(rho))
    
    # Get element stiffness matrix
    KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
            fe_solver.mat_prop.youngs_modulus,
            fe_solver.mat_prop.poissons_ratio,
            mesh.elem_size )

    # Augmented Lagrangian parameters (Challis lines 38-39)
    lambda_lag = -0.01
    Lambda = 1000.0
    alpha = 0.9
    
    # Identify load-bearing elements
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)
    
    # History tracking
    history = {'objective': [], 'volfrac': []}
    success = True
    errorMsg = "No errors."
    
    # Main optimization loop (Challis line 16: for iterNum = 1:200)
    for iteration in range(maxIterations):
        mesh.setPseudoDensity(np.asarray(rho))
        
        if plot_progress:
            fe_solver.plot_pseudo_density_realtime(
                title=f"Iter {iteration + 1}",
                external_plotter=plotter
            )
        
        sol = fe_solver.solve(rho, material_model)
        fe_solver.postprocess()
        obj, shapeSens = compute_levelset_objective_and_gradient(feaMode, sol, rho, fe_solver, KE)
        T = compute_topological_sensitivity(fe_solver)
        topSens = rho * T # zero out void elements 
        
        # 4. Store history (Challis line 28)
        volCurr = np.mean(rho)
        history['objective'].append(obj)
        history['volfrac'].append(volCurr)
        
        # 5. Print progress (Challis lines 29-30)
        if print_progress:
            objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
            print('-' * 50)
            print(f"Iteration: {iteration + 1}")
            print(f"Objective ({objective_name}): {obj:.3g}")
            print(f"Volume fraction: {volCurr:.3f} (target: {volFractionConstraint:.3f})")
        
        # 6. Check convergence (Challis lines 35-36)
        if iteration > 5:
            obj_err = abs(history['objective'][-1] - history['objective'][-3]) / abs(history['objective'][-3])
            vol_err = abs(volCurr - volFractionConstraint)
            
            if obj_err < objective_tol and vol_err < constraint_tol:
                print("Convergence achieved!")
                break
        
        # 7. Update Lagrangian parameters (Challis lines 38-42)
        if iteration == 0:
            # Already initialized above
            pass
        else:
            lambda_lag = lambda_lag - (1 / Lambda) * (volCurr - volFractionConstraint)
            Lambda = alpha * Lambda
        
        # 8. Add volume constraint sensitivities (Challis lines 43-44)
        shapeSens = shapeSens - lambda_lag + (1 / Lambda) * (volCurr - volFractionConstraint)
        topSens = topSens + np.pi * (lambda_lag - (1 / Lambda) * (volCurr - volFractionConstraint))
        
        # 9. Design update (Challis line 46: [struc,lsf] = updateStep(...))
        rho, lsf = update_step(
            mesh=mesh,
            lsf=lsf,
            shapeSens=shapeSens,
            topSens=topSens,
            stepLength=stepLength,
            topWeight=topWeight,
            elemsWithForces=elemsWithForces,
            elemsToKeep=to_params.ElemsToKeep,
            H=H,
            Hs=Hs
        )
        
        # 10. Periodic reinitialization (Challis lines 48-50)
        if (iteration + 1) % numReinit == 0:
            if print_progress:
                print("  Reinitializing level set function...")
            lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)

    
    # Final solve
    mesh.setPseudoDensity(np.asarray(rho))
    sol = fe_solver.solve(rho, material_model)
    obj, _ = compute_levelset_objective_and_gradient(feaMode, sol, rho, fe_solver, KE)
    
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


def update_step(mesh, lsf, shapeSens, topSens, stepLength, topWeight,
                elemsWithForces, elemsToKeep, H, Hs):
    """
    Design update step (Challis's updateStep function, lines 54-62).
    
    Performs:
    1. Smooth sensitivities
    2. Enforce boundary conditions
    3. Evolve level set
    4. Extract new structure
    """
    # 1. Smooth the sensitivities (Challis lines 55-56)
    shapeSens_smooth = (H @ shapeSens) / Hs
    topSens_smooth = (H @ topSens) / Hs
    
    # 2. Normalize sensitivities (not in Challis, but helps with arbitrary meshes)
    max_shape = np.max(np.abs(shapeSens_smooth))
    if max_shape > 1e-10:
        shapeSens_smooth /= max_shape
    
    max_top = np.max(np.abs(topSens_smooth))
    if max_top > 1e-10:
        topSens_smooth /= max_top
    
    # 3. Load bearing elements must remain solid (Challis lines 58-59)
    if elemsWithForces is not None and len(elemsWithForces) > 0:
        shapeSens_smooth[elemsWithForces] = 0
        topSens_smooth[elemsWithForces] = 0
    
    if elemsToKeep is not None and len(elemsToKeep) > 0:
        shapeSens_smooth[elemsToKeep] = 0
        topSens_smooth[elemsToKeep] = 0
    
    # 4. Design update via evolution (Challis line 61)
    # [struc,lsf] = evolve(-shapeSens,topSens.*(lsf(2:end-1,2:end-1)<0),lsf,stepLength,topWeight);
    rho, lsf = evolve(
        mesh=mesh,
        lsf=lsf,
        v=-shapeSens_smooth,  # Velocity is negative of shape sensitivity
        g=topSens_smooth,     # Topological sensitivity (masking happens in evolve)
        stepLength=stepLength,
        topWeight=topWeight
    )
    
    return rho, lsf


def evolve(mesh, lsf, v, g, stepLength, topWeight):
    """
    Evolution of level set function (Challis's evolve function, lines 64-85).
    
    Implements Hamilton-Jacobi equation:
        ∂ψ/∂t = -v|∇ψ| - ω*g
    """
    # Forcing term only in solid region (Challis line 61: topSens.*(lsf<0))
    # Note: Challis masks at the call site, we mask here - equivalent
    g_masked = g * (lsf < 0)
    
    # CFL time step (Challis line 71: dt = 0.1/max(abs(v(:))))
    h = np.min(mesh.elem_size)
    max_v = np.abs(v).max()
    if max_v < 1e-10:
        rho = (lsf < 0).astype(float)
        return rho, lsf
    
    dt = 0.1 * h / max_v
    
    # Evolve for total time stepLength * CFL value (Challis line 73: for i = 1:(10*stepLength))
    num_steps = int(10 * stepLength)
    
    lsf_new = lsf.copy()
    for step in range(num_steps):
        lsf_new = upwind_step_arbitrary_mesh(mesh, lsf_new, v, g_masked, dt, topWeight)
    
    # Extract new structure (Challis lines 84-85)
    # strucFull = (lsf<0); struc = strucFull(2:end-1,2:end-1);
    rho = (lsf_new < 0).astype(float)
    
    return rho, lsf_new


def upwind_step_arbitrary_mesh(mesh, lsf, v, g, dt, omega):
    """
    Single upwind time step for arbitrary hex mesh.
    
    Implements one time step of Hamilton-Jacobi equation.
    Challis uses Godunov upwind on regular grid (lines 79-81).
    We adapt this for arbitrary hex meshes.
    """
    num_elems = mesh.num_elems
    lsf_new = lsf.copy()
    eps = 1e-12
    
    for e in range(num_elems):
        # Get valid neighbors (excluding self and -1)
        neighbors = mesh.elemNeighborsArray[e]
        valid_neighbors = neighbors[(neighbors != -1) & (neighbors != e)]
        
        if len(valid_neighbors) == 0:
            continue
        
        # Compute distances
        elem_center = mesh.elem_centers[e]
        neighbor_centers = mesh.elem_centers[valid_neighbors]
        deltas = neighbor_centers - elem_center
        distances = np.linalg.norm(deltas, axis=1)
        
        # Filter out zero distances
        nonzero_mask = distances > eps
        if not np.any(nonzero_mask):
            continue
        
        valid_neighbors = valid_neighbors[nonzero_mask]
        distances = distances[nonzero_mask]
        
        # Level set differences
        lsf_diffs = lsf[valid_neighbors] - lsf[e]
        
        # Upwind scheme: compute gradient magnitude
        # Challis lines 79-81 use separate terms for min(v,0) and max(v,0)
        # We simplify for arbitrary meshes
        if v[e] < 0:
            # Shrinking: use forward differences
            grads = np.maximum(lsf_diffs / distances, 0)
        else:
            # Expanding: use backward differences
            grads = np.maximum(-lsf_diffs / distances, 0)
        
        grad_mag = np.sqrt(np.sum(grads**2))
        
        # Update level set (Challis line 81: lsf = lsf - dt*min(vFull,0)*... - dt*max(vFull,0)*... - w*dt*gFull)
        lsf_new[e] = lsf[e] - dt * v[e] * grad_mag - omega * dt * g[e]
    
    return lsf_new


def compute_levelset_objective_and_gradient(feaMode, sol, rho, fe_solver, KE):
    """
    Compute objective and shape sensitivity for level set optimization.
    
    Challis line 24: shapeSens(ely,elx) = -max(struc(ely,elx),0.0001)*Ue'*KE*Ue;
    
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
    
    # Small stiffness for void (Challis uses 0.0001)
    rho_clipped = np.maximum(rho, 0.0001)
    
    # Shape sensitivity: -ρ * ce (no SIMP penalty!)
    shapeSens = -rho_clipped * ce
    
    # Compliance (Challis line 28: objective(iterNum) = -sum(shapeSens(:)))
    obj = np.sum(rho_clipped * ce)
    
    return obj, shapeSens
    
if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.Mitchell_1 # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
     
	run_topopt_levelset(to_problem)