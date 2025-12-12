"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from topopt_obj_cons_sensitivities import *
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from hex_mesher import DISTANCE_TYPE


def compute_compliance_and_sensitivity(feaMode, rho, fe_solver):
    """
    Compute objective and shape sensitivity for level set optimization.
    
    
    Shape sensitivity is negative strain energy density (NO SIMP penalization).
    """
    if feaMode == FEA_MODE.STRUCTURAL:
        dofMat = fe_solver.mesh.edofMatStructural
        # Get element stiffness matrix
        KE = fe_solver.elem_stiff[0]
    elif feaMode == FEA_MODE.THERMAL:
        dofMat = fe_solver.mesh.edofMatThermal
    else:
        raise ValueError(f"Invalid FEA mode: {feaMode}")
    
    sol = fe_solver.sol
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
    

def  compute_topological_sensitivity(rho,fe_solver):
     
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

    return rho*T

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
                    stepLength: int = 2,
                    numReinit: int = 10,
                    topWeight: float = 0,
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

    fe_solver.mesh.setPseudoDensity(np.asarray(rho))
    
    # Identify load-bearing elements
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)
    
    # History tracking
    history = {'objective': [], 'volfrac': []}
    success = True
    errorMsg = "No errors."
    
    # Main optimization loop 
    volPrev = 1.0
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
        obj, shapeSens = compute_compliance_and_sensitivity(feaMode, rho, fe_solver)
       
        topSens= compute_topological_sensitivity(rho,fe_solver)
         

        # Scaling of sensitivities; this will alow us to set stepLength and topWeight independent of problem scale
        if iteration == 0:
            shapeScaling = np.max(np.abs(shapeSens)) + 1e-12
            topoScaling = np.max(np.abs(topSens)) + 1e-12

        shapeSens = shapeSens / shapeScaling
        topSens = topSens / topoScaling
        # fe_solver.plot_elem_field(shapeSens, title = f"Shape Sensitivity - Iter {iteration}")
        # fe_solver.plot_elem_field(topSens, title = f"Topological Sensitivity - Iter {iteration}")
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
        vol_error = volCurr - volFractionConstraint
        if iteration == 0:
            la = -0.01
            La = 1000
            alpha = 0.9;  
        else:
            la = la - 1/La * vol_error; 
            La= alpha * La

        vol_penalty_term =  la - 1/La*vol_error

        shapeSens = shapeSens - vol_penalty_term
        topSens = topSens + 4*np.pi/3*(vol_penalty_term)


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

        volChangePercent = abs(volPrev - volCurr)/volCurr
        volPrev = volCurr
        # if (iteration > 5 and volChangePercent < 0.001):
        #     topWeight = topWeight * 1.1
        #     topWeight = min(topWeight,5)
        # elif volChangePercent > 0.025:
        #     topWeight = topWeight * 1
        #     topWeight = max(topWeight,0.1)
    
        print(f" Volume change: {volChangePercent:.5f}")
        print(f" Topological weight: {topWeight:.2f}")
    # Final solve
    
    mesh.setPseudoDensity(np.asarray(rho))
    sol = fe_solver.solve(rho, material_model)
    obj, _ = compute_compliance_and_sensitivity(feaMode, rho, fe_solver)
    
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
    """Godunov upwind using ONLY face neighbors."""
    neighbors = mesh.elemNeighborsArray
    
    # Face neighbor indices (from your 27-neighbor ordering)
    idx_xminus, idx_xplus = 12, 14
    idx_yminus, idx_yplus = 10, 16
    idx_zminus, idx_zplus = 4, 22
    
    def get_neighbor_lsf(idx):
        neighbor_id = neighbors[:, idx]
        return np.where(neighbor_id >= 0, lsf[neighbor_id], lsf)
    
    lsf_xm = get_neighbor_lsf(idx_xminus)
    lsf_xp = get_neighbor_lsf(idx_xplus)
    lsf_ym = get_neighbor_lsf(idx_yminus)
    lsf_yp = get_neighbor_lsf(idx_yplus)
    lsf_zm = get_neighbor_lsf(idx_zminus)
    lsf_zp = get_neighbor_lsf(idx_zplus)
    
    hx, hy, hz = mesh.elem_size
    
    # Finite differences
    dpx = (lsf_xp - lsf) / hx
    dmx = (lsf - lsf_xm) / hx
    dpy = (lsf_yp - lsf) / hy
    dmy = (lsf - lsf_ym) / hy
    dpz = (lsf_zp - lsf) / hz
    dmz = (lsf - lsf_zm) / hz
    
    # Godunov upwind (component-wise selection)
    grad_mag_shrink = np.sqrt(
        np.minimum(dmx, 0)**2 + np.maximum(dpx, 0)**2 +
        np.minimum(dmy, 0)**2 + np.maximum(dpy, 0)**2 +
        np.minimum(dmz, 0)**2 + np.maximum(dpz, 0)**2
    )
    
    grad_mag_expand = np.sqrt(
        np.maximum(dmx, 0)**2 + np.minimum(dpx, 0)**2 +
        np.maximum(dmy, 0)**2 + np.minimum(dpy, 0)**2 +
        np.maximum(dmz, 0)**2 + np.minimum(dpz, 0)**2
    )
    
    grad_mag = np.where(v < 0, grad_mag_shrink, grad_mag_expand)
    
    lsf_new = lsf - dt * v * grad_mag - topWeight * dt * g
    
    return lsf_new

if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.MBBBeam # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
     
	run_topopt_levelset(to_problem)