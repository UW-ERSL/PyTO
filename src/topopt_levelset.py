"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from topopt_obj_cons_sensitivities import *
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from hex_mesher import DISTANCE_TYPE


def compute_compliance_and_sensitivity(feaMode, rho, fe_solver,void = 0.0001):
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
        KE = fe_solver.elem_stiff[0]
    else:
        raise ValueError(f"Invalid FEA mode: {feaMode}")
    
    sol = fe_solver.sol
    num_elems = fe_solver.mesh.num_elems
    nRows = KE.shape[0]
    
    # Element strain energy: u_e^T * K_e * u_e
    ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * 
          sol[dofMat].reshape(num_elems, nRows)).sum(1)
    
    # Small stiffness for void 
    rho_clipped = np.maximum(rho, void)
    
    element_volume = np.prod(fe_solver.mesh.elem_size)
    # Shape sensitivity: -ρ * ce/element_volume
    # We divide by element volume to get sensitivity per unit volume
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
		nDof = 3*fe_solver.mesh.num_nodes
	elif (feaMode == FEA_MODE.THERMAL):
		fe_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = solver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
		nDof = fe_solver.mesh.num_nodes

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", nDof)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {nDof}, nElem: {fe_solver.mesh.num_elems}'
	#plots.plotMesh(mesh, bc,title = title)

	startTime = time.time()
	print("OptimizationMethod: Level Set")
	u, history, success,errorMsg,nFEAs = topopt_levelset(feaMode,fe_solver=fe_solver,
                                                    to_params=to_params,
                                                    plot_progress = True,
                                                    print_progress = True,
                                                    maxIterations=150,
                                                    debug = debug)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {nDof}, vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"	


	fe_solver.plot_mesh(title = title, save_path = None)
	plt.close('all')  
	fig, ax1 = plt.subplots()

    # Plot compliance on left y-axis
	ax1.set_xlabel('Iterations')
	ax1.set_ylabel('objective', color='tab:blue')
	ax1.plot(history['objective'], color='tab:blue', label='objective')
	ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot volume fraction on right y-axis with dotted line
	ax2 = ax1.twinx()
	ax2.set_ylabel('Volume Fraction', color='tab:orange')
	ax2.plot(history['volfrac'], color='tab:orange', linestyle=':', label='Volume Fraction')
	ax2.tick_params(axis='y', labelcolor='tab:orange')
	ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

	plt.title('OC: Volume and Compliance vs. Iterations')

    # Add legend
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2)

	plt.grid(True)
	plt.show()


def topopt_levelset(feaMode,
                    fe_solver,
                    to_params,
                    maxIterations: int = 150,
                    numReinit: int = 5,
                    numHolesAlongEachDimension: int = 4,
                    objective_tol: float = 0.005,
                    constraint_tol: float = 0.001,
                    void: float = 0.0001,
                    plot_progress: bool = False,
                    print_progress: bool = False,
                    plotter=None,
                    debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    Level Set Method for Topology Optimization.
    
    References:
    """
    
    def evolveUpWind(mesh, lsf, v, maxVolChange=0.02):
        """
        Evolution of level set function.
        Implements Hamilton-Jacobi equation:
            ∂ψ/∂t = -v|∇ψ| - ω*g
        """
        # CFL time step 
        h = np.min(mesh.elem_size)
        max_v = np.abs(v).max()
        if max_v < 1e-10:
            rho = (lsf < 0).astype(float)
            return rho, lsf
        
        cflLimit = 0.25
        dt = cflLimit * h / max_v
        max_steps = 100 
        vol_current = np.mean(lsf < 0)
        lsf_prev = lsf.copy()
        for _ in range(max_steps):
            lsf = upwind_step(mesh, lsf, v, dt)
            vol_est = np.mean(lsf < 0)
            diff = np.max(np.abs(lsf - lsf_prev))
            if diff == 0: # no change
                break
            lsf_prev = lsf.copy()
            vol_diff = np.abs(vol_est - vol_current)
            if (vol_diff > maxVolChange): # do not let volume change too much in one evolve
                break
        return  lsf

    def upwind_step(mesh, lsf, v, dt):
        """
        Godunov upwind scheme for Hamilton-Jacobi equations using first-order differences.
        
        Implements the upwind finite difference scheme for evolving the level set equation:
            ∂φ/∂t + v|∇φ| = 0
        
        Uses face neighbors only (6-connected in 3D) and selects upwind direction based on
        the sign of velocity to ensure stability and entropy satisfaction.
        
        Args:
            mesh: Hexahedral mesh object
            lsf: Current level set function values
            v: Velocity field (negative of shape sensitivity)
            dt: Time step (CFL condition enforced)
        
        Returns:
            np.ndarray: Updated level set function values
        
        References:
            Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent 
            speed: Algorithms based on Hamilton-Jacobi formulations. Journal of Computational 
            Physics, 79(1), 12-49.
            
            Sethian, J. A. (1999). Level Set Methods and Fast Marching Methods: Evolving 
            Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and 
            Materials Science. Cambridge University Press.
            
            Osher, S., & Fedkiw, R. (2003). Level Set Methods and Dynamic Implicit Surfaces. 
            Springer. (Chapter 6: Hamilton-Jacobi Equations)
        """
        neighbors = mesh.elemNeighborsArray
        
        # Face neighbor indices (from your 27-neighbor ordering)
        idx_xminus, idx_xplus = 12, 14
        idx_yminus, idx_yplus = 10, 16
        idx_zminus, idx_zplus = 4, 22
        
        # Get neighbor LSF values with FIRST-ORDER EXTRAPOLATION for boundaries
        def get_neighbor_lsf(idx, opp_idx):
            neighbor_id = neighbors[:, idx]
            opposite_id = neighbors[:, opp_idx]
            
            result = lsf.copy()
            valid = neighbor_id >= 0
            
            # Use actual neighbor where available
            result[valid] = lsf[neighbor_id[valid]]
            
            # For domain boundaries, extrapolate from opposite direction
            # This allows boundary to move freely
            boundary = ~valid
            has_opposite = opposite_id >= 0
            can_extrapolate = boundary & has_opposite
            
            # Linear extrapolation: φ_boundary = 2*φ_center - φ_opposite
            result[can_extrapolate] = 2*lsf[can_extrapolate] - lsf[opposite_id[can_extrapolate]]
            
            return result
        
        lsf_xm = get_neighbor_lsf(idx_xminus, idx_xplus)
        lsf_xp = get_neighbor_lsf(idx_xplus, idx_xminus)
        lsf_ym = get_neighbor_lsf(idx_yminus, idx_yplus)
        lsf_yp = get_neighbor_lsf(idx_yplus, idx_yminus)
        lsf_zm = get_neighbor_lsf(idx_zminus, idx_zplus)
        lsf_zp = get_neighbor_lsf(idx_zplus, idx_zminus)
            
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
    
        lsf_new = lsf - dt * v * grad_mag
        return lsf_new

    if (feaMode == FEA_MODE.STRUCTURAL):
        nDOFPerNode = 3
    else:
        nDOFPerNode = 1
    
    tStart = time.time()
    material_model = MaterialModel.SIMP  # Not really used since rho is 0/1
    mesh = fe_solver.mesh

    #fe_solver.plot_mesh(title="Initial Design", save_path=None) 
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


    initialVolfraction =   min([volFractionConstraint+0.5,0.9]) # heuristic for initial vol fraction
    # Initialize level set function (start with full domain)
    rho = mesh.initialize_with_holes(initialVolfraction,holes_per_dim= numHolesAlongEachDimension, distance_type=distanceType)
    # Ensure load-bearing elements are solid
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)
    if elemsWithForces is not None and len(elemsWithForces) > 0:
        rho[elemsWithForces] = 1
    if (to_params.ElemsToKeep is not None):
        rho[to_params.ElemsToKeep] = 1

    lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)
    
    # History tracking
    history = {'objective': [], 'volfrac': []}
    success = True
    errorMsg = "No errors."
    
    # Main optimization loop 
    for iteration in range(maxIterations):
        mesh.setPseudoDensity(np.asarray(rho))
        if plot_progress:
            fe_solver.plot_pseudo_density_realtime(
                title=f"Iter {iteration }",
                iteration = iteration,
                external_plotter=plotter
            )

        sol = fe_solver.solve(rho, material_model)
        obj, shapeSens = compute_compliance_and_sensitivity(feaMode, rho, fe_solver,void)
        if (iteration == 0):
             obj0 = obj
       
        # Scaling of sensitivities; this will alow us to set stepLength and topWeight independent of problem scale
        if iteration == 0:
            shapeScaling = np.max(np.abs(shapeSens)) + 1e-12
   
        rho_anstaz = void + (1 - void) * rho
        shapeSens = rho_anstaz * shapeSens / shapeScaling
       
        #  Load bearing elements must remain solid 
        if elemsWithForces is not None and len(elemsWithForces) > 0:
            shapeSens[elemsWithForces] = min(shapeSens)
         
        if (to_params.ElemsToKeep is not None):
            shapeSens[to_params.ElemsToKeep] = min(shapeSens) # also retain elements that are in the keep list
    
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
        if iteration > 5 and obj < 100*obj0:
            obj_err = abs(history['objective'][-1] - history['objective'][-2]) / abs(history['objective'][-2])
            vol_err = abs(volCurr - volFractionConstraint)  
            if obj_err < objective_tol and vol_err < constraint_tol:
                if print_progress:
                    print("Convergence criteria met.")
                break
        
        # 7.  Lagrangian parameters 
        vol_error = volCurr - volFractionConstraint
        if iteration == 0:
            la = -0.01
            La = 1000
            alpha = 0.95
            maxVolChange = 0.05
        else:
            la = la - 1/La * vol_error; 
            La  = alpha * La
            maxVolChange = max(0.01, alpha * maxVolChange)

  
        vol_penalty_term = la - 1/La*vol_error
 
        shapeSens = shapeSens - vol_penalty_term

        # Smooth the sensitivities 
        shapeSens_smooth = (H @ shapeSens) / Hs
        #fe_solver.plot_elem_field(shapeSens_smooth, title=f"Shape Sensitivity - Iter {iteration}", save_path=None)
        # Design update via evolution 
        lsf = evolveUpWind(
                mesh=mesh,
                lsf=lsf,
                v=-shapeSens_smooth,  # Velocity is negative of shape sensitivity
                maxVolChange = maxVolChange
        )
   
        rho = (lsf < 0).astype(float)

        # Periodic reinitialization 
        if (iteration + 1) % numReinit == 0:
            if print_progress:
                print("  Reinitializing level set function...")
            lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)

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


if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.TensilePlate # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
     
	run_topopt_levelset(to_problem)