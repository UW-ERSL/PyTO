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
                                                    debug = debug)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, time: {timeTaken:.0f} s"	


	fe_solver.plot_mesh(title = title, save_path = None)


def initialize_with_holes(mesh, volFractionConstraint, num_holes=8, 
                         distance_type=DISTANCE_TYPE.DISTANCE_3D):
    """
    Initialize design with holes distributed throughout the domain.
    
    Handles different distance types properly:
    - DISTANCE_3D: Full 3D spherical holes
    - DISTANCE_XY: 2D circular holes (through-holes in Z direction)
    - DISTANCE_XZ: 2D circular holes (through-holes in Y direction)
    - DISTANCE_YZ: 2D circular holes (through-holes in X direction)
    
    Args:
        mesh: HexMesher object
        volFractionConstraint: Target volume fraction (e.g., 0.5)
        num_holes: Number of holes to create (default: 8 for 2x2x2)
        distance_type: Type of distance function
        
    Returns:
        rho: Initial density field with holes
    """
    
    # Start with full solid
    rho = np.ones(mesh.num_elems)
    
    # Get element centers
    # Get element centers for hole placement
    elem_centers = mesh.elem_centers  # Shape: (num_elems, 3)
    
    # Get domain bounds from NODES (not element centers!)
    # This is critical for extruded geometries
    node_xyz = mesh.node_xyz  # Shape: (num_nodes, 3)
    # Get domain bounds
    x_min, x_max = node_xyz[:, 0].min(), node_xyz[:, 0].max()
    y_min, y_max = node_xyz[:, 1].min(), node_xyz[:, 1].max()
    z_min, z_max = node_xyz[:, 2].min(), node_xyz[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Margin from boundaries (don't put holes too close to edges)
    margin = 0.15  # 15% margin
    
    # Determine hole distribution based on distance type
    if distance_type == DISTANCE_TYPE.DISTANCE_3D:
        # Full 3D problem - distribute holes in 3D grid
        # For num_holes=8: 2x2x2 grid
        # For num_holes=27: 3x3x3 grid
        n_per_dim = int(np.ceil(num_holes ** (1/3)))
        
        # Create 3D grid of hole centers
        x_positions = np.linspace(x_min + margin*x_range, 
                                 x_max - margin*x_range, n_per_dim)
        y_positions = np.linspace(y_min + margin*y_range, 
                                 y_max - margin*y_range, n_per_dim)
        z_positions = np.linspace(z_min + margin*z_range, 
                                 z_max - margin*z_range, n_per_dim)
        
        # Compute hole radius to achieve target volume
        # Volume of sphere: (4/3)πr³
        domain_volume = x_range * y_range * z_range
        vol_to_remove = domain_volume * (1.0 - volFractionConstraint) * 0.9  # 90% via holes
        vol_per_hole = vol_to_remove / (n_per_dim**3)
        hole_radius = (3 * vol_per_hole / (4 * np.pi)) ** (1/3)
        
        # Create spherical holes
        for xc in x_positions:
            for yc in y_positions:
                for zc in z_positions:
                    # Distance from hole center (3D Euclidean)
                    dist = np.sqrt((elem_centers[:, 0] - xc)**2 + 
                                  (elem_centers[:, 1] - yc)**2 + 
                                  (elem_centers[:, 2] - zc)**2)
                    
                    # Remove material inside hole
                    rho[dist < hole_radius] = 0.0
        

    elif distance_type == DISTANCE_TYPE.DISTANCE_XY:
        # 2D in XY plane, extruded in Z
        # Create cylindrical through-holes spanning entire Z
        n_per_dim = int(np.ceil(np.sqrt(num_holes)))
        
        # Create 2D grid of hole centers in XY plane
        x_positions = np.linspace(x_min + margin*x_range, 
                                 x_max - margin*x_range, n_per_dim)
        y_positions = np.linspace(y_min + margin*y_range, 
                                 y_max - margin*y_range, n_per_dim)
        
        # Compute hole radius for cylinders
        # Volume of cylinder: πr²h where h = z_range
        domain_volume = x_range * y_range * z_range
        vol_to_remove = domain_volume * (1.0 - volFractionConstraint) * 0.9
        vol_per_hole = vol_to_remove / (n_per_dim**2)
        hole_radius = np.sqrt(vol_per_hole / (np.pi * z_range))
        
        # Create cylindrical through-holes
        for xc in x_positions:
            for yc in y_positions:
                # Distance in XY plane only (ignore Z)
                dist_xy = np.sqrt((elem_centers[:, 0] - xc)**2 + 
                                 (elem_centers[:, 1] - yc)**2)
                
                # Remove material inside cylinder (all Z levels)
                rho[dist_xy < hole_radius] = 0.0
        
 
    elif distance_type == DISTANCE_TYPE.DISTANCE_XZ:
        # 2D in XZ plane, extruded in Y
        # Create cylindrical through-holes spanning entire Y
        n_per_dim = int(np.ceil(np.sqrt(num_holes)))
        
        x_positions = np.linspace(x_min + margin*x_range, 
                                 x_max - margin*x_range, n_per_dim)
        z_positions = np.linspace(z_min + margin*z_range, 
                                 z_max - margin*z_range, n_per_dim)
        
        domain_volume = x_range * y_range * z_range
        vol_to_remove = domain_volume * (1.0 - volFractionConstraint) * 0.9
        vol_per_hole = vol_to_remove / (n_per_dim**2)
        hole_radius = np.sqrt(vol_per_hole / (np.pi * y_range))
        
        for xc in x_positions:
            for zc in z_positions:
                # Distance in XZ plane only (ignore Y)
                dist_xz = np.sqrt((elem_centers[:, 0] - xc)**2 + 
                                 (elem_centers[:, 2] - zc)**2)
                
                rho[dist_xz < hole_radius] = 0.0
        
  
    elif distance_type == DISTANCE_TYPE.DISTANCE_YZ:
        # 2D in YZ plane, extruded in X
        # Create cylindrical through-holes spanning entire X
        n_per_dim = int(np.ceil(np.sqrt(num_holes)))
        
        y_positions = np.linspace(y_min + margin*y_range, 
                                 y_max - margin*y_range, n_per_dim)
        z_positions = np.linspace(z_min + margin*z_range, 
                                 z_max - margin*z_range, n_per_dim)
        
        domain_volume = x_range * y_range * z_range
        vol_to_remove = domain_volume * (1.0 - volFractionConstraint) * 0.9
        vol_per_hole = vol_to_remove / (n_per_dim**2)
        hole_radius = np.sqrt(vol_per_hole / (np.pi * x_range))
        
        for yc in y_positions:
            for zc in z_positions:
                # Distance in YZ plane only (ignore X)
                dist_yz = np.sqrt((elem_centers[:, 1] - yc)**2 + 
                                 (elem_centers[:, 2] - zc)**2)
                
                rho[dist_yz < hole_radius] = 0.0
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
    

    return rho



def topopt_levelset(feaMode,
                    fe_solver,
                    to_params,
                    maxIterations: int = 250,
                    stepLength: int = 3,
                    numReinit: int = 5,
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


    # Initialize level set function (start with full domain)
    rho = np.ones(mesh.num_elems)
    rho = initialize_with_holes(mesh, 0.75 ,
                                num_holes=20, 
                                distance_type=distanceType)
     # Identify load-bearing elements
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)
    
    if elemsWithForces is not None and len(elemsWithForces) > 0:
            rho[elemsWithForces] = 1
    if (to_params.ElemsToKeep is not None):
            rho[to_params.ElemsToKeep] =1
    lsf = mesh.compute_signed_distance_function(rho, distance_type=distanceType)


    fe_solver.mesh.setPseudoDensity(np.asarray(rho))
    
   
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
        #input("Press Enter to continue...")
        sol = fe_solver.solve(rho, material_model)
        fe_solver.postprocess()
        obj, shapeSens = compute_compliance_and_sensitivity(feaMode, rho, fe_solver)
        if (iteration == 0):
             obj0 = obj
       
        # Scaling of sensitivities; this will alow us to set stepLength and topWeight independent of problem scale
        if iteration == 0:
            shapeScaling = np.max(np.abs(shapeSens)) + 1e-12
   
        shapeSens = shapeSens / shapeScaling
   
        # 3. Load bearing elements must remain solid 
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
        else:
            la = la - 1/La * vol_error; 
            La= alpha * La

        vol_penalty_term =  la - 1/La*vol_error
        shapeSens = shapeSens - vol_penalty_term

        #input("Press Enter to continue...")
        # 1. Smooth the sensitivities 
        shapeSens_smooth = (H @ shapeSens) / Hs
    
        # 4. Design update via evolution 
        rho, lsf = evolveUpWind(
                mesh=mesh,
                lsf=lsf,
                v=-shapeSens_smooth,  # Velocity is negative of shape sensitivity
                stepLength=stepLength,
        )

        # 10. Periodic reinitialization 
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

def evolveUpWind(mesh, lsf, v, stepLength):
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
    
    increment = 0.1
    dt = increment * h / max_v
 
    # Evolve for total time stepLength * CFL value 
    num_steps = int( stepLength/increment)

    for _ in range(num_steps):
        lsf = upwind_step(mesh, lsf, v, dt)
  
    rho = (lsf < 0).astype(float)
    return rho, lsf


def upwind_step(mesh, lsf, v, dt):
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
    lsf_new = lsf - dt * v * grad_mag
    return lsf_new

if __name__ == "__main__":    
	
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.Mitchell_1 # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem
     
	run_topopt_levelset(to_problem)