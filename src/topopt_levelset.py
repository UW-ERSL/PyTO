"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from scipy.ndimage import distance_transform_edt
import time

forward_diffx = None
backward_diffx = None
forward_diffy = None
backward_diffy = None
forward_diffz = None
backward_diffz = None


def topopt_levelset(fe_solver: hex_structural_fea.HexStructuralFEA,
                    to_params,
                    maxIterations: int = 250,
                    time_step: float = 0.1,
                    numReinit: int = 2,
                    topWeight: float =0,
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
    tStart = time.time()
    totalIter = 1
    mesh=fe_solver.mesh
    global forward_diffx, backward_diffx, forward_diffy, backward_diffy, forward_diffz, backward_diffz
    forward_diffx = XDerivative(mesh,direction="forward")
    backward_diffx = XDerivative(mesh,direction="backward")
    forward_diffy = YDerivative(mesh,direction="forward")
    backward_diffy = YDerivative(mesh,direction="backward")
    forward_diffz = ZDerivative(mesh,direction="forward")
    backward_diffz = ZDerivative(mesh,direction="backward")


    # Initialize level set function and design variables
    rho = np.ones((fe_solver.mesh.num_elems))
    lsf = fe_solver.mesh.compute_signed_distance_function(rho)
    fe_solver.plot_elem_field(lsf, title = f"Sdf", save_path = None,cross_section= {'axis': 'x', 'position': 0.5})

    shapeSens = np.zeros((fe_solver.mesh.num_elems))
    topSens = np.zeros((fe_solver.mesh.num_elems))

    history = {'compliance': [], 'volume': []}

    # Material properties
    mat_prop = fe_solver.mat_prop
    # if (print_progress):
    #  print("Computing Filters ...")
    [H,Hs] = createFilters(fe_solver, to_params)
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)
    print("elem size: ", fe_solver.mesh.elem_size)
    if isinstance(fe_solver.mat_prop, list):
        KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp,fe_solver.mesh.elem_size)
			 for mp in fe_solver.mat_prop]
        KE = KE_list[0]
        print("Density-OC: Assuming all elements have the same material properties")
    else:
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
    for iterNum in range(maxIterations):
        # Perform FE analysis
        print("Shape of rho:", rho.shape)
        print("Data type of rho:", rho.dtype)
        comp,u = compliance(rho, fe_solver)
        shapeSens =(-rho)* (np.dot(u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24)).sum(1)
        shapeSens = (H * shapeSens)/Hs
        fe_solver.plot_elem_field(shapeSens, title = f"ShapeSensitivity: Iteration {iterNum}", save_path = None)
        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)
        # Compute topological sensitivity 
        fe_solver.postprocess()
        #topSens = computeTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio,fe_solver.strainComponents,fe_solver.stressComponents,rho)
        fe_solver.plot_elem_field(lsf, title = f"LSF: Iteration {iterNum}", save_path = None)

        # Compute objective and volume
        
        print("Compliance: ", comp)
        # print("Complianceold: ", compold)
        volCurr = np.mean(rho)
        history['compliance'].append(comp)
        history['volume'].append(volCurr)
        print(f"Iter: {iterNum}, Compliance: {comp:.4f}, Volume: {volCurr:.3f}")
        # Check for convergence
        if iterNum > 5 and abs(volCurr - to_params.DesiredVolFraction) < 0.005 and \
                all(abs(history['compliance'][-1] - np.array(history['compliance'][-5:])) < 0.01 * abs(history['compliance'][-1])):
            break
        # Update level set function
        if iterNum == 0:
            la = -0.01
            La = 1000
            alpha = 0.9
        else:
            la = la - 1 / La * (volCurr - to_params.DesiredVolFraction)
            La = alpha * La
        shapeSens = shapeSens - la + 1 / La * (volCurr - to_params.DesiredVolFraction)
        topSens= 0
        #topSens = topSens + np.pi * (la - 1 / La * (volCurr - to_params.DesiredVolFraction))
        [rho,lsf] = evolve(-shapeSens, topSens * (lsf< 0), lsf, time_step, topWeight)
        fe_solver.plot_elem_field(lsf, title = f"After evolve LSF: Iteration {iterNum}", save_path = None)

        if False:
            lsf = fe_solver.mesh.compute_signed_distance_function(rho)
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['compliance'][-1]:.4f}, Final Volume: {history['volume'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s")

    return u, history

def computeGradUpwind(lsf: np.ndarray,shapeSens: np.ndarray) -> np.ndarray:
	"""Compute the upwind gradient based on the velocity field.

	Args:
		mesh: The mesh object.
        lsf: Level set function array.
		v_full: Shape sensitivity array.

	Returns:
		grad_upwind: Upwind gradient array.
	"""
	# Compute forward and backward differences in x, y, z directions
	

	# Compute gradients
	dpx = forward_diffx @ lsf
	dmx = backward_diffx @ lsf
	dpy = forward_diffy @ lsf
	dmy = backward_diffy @ lsf
	dpz = forward_diffz @ lsf
	dmz = backward_diffz @ lsf

	grad_plus = np.sqrt(
		np.maximum(dmx, 0)**2 + np.maximum(dmy, 0)**2 + np.maximum(dmz, 0)**2 +
		np.minimum(dpx, 0)**2 + np.minimum(dpy, 0)**2 + np.minimum(dpz, 0)**2
	)
	grad_minus = np.sqrt(
		np.maximum(dpx, 0)**2 + np.maximum(dpy, 0)**2 + np.maximum(dpz, 0)**2 +
		np.minimum(dmx, 0)**2 + np.minimum(dmy, 0)**2 + np.minimum(dmz, 0)**2
	)
	grad_upwind = (
		np.minimum(shapeSens, 0) * grad_minus + np.maximum(shapeSens, 0) * grad_plus
	)

	return grad_upwind

def evolve(v,g,lsf,step_length,w):
    """Evolve the level set function using the Hamilton-Jacobi equation.

    Args:
        v: Velocity field (shape sensitivity).
        g: topological sensitivity.
        lsf: Level set function.
        step_length: Time step for the evolution.
        w: Weighting factor for the gtopological sensitivity.

    Returns:
        Updated level set function and density field.
    """
    N = 10
    dt = (1/N) / np.max(np.abs(v))
    print("dt", dt)
    for _ in range(int(N * step_length)):
        lsf -= dt * (computeGradUpwind(lsf,v))
    rhoneg = (lsf.copy() < 0).astype(np.float64)
    rho = rhoneg.copy()
    return rho,lsf

def XDerivative(mesh, direction: str = "forward") -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial x derivative.
	
	Uses forward or backward differences to approximate the derivative.
	
	Args:
		mesh: The mesh object.
		direction: Direction of the derivative, either "forward" or "backward".
	
	Returns:
		tuple containing:
			HX: Sparse matrix operator approximating the x derivative.
			HX_s: Array of row sums of HX matrix.
	"""
	if direction not in {"forward", "backward"}:
		raise ValueError("Invalid direction. Choose either 'forward' or 'backward'.")
	
	num_elems = mesh.num_elems
	dx = mesh.elem_size[0]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		if direction == "forward":
			neighbor_point = elemCenter + np.array([dx, 0, 0])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Forward difference: (f[neighbor] - f[i]) / dx
				rows.append(i)
				cols.append(i)
				data.append(-1.0 / dx)
				
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(1.0 / dx)
			else:
				# No neighbor found in the forward x direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
		elif direction == "backward":
			neighbor_point = elemCenter + np.array([-dx, 0, 0])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Backward difference: (f[i] - f[neighbor]) / dx
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(-1.0 / dx)
				
				rows.append(i)
				cols.append(i)
				data.append(1.0 / dx)
			else:
				# No neighbor found in the backward x direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
	
	HX = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HX

def YDerivative(mesh,direction: str = "forward") -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial y derivative.
	
	Uses forward or backward differences to approximate the derivative.
	
	Args:
		mesh: The mesh object.
		direction: Direction of the derivative, either "forward" or "backward".
	
	Returns:
		tuple containing:
			HY: Sparse matrix operator approximating the y derivative.
			HY_s: Array of row sums of HY matrix.
	"""
	if direction not in {"forward", "backward"}:
		raise ValueError("Invalid direction. Choose either 'forward' or 'backward'.")
	
	num_elems = mesh.num_elems
	dy = mesh.elem_size[1]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		if direction == "forward":
			neighbor_point = elemCenter + np.array([0, dy, 0])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Forward difference: (f[neighbor] - f[i]) / dy
				rows.append(i)
				cols.append(i)
				data.append(-1.0 / dy)
				
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(1.0 / dy)
			else:
				# No neighbor found in the forward y direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
		elif direction == "backward":
			neighbor_point = elemCenter + np.array([0, -dy, 0])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Backward difference: (f[i] - f[neighbor]) / dy
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(-1.0 / dy)
				
				rows.append(i)
				cols.append(i)
				data.append(1.0 / dy)
			else:
				# No neighbor found in the backward y direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
	
	HY = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HY

def ZDerivative(mesh,direction: str = "forward") -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial z derivative.
	
	Uses forward or backward differences to approximate the derivative.
	
	Args:
		mesh: The mesh object.
		direction: Direction of the derivative, either "forward" or "backward".
	
	Returns:
		tuple containing:
			HZ: Sparse matrix operator approximating the z derivative.
			HZ_s: Array of row sums of HZ matrix.
	"""
	if direction not in {"forward", "backward"}:
		raise ValueError("Invalid direction. Choose either 'forward' or 'backward'.")
	
	num_elems = mesh.num_elems
	dz = mesh.elem_size[2]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		if direction == "forward":
			neighbor_point = elemCenter + np.array([0, 0, dz])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Forward difference: (f[neighbor] - f[i]) / dz
				rows.append(i)
				cols.append(i)
				data.append(-1.0 / dz)
				
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(1.0 / dz)
			else:
				# No neighbor found in the forward z direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
		elif direction == "backward":
			neighbor_point = elemCenter + np.array([0, 0, -dz])
			neighbor_idx = mesh.get_element_near_point(neighbor_point)
			if neighbor_idx != -1:
				# Backward difference: (f[i] - f[neighbor]) / dz
				rows.append(i)
				cols.append(neighbor_idx)
				data.append(-1.0 / dz)
				
				rows.append(i)
				cols.append(i)
				data.append(1.0 / dz)
			else:
				# No neighbor found in the backward z direction; set derivative to zero.
				rows.append(i)
				cols.append(i)
				data.append(0.0)
	
	HZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HZ
    
if __name__ == "__main__":    
	
	from topopt_benchmarks import *
	# jax.config.update("jax_enable_x64", True)
	print("-" * 50)
	to_problem = StructuralTOExamples.Mitchell_1 # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem,nDOFDesired=20000)


	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	fe_solver = sfea.HexStructuralFEA(mesh = mesh,
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
	u, history = topopt_levelset(fe_solver=fe_solver,
                                                    to_params=to_params,
                                                    maxIterations = 50,
                                                    time_step = 1,
                                                    numReinit = 2,
                                                    topWeight = 0,
                                                    debug = False)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"	

	print(f"Time taken: {timeTaken:.0f} s")
	# if not success:
	# 	print(f"Error: {errorMsg}")
	fe_solver.plot_mesh(title = title, save_path = None)

	#plots.plotIsocontour(fe_solver.mesh, title = title, save_path = None)
	# Save the mesh and results