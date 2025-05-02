"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from scipy.ndimage import distance_transform_edt
import time

def topopt_levelset(fe_solver: hex_structural_fea.HexStructuralFEA,
                    to_params,
                    maxIterations: int = 250,
                    volfrac: float = 0.5,
                    time_step: float = 0.1,
                    numReinit: int = 2,
                    topWeight: float = 3.0,
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
    nx, ny, nz = fe_solver.mesh.grid

    # Initialize level set function and design variables
    lsf = reinit(np.ones((nx, ny, nz)),(nx,ny,nz))
    rho = np.ones((fe_solver.mesh.num_elems))
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

        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)
        # Compute topological sensitivity 
        fe_solver.postprocess()
        topSens = computeTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio,fe_solver.strainComponents,fe_solver.stressComponents,rho)
        fe_solver.plot_elem_field(shapeSens, title = f"Density: Iteration {iterNum}", save_path = None)

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
        topSens = topSens + np.pi * (la - 1 / La * (volCurr - to_params.DesiredVolFraction))
        ss3D=convert_rho(shapeSens, mesh.elem_centers, mesh.elem_size, (nx,ny,nz))
        ts3D=convert_rho(topSens, mesh.elem_centers, mesh.elem_size, (nx,ny,nz))
        [rho,lsf] = evolve(-ss3D, ts3D * (lsf[1:-1, 1:-1, 1:-1] < 0), lsf, time_step, topWeight,(nx,ny,nz))
        if iterNum % numReinit == 0:
            lsf = reinit(rho,( nx,ny,nz))
		# Update level set function using Hamilton-Jacobi equation
        # [rho,lsf] = update_step(lsf, shapeSens, topSens, time_step, topWeight)
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['compliance'][-1]:.4f}, Final Volume: {history['volume'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s")

    return u, history

def evolve(v, g, lsf, step_length, w,grid):
    nx,ny,nz=grid
    v_full = np.zeros(np.array(v.shape) + 2)
    v_full[1:-1, 1:-1,1:-1] = v.copy()

    g_full = np.zeros(np.array(g.shape) + 2)
    g_full[1:-1, 1:-1,1:-1] = g.copy()
    # print("Shape of v_full:", v_full)

    dt = 0.1 / np.max(np.abs(v))
    print("dt", dt)
    for _ in range(int(10 * step_length)):
        #
        # Approximate gradient using nearest neighbors in 3D
        dpx = np.roll(lsf, -1, axis=1) - lsf 
        dmx = lsf - np.roll(lsf, 1, axis=1)
        dpy = np.roll(lsf, -1, axis=0) - lsf
        dmy = lsf - np.roll(lsf, 1, axis=0)
        dpz = np.roll(lsf, -1, axis=2) - lsf
        dmz = lsf - np.roll(lsf, 1, axis=2)
        grad_plus = np.sqrt(np.maximum(dmx, 0)**2 + np.maximum(dmy, 0)**2 + np.maximum(dmz, 0)**2+np.minimum(dpx, 0)**2 + np.minimum(dpy, 0)**2 + np.minimum(dpz, 0)**2)
        grad_minus = np.sqrt(np.maximum(dpx, 0)**2 + np.maximum(dpy, 0)**2 + np.maximum(dpz, 0)**2+np.minimum(dmx, 0)**2 + np.minimum(dmy, 0)**2 + np.minimum(dmz, 0)**2)
        lsf -= dt * (np.minimum(v_full, 0) * grad_minus + np.maximum(v_full, 0) * grad_plus + w * g_full)
    rhofull = (lsf.copy() < 0).astype(np.float64)
    # print("Shape of rhofull:", rhofull.shape)
    # print("Data type of rhofull:", rhofull.dtype)
    rho = rhofull[1:-1, 1:-1, 1:-1].copy()
	# Count the number of elements in lsf that are less than 0
    num_elements_less_than_zero = np.count_nonzero(lsf[1:-1, 1:-1, 1:-1] < 0)
    # Print the result
    print("Number of elements in lsf < 0:", num_elements_less_than_zero)
    print("Number of non-zero elements in rho:", np.count_nonzero(rho))
    # print(rho)
    # print("Shape of rho:", rho.shape)
    # print("Data type of rho:", rho.dtype)
    rho=convert_rho(rho.copy(), fe_solver.mesh.elem_centers, fe_solver.mesh.elem_size, (nx,ny,nz))
    return rho, lsf

def reinit(rho,grid):
    if rho.ndim == 1:
        rho=convert_rho(rho.copy(),fe_solver.mesh.elem_centers, fe_solver.mesh.elem_size, grid)
    
    # Create an expanded array with padding
    # rho=convert_rho_to_3d(rho, np.zeros((rho.shape[0], 3)), 1, rho.shape[0], rho.shape[1], rho.shape[2])
    rho_full = np.zeros(np.array(rho.shape) + 2, dtype=int)
    rho_full[1:-1, 1:-1, 1:-1] = rho.copy()

    # Compute the level set function
    lsf = (~rho_full.astype(bool)) * (distance_transform_edt(rho_full==0) - 0.5) \
        - rho_full * (distance_transform_edt((rho_full - 1)==0) - 0.5)
    print("Shape of lsf:", lsf.shape)
    print("Data type of lsf:", lsf.dtype)

    return lsf
		
def convert_rho(rho, element_centers, element_size, grid):
    """
    Converts rho between 1D and 3D formats based on its shape.

    Parameters:
    rho (np.array): Either a 1D or 3D array of density values.
    element_centers (np.array): Nx3 array of element center coordinates.
    element_size (float): Fixed size of each element (assuming cubic elements).
    nx, ny, nz (int): Dimensions of the 3D structured grid.

    Returns:
    np.array: The transformed rho array (1D if input is 3D, 3D if input is 1D).
    """
    nx,ny,nz=grid
    dx,dy,dz=element_size
    if rho.ndim == 1:  # Convert 1D to 3D
        rho_3d = np.zeros((nx, ny, nz))
        for idx, (x, y, z) in enumerate(element_centers):
            i = int(x // dx)
            j = int(y // dy)
            k = int(z //dz)
            rho_3d[i, j, k] = rho[idx]
        return rho_3d

    elif rho.ndim == 3:  # Convert 3D to 1D
        rho_1d = np.zeros(len(element_centers))
        for idx, (x, y, z) in enumerate(element_centers):
            i = int(x // dx)
            j = int(y // dy)
            k = int(z // dz)
            rho_1d[idx] = rho[i, j, k]
        return rho_1d

    else:
        raise ValueError("rho must be either a 1D or 3D NumPy array")
    
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
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

	dsolver = deflation.DeflationSolver()
    
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
                                                    volfrac = 0.5,
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