"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from scipy.ndimage import distance_transform_edt
import time


def topopt_levelset(fe_solver: hex_structural_fea.HexStructuralFEA,
                    to_params,
                    maxIterations: int = 250,
                    numReinit: int = 10,
                    plot_progress: bool = False,
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
   
    mesh=fe_solver.mesh


    print("Computing Derivative Filters ...")
    HXD = createXDerivativeFilter(mesh)
    HYD = createYDerivativeFilter(mesh)
    HZD = createZDerivativeFilter(mesh)

    print("Computing Smoothing Filters ...")
    [H,Hs] = createFilters(fe_solver, to_params)

    # Initialize level set function and design variables
    rho = np.ones((fe_solver.mesh.num_elems))
    lsf = fe_solver.mesh.compute_signed_distance_function(rho)
    lsf /= np.max(np.abs(lsf))

    shapeSens = np.zeros((fe_solver.mesh.num_elems))
    topSens = np.zeros((fe_solver.mesh.num_elems))

    history = {'compliance': [], 'volume': []}

    # Material properties
    mat_prop = fe_solver.mat_prop
    # if (print_progress):
    
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)
   
    if isinstance(fe_solver.mat_prop, list):
        KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp,fe_solver.mesh.elem_size)
			 for mp in fe_solver.mat_prop]
        KE = KE_list[0]
        print("Density-OC: Assuming all elements have the same material properties")
    else:
        KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
    volCurr = 1.0
    volDecrementWeight = 0.1
    for iterNum in range(maxIterations):
        if (plot_progress):
            fe_solver.plot_mesh(plot_bc = False,auto_close = False, title = f'Volfrac: {volCurr:0.3f}')

        comp,u = compliance(rho, fe_solver)
        shapeSens =(-rho)* (np.dot(u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24)).sum(1)
        shapeSens = (H * shapeSens)/Hs
        shapeSens /= np.max(np.abs(shapeSens))

        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)

        # Compute topological sensitivity 
        fe_solver.postprocess()

        volCurr = np.mean(rho)
        history['compliance'].append(comp)
        history['volume'].append(volCurr)
        print(f"Iter: {iterNum}, Compliance: {comp:.4f}, Volume: {volCurr:.3f}")
        if (abs(volCurr - to_params.DesiredVolFraction) < 0.01):
             break
       
        shapeSens = shapeSens + volDecrementWeight * (volCurr - to_params.DesiredVolFraction)
       
        gradMag = GradientMagnitude(lsf,HXD,HYD,HZD)
        gradMagSmooth = H*gradMag/Hs
        lsf += (shapeSens*gradMagSmooth)
        
        rho = (lsf < 0).astype(np.float64)
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
        if (np.max(np.abs(lsf)) > 100):
            print("Reinitializing level set function")
            lsf = fe_solver.mesh.compute_signed_distance_function(rho)
        lsf /= np.max(np.abs(lsf))
        lsf = H*lsf/Hs
        volDecrementWeight += 0.0025
        

    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['compliance'][-1]:.4f}, Final Volume: {history['volume'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s")

    return u, history

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
	
	from topopt_benchmarks import *
	# jax.config.update("jax_enable_x64", True)
	print("-" * 50)
	to_problem = StructuralTOExamples.LBracketThickTopLoad # Choose the TO problem
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

	fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
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
                                                    maxIterations = 200,
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