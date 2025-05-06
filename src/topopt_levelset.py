"""Optimization routines for topology optimization."""

from topopt_common import *
from topopt_material_model import *
from scipy.ndimage import distance_transform_edt
import time

HXD = HYD = HZD = None

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
    global  HXD,HYD,HZD

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

    # print("Shape of lsf:", lsf)
    fe_solver.plot_elem_field(lsf, title = f"Sdf", save_path = None,cross_section= {'axis': 'z', 'position': 0})
   
    # lsf_xD = HXD@lsf.T
    # fe_solver.plot_elem_field(lsf_xD, title = f"Sdf_xgrad", save_path = None,cross_section= {'axis': 'z', 'position': 0})

    # lsf_yD = HYD@lsf.T
    # fe_solver.plot_elem_field(lsf_yD, title = f"Sdf_ygrad", save_path = None,cross_section= {'axis': 'z', 'position': 0})

    # lsf_zD = HZD@lsf.T
    # fe_solver.plot_elem_field(lsf_zD, title = f"Sdf_zgrad", save_path = None,cross_section= {'axis': 'z', 'position': 0})

    # gradMag = GradientMagnitude(lsf)
    # fe_solver.plot_elem_field(gradMag, title = f"GradientMagnitude", save_path = None,cross_section= {'axis': 'z', 'position': 0})

    # gradMagSmooth = H*gradMag/Hs
    # fe_solver.plot_elem_field(gradMagSmooth, title = f"GradientMagnitudeSmooth", save_path = None,cross_section= {'axis': 'z', 'position': 0})
    
    shapeSens = np.zeros((fe_solver.mesh.num_elems))
    topSens = np.zeros((fe_solver.mesh.num_elems))

    history = {'compliance': [], 'volume': []}

    # Material properties
    mat_prop = fe_solver.mat_prop
    # if (print_progress):
    
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
        # print("Shape of rho:", rho.shape)
        # print("Data type of rho:", rho.dtype)
        comp,u = compliance(rho, fe_solver)
        shapeSens =(-rho)* (np.dot(u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(fe_solver.mesh.num_elems, 24)).sum(1)
        shapeSens = (H * shapeSens)/Hs
        
        shapeSens /= np.max(np.abs(shapeSens))
        # fe_solver.plot_elem_field(shapeSens, title = f"ShapeSensitivity: Iteration {iterNum}", save_path = None)
        if (elemsWithForces.size > 0):
           shapeSens[elemsWithForces] = min(shapeSens)

        if (to_params.ElemsToKeep is not None):
           shapeSens[to_params.ElemsToKeep] = min(shapeSens)
        # Compute topological sensitivity 
        fe_solver.postprocess()

        print("Compliance: ", comp)
        # print("Complianceold: ", compold)
        volCurr = np.mean(rho)
        history['compliance'].append(comp)
        history['volume'].append(volCurr)
        print(f"Iter: {iterNum}, Compliance: {comp:.4f}, Volume: {volCurr:.3f}")
      
        # Update level set function
        if iterNum == 0:
            la = -0.01
            La = 1000
            alpha = 0.9
            # la = -0.01
            # La = 1000000
            # alpha = 0.9
        else:
            la = la - 1 / La * (volCurr - to_params.DesiredVolFraction)
            La = alpha * La
        shapeSens = shapeSens - la + (1 / La) * (volCurr - to_params.DesiredVolFraction)
        topSens= 0
       
        fe_solver.plot_elem_field(shapeSens, title = f"Sens  Iteration: {iterNum}", save_path = None)
        
        [rho,lsf] = evolve(shapeSens, topSens * (lsf< 0), lsf, time_step, topWeight)
        fe_solver.plot_elem_field(lsf, title = f"LSF Iteration: {iterNum}", save_path = None)
      
        fe_solver.mesh.setPseudoDensity(np.asarray(rho))
        #fe_solver.plot_elem_field(rho, title = f"Rho Iteration: {iterNum}", save_path = None,cross_section= {'axis': 'z', 'position': 0})

    totalTime = time.time() - tStart
    print(f"Final Compliance: {history['compliance'][-1]:.4f}, Final Volume: {history['volume'][-1]:.3f}")
    print(f"Total Time: {totalTime:.2f} s")

    return u, history

def GradientMagnitude(lsf: np.ndarray):
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
		# Compute the gradient magnitude
		# Gradient magnitude is sqrt((dx)^2 + (dy)^2 + (dz)^2)
		gradient_magnitude = np.sqrt(lsfdx**2 + lsfdy**2 + lsfdz**2)
		return gradient_magnitude

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
    # print(" np.max(np.abs(v))", np.max(np.abs(v)))
    dt = (1/N) / np.max(np.abs(v))
    print("dt", dt)
    for _ in range(int(N * step_length)):
        gradMag = GradientMagnitude(lsf)
        gradMag /= np.max(np.abs(gradMag))
        gradMagSmooth = HXD @ gradMag.T
        lsf -= dt * (gradMagSmooth*v)
        
    rho = (lsf < 0).astype(np.float64)
    return rho,lsf

    
if __name__ == "__main__":    
	
	from topopt_benchmarks import *
	# jax.config.update("jax_enable_x64", True)
	print("-" * 50)
	to_problem = StructuralTOExamples.LBracketThickMidLoad # Choose the TO problem
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
                                                    time_step = 0.1,
                                                    numReinit = 20,
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