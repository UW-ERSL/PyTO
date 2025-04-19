from topopt_common import *


def topopt_pareto(fe_solver: sfea.StructFEA,
				  to_params,
							rel_err: float = 0.02,
							vol_decr_max: float = 0.05,
							vol_decr_min: float = 0.0025,
							min_local_iters: int = 2,
							max_local_iters: int = 5,
							xVoid: float = 0,
							plotIntermediateTopologies: bool = False,
							debug: bool = False
							)-> tuple[np.ndarray, dict]:
	"""Pareto method for Topology Optimization.

	Args:
		fe_solver: The structural FEA solver object.
		desiredVolFrac: The target volume fraction.
		rel_err: The relative error tolerance. Smaller values lead to more
			accurate results but require more iterations.
		vol_decr_max: The maximum volume decrease in each iteration. The step size
			for pareto tracing.
		min_local_iters: The minimum number of local iterations for each volume
			fraction.
		max_local_iters: The maximum number of local iterations for each volume
			fraction.

	Returns: A tuple containing the displacement field of the optimized structure
		and a dictionary containing the optimization history.
	"""

	tStart = time.time()
	
	removeHangingElems = to_params.RemoveHangingElems
	if fe_solver.elem_body_force is not None and (np.linalg.norm(fe_solver.elem_body_force) > 0) and not removeHangingElems:
		removeHangingElems = True #For body forces, must remove hanging elements in Pareto

	totalIter = 1

	# Initialize design field
	x = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'compliance': [], 'volume': []}
	[H,Hs] = createFilters(fe_solver, to_params)

	print("Computing element with forces ...")
	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None

	print("Initial FEA...")
	
	u = np.asarray(fe_solver.solve(x))
	nFEAs = 1
	# Store initial compliance
	history['compliance'].append(fe_solver.total_force.T @ u)
	history['volume'].append(volfrac)
	fe_solver.postprocess() # compute stresses and strains for the initial design
	# Compute initial topological sensitivity
	T = computeTopologicalSensitivity(fe_solver.mat_prop,fe_solver.strainComponents,fe_solver.stressComponents,x)
	
	# Add contribution from body force to topological sensitivity if present
	if (nodal_body_force is not None):
		T_body = np.zeros(fe_solver.mesh.num_elems)
		for elem in range(fe_solver.mesh.num_elems):
			edof = fe_solver.mesh.edofMat[elem]
			T_body[elem] =  (x[elem]*u[edof] * nodal_body_force[edof]).sum()
		T += 2*T_body

	if (elemsWithForces.size > 0): #For pure body forces, this may be empty
		T[elemsWithForces] = np.max(T)
	if (to_params.ElemsToKeep is not None):
		T[to_params.ElemsToKeep] = np.max(T)
	T = (H * T) / Hs


	print(f"vf={history['volume'][-1]:.3f}, J={history['compliance'][-1]:.3g}, #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	
	success = True
	terminatePareto = False
	errorMsg = ""
	wtDamping = 0.25 # 0 means full wt to current T values, else previous T values are damped in

	while volfrac > to_params.DesiredVolFraction:
		
		# Move to next volume fraction
		volfrac = max(to_params.DesiredVolFraction, volfrac - vol_decr)
		if (debug):
			print("-" * 50)
			print(f"Attempting v={volfrac:.3f}")
		# Initialize local iteration variables
		localIter = 0
		JTemp = history['compliance'][-1]  # Store previous value
		JPrev = JTemp  # Initialize JPrev
		JPrevPrev = JTemp # Initialize JPrevPrev
		TPrev = T.copy()  # Store previous sensitivity
		xPrev = x.copy()  # Store previous design
		innerLoopSuccess = True
		while True:
			if (debug):
				print(f"Local Iteration: {localIter}/{max_local_iters}, JTemp: {JTemp:.3g}, JPrev: {JPrev:.3g}")
			# Check convergence, and break if converged
			if localIter >= min_local_iters:
				if abs(JPrev - JTemp)/abs(JTemp) < rel_err or abs(min(JPrev,JPrevPrev) - JTemp)/abs(JTemp)  < rel_err:
					vol_frac_success = volfrac
					innerLoopSuccess = True
					break
			if (localIter >= max_local_iters) or abs(JTemp) > 10 * history['compliance'][-1]:  # large change in compliance	
				innerLoopSuccess = False
				x = xPrev.copy()
				T = TPrev.copy()
				fe_solver.mesh.setPseudoDensity(x.flatten())
				JTemp = JPrev
				volfrac = volfrac + vol_decr # Restore volume fraction
				vol_decr *= 0.75 # Reduce volume decrement
				if (debug):
					print("**Failed to converge, restoring previous design")
					print(f"Previous successful vol_frac: {vol_frac_success:.5g}")
					print(f"Decrementing vol_decr to: {vol_decr:.5g}")
				if vol_decr < vol_decr_min:
					terminatePareto = True
				break


			# Find cutoff value and update design
			value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
			x = np.ones((fe_solver.mesh.num_elems))
			x[T < value] = xVoid
			fe_solver.mesh.setPseudoDensity(x.flatten())
			if (removeHangingElems):
				meshComponents = fe_solver.mesh.find_connected_components()
				if (len(meshComponents) > 1):
					# Find the largest connected component and its size
					largest_component = max(meshComponents, key=len)
					# Set density to xVoid for all elements
					x[:] = xVoid
					# Set density to 1 for elements in largest component
					x[list(largest_component)] = 1.0
					fe_solver.mesh.setPseudoDensity(x.flatten())
			
			JPrevPrev = JPrev  # Store previous to previous value
			JPrev = JTemp  # Store previous value

			u = np.asarray(fe_solver.solve(x))
			nFEAs += 1
			JTemp = float(fe_solver.total_force.T @ u)
			#plots.plotMesh(fe_solver.mesh, bc = None, u=u, title = title)
			# Update sensitivity
			fe_solver.postprocess()
			T = computeTopologicalSensitivity(fe_solver.mat_prop,fe_solver.strainComponents,fe_solver.stressComponents,x)
		
			# Add contribution from body force to topological sensitivity if present
			if (nodal_body_force is not None):
				T_body = np.zeros(fe_solver.mesh.num_elems)
				for elem in range(fe_solver.mesh.num_elems):
					edof = fe_solver.mesh.edofMat[elem]
					T_body[elem] =  (x[elem]*u[edof] * nodal_body_force[edof]).sum()
				T += 2*T_body

			T = (H * T) / Hs
			T = ((1-wtDamping)*T + wtDamping*TPrev)  # Damping

			if (elemsWithForces.size > 0):
				T[elemsWithForces] = np.max(T)

			if (to_params.ElemsToKeep is not None):
				T[to_params.ElemsToKeep] = np.max(T)

			localIter += 1
			totalIter += 1
			
		if terminatePareto:
			if (volfrac > 1.1*to_params.DesiredVolFraction):
				success = False
				errorMsg =  f"vf {to_params.DesiredVolFraction:0.3f} not reached"
				print("-" * 50)
				print("Pareto: Failed to reach volume fraction.")
				print("1. Check for incorrect symmetry constraints")
				print("2. Increase mesh size")
			break
		if innerLoopSuccess:
			history['compliance'].append(JTemp)
			history['volume'].append(volfrac)
			scale = history['compliance'][-1] / history['compliance'][0]
			vol_decr = max(vol_decr_min,min(vol_decr,vol_decr_max/scale)) # Reduce volume increment for steep increase in compliance
			print(f"vf={history['volume'][-1]:.3f}, J={history['compliance'][-1]:.3g}, #FEA={nFEAs:2d}")
			fe_solver.mesh.setPseudoDensity(x.flatten())
	totalTime = time.time() - tStart

	print(f"Final vf: {history['volume'][-1]:.3f},  objective: {history['compliance'][-1]:.4g}")
	print(f"Total Time: {totalTime:.2f} s")
	return u, history, success,errorMsg,nFEAs


if __name__ == "__main__":    
	jax.config.update("jax_enable_x64", True)
	from topopt_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.EdgeCantilever # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.DPCG # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

	dsolver = deflation.DeflationSolver()
	# initialize the fe solver 
	if (solver == lin_solv.Solvers.DPCG):
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_delfation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	fe_solver = fea.StructFEA(mesh = mesh,
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
	fe_solver.plot_mesh(title = title, save_path = None)

	startTime = time.time()

	print("OptimizationMethod: Pareto")
	u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver = fe_solver,
									to_params = to_params,
									debug = debug)
	
	timeTaken = time.time() - startTime
	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")

	title = f"Pareto: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
	fe_solver.plot_mesh(title = title, save_path = None)

	# Plot volume vs compliance history
	plt.figure()
	plt.plot(history['volume'], history['compliance'], marker='o')
	plt.xlabel('Volume Fraction')
	plt.ylabel('Compliance')
	plt.title('Pareto: Volume vs Compliance History')
	plt.grid(True)
	plt.show(block=False)
	
	
	
	# plot other quantities over the optimized mesh
	fe_solver.plot_deformation()
	fe_solver.plot_vonMisesStress()