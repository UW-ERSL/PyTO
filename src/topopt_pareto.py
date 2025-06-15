from topopt_common import *
import time
import numpy as np
def topopt_pareto(fe_solver,
				  to_params,
							rel_err: float = 0.02,
							vol_decr_max: float = 0.05,
							vol_decr_min: float = 0.0025,
							min_local_iters: int = 2,
							max_local_iters: int = 5,
							xVoid: float = 0,
							print_progress: bool = True,
							plot_progress: bool = False,
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
	nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
	tStart = time.time()
	
	removeHangingElems = to_params.RemoveHangingElems
	if fe_solver.elem_body_force is not None and (np.linalg.norm(fe_solver.elem_body_force) > 0) and not removeHangingElems:
		removeHangingElems = True #For body forces, must remove hanging elements in Pareto

	totalIter = 1

	# Initialize design field
	x = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'objective': [],'compliance':[], 'volume': []}
	if (print_progress):
		print("Computing Filters ...")
	[H,Hs] = createFilters(fe_solver, to_params)

	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,nDOFPerNode)

	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None

	
	if isinstance(fe_solver.mat_prop, list): # multiple materials
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp.youngs_modulus,mp.poissons_ratio,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal( mp.thermal_conductivity,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]	
		print("Assuming all elements have the same material properties")
	else: # single material
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
	sol = fe_solver.solve(x)
	fe_solver.postprocess()
	nFEAs = 1

	obj, T,compliance = compute_objective_topological_sensitivity_compliance(to_params,sol,x, fe_solver,KE)
	J = obj

	history['objective'].append( obj)# may be the same as compliance
	history['compliance'].append(compliance) 
	history['volume'].append(volfrac)
	
	# Add contribution from body force to topological sensitivity if present
	if (nodal_body_force is not None):
		T_body = np.zeros(fe_solver.mesh.num_elems)
		for elem in range(fe_solver.mesh.num_elems):
			edof = fe_solver.mesh.edofMat[elem]
			T_body[elem] =  (x[elem]*sol[edof] * nodal_body_force[edof]).sum()
		T += 2*T_body

	if (elemsWithForces.size > 0): #For pure body forces, this may be empty
		T[elemsWithForces] = np.max(T)
	if (to_params.ElemsToKeep is not None):
		T[to_params.ElemsToKeep] = np.max(T)
	T = (H * T) / Hs

	T /= np.max(np.abs(T))  # Normalize sensitivity

	if (print_progress):
		print(f"vf={history['volume'][-1]:.3f}, obj={history['objective'][-1]:.3g}, compliance={history['compliance'][-1]:.3g}, #FEA={nFEAs:2d}")
	vol_decr = vol_decr_max
	
	success = True
	terminatePareto = False
	errorMsg = "None"
	# Observation: Damping using the previous sensitivity values avoids getting trapped in local minima
	wtDamping = 0.5 # 0 means full wt to current T values, else previous T values are damped in
	nSmoothSteps = 2 # Number of smoothing steps to apply
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		raise ValueError(f"Unknown constraint type: {constraintType}")
	
	while volfrac > volFractionConstraint:
		if (plot_progress):
			fe_solver.plot_mesh(plot_bc = False,auto_close = False, title = f'Volfrac: {volfrac:0.3f}')
		# Move to next volume fraction
		volfrac = max(volFractionConstraint, volfrac - vol_decr)
		if (debug):
			print("-" * 50)
			print(f"Attempting v={volfrac:.3f}")
		# Initialize local iteration variables
		localIter = 0
		JTemp = history['objective'][-1]  # Store previous value
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
			if (localIter >= max_local_iters) or abs(JTemp) > 10 * history['objective'][-1]:  # large change in compliance	
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
					print(f"Reducing vol_decr to: {vol_decr:.5g}")
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

			sol = fe_solver.solve(x)
			fe_solver.postprocess()
			nFEAs += 1
			obj, TTemp,compliance = compute_objective_topological_sensitivity_compliance(to_params,sol,x, fe_solver,KE)
			JTemp = obj  # Update current objective value
			if (to_params.Objective[0] == TO_QOI.COMPLIANCE):
				T = TTemp.copy()  # Use current sensitivity for compliance objective
			else:
				# If x = 0, use previous sensitivity, else use current sensitivity
				T = np.where(x == 0, TPrev.copy(), TTemp.copy())
			# Add contribution from body force to topological sensitivity if present
			if (nodal_body_force is not None):
				T_body = np.zeros(fe_solver.mesh.num_elems)
				for elem in range(fe_solver.mesh.num_elems):
					edof = fe_solver.mesh.edofMat[elem]
					T_body[elem] =  (x[elem]*sol[edof] * nodal_body_force[edof]).sum()
				T += 2*T_body

			for _ in range(nSmoothSteps):
				T = (H * T) / Hs

			T /= np.max(np.abs(T))  # Normalize sensitivity
			T = ((1-wtDamping)*T + wtDamping*TPrev)  # Damping

			if (elemsWithForces.size > 0):
				T[elemsWithForces] = np.max(T)

			if (to_params.ElemsToKeep is not None):
				T[to_params.ElemsToKeep] = np.max(T)

			localIter += 1
			totalIter += 1
			
		if terminatePareto:
			if (volfrac > 1.1*volFractionConstraint):
				success = False
				errorMsg =  f"vf {volFractionConstraint:0.3f} not reached"
				print("-" * 50)
				print("Pareto: Failed to reach volume fraction.")
				print("1. Check for incorrect symmetry constraints")
				print("2. Increase mesh size")
			break
		if innerLoopSuccess:
			meshComponents = fe_solver.mesh.find_connected_components()
			if (len(meshComponents) > 1):
				# Find the largest connected component and its size
				largest_component = max(meshComponents, key=len)
				# Set density to xVoid for all elements
				x[:] = xVoid
				# Set density to 1 for elements in largest component
				x[list(largest_component)] = 1.0
				fe_solver.mesh.setPseudoDensity(x.flatten())
				volfrac = np.mean(x)
			history['objective'].append(obj)
			history['compliance'].append(compliance)
			history['volume'].append(volfrac)
			scale = history['objective'][-1] / history['objective'][0]
			vol_decr = max(vol_decr_min,min(vol_decr,vol_decr_max/scale)) # Reduce volume increment for steep increase in compliance
			if (print_progress):
				print(f"vf={history['volume'][-1]:.3f}, obj={history['objective'][-1]:.3g}, compliance={history['compliance'][-1]:.3g}, #FEA={nFEAs:2d}")
			fe_solver.mesh.setPseudoDensity(x.flatten())


	totalTime = time.time() - tStart

	print(f"Final: vf={history['volume'][-1]:.3f}, obj={history['objective'][-1]:.3g}, compliance={history['compliance'][-1]:.3g}, #FEA={nFEAs:2d}")
	print(f"Total Time: {totalTime:.2f} s")
	print("Error: ", errorMsg)
	return sol, history, success,errorMsg,nFEAs


if __name__ == "__main__":    
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.Multiload # Choose the TO problem
	#to_problem = ThermalTOExamples.BridgeThermal # Choose the TO problem

	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem,nDOFDesired=50000)
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	
	debug = False

	solver = lin_solv.Solvers.PARDISO
	dsolver = deflation.DeflationSolver()
	if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):#  # Choose solver. Typically PARDISO, but DPCG for large DOF problems
		solver = lin_solv.Solvers.DPCG
		# DPCG solver is used for large DOF problems
		# Create deflation solver object
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		#dsolver.plot_deflation_groups(mesh)
		dsolver.create_deflation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]
	 
	
	if (to_problem in StructuralTOExamples):
		fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					elem_body_force = elem_body_force)
	elif (to_problem in ThermalTOExamples):
		fe_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					elem_body_force = elem_body_force)

	
	print('Solver: ', fe_solver.solver.name)
	print("nNodes: ", fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	#print("Close the plot to continue...")
	title = f'nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#fe_solver.plot_mesh(title = title, save_path = None)
	
	startTime = time.time()

	print("OptimizationMethod: Pareto")
	sol, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver = fe_solver,
									to_params = to_params,
									plot_progress= True,
									debug = debug)
	
	timeTaken = time.time() - startTime
	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")

	title = f"Pareto: vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {nFEAs:3d}, time: {timeTaken:.0f} s"
	fe_solver.plot_mesh(title = title, save_path = None)

		
	# Plot volume vs compliance history
	plt.figure()
	plt.plot(history['volume'], history['objective'], marker='o')
	plt.xlabel('Volume Fraction')
	plt.ylabel('objective')
	plt.title('Pareto: Volume vs Compliance History')
	plt.grid(True)
	plt.show()
	
	
