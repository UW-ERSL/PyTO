from topopt_common import *




def topopt_optimality_criteria(
							fe_solver: sfea.StructFEA,
							to_params,
			  				maxIterations: int = 250,
							penal: float = 3,
							move: float = 0.2,
							move_tol: float = 0.025,
							rel_conv_tol: float = 1.e-3,
							directLagrangeMethod: bool = False,
							print_progress: bool = True,
							plot_progress: bool = False,
							debug: bool = False,
							) -> tuple[np.ndarray, dict]:
	"""Optimality Criteria based topology optimization for minimum compliance.

	Args:
		fe_solver: The structural FEA solver object.
		maxIterations: Maximum number of iterations.
		volfrac: The target volume fraction.
		penal: The penalization factor for the SIMP method.
		move: The maximum change allowed for the design variables in each iteration.
		verbose: If True, prints the optimization progress.
	
	Returns: A tuple containing the displacement field of the optimized structure
		and a dictionary containing the optimization history.
	"""
	tStart = time.time()
	elem_body_force = fe_solver.elem_body_force
	if elem_body_force is None or (np.linalg.norm(elem_body_force) == 0):
		material_model = MaterialModel.SIMP #For no body forces, using SIMP material model
		material_model_dict = {'name': 'SIMP', 'penal': 3.0} # Default SIMP model
	else:
		material_model = MaterialModel.SIMPPLUS #For body forces, using SIMPPLUS material model
		material_model_dict = {'name': 'SIMPPLUS', 'penal': 3.0, 'alpha': 16} 
		#  body-force model from the papers here:
		#  https://doi.org/10.1002/nme.2499, https://doi.org/10.1016/j.cma.2017.04.021 

	num_elems = fe_solver.mesh.num_elems
	if (print_progress):
		print("Computing Filters ...")
	[H,Hs] = createFilters(fe_solver, to_params)
	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	# Initialize design variables
	x = to_params.DesiredVolFraction * np.ones(num_elems, dtype = float)
	xPhys = x.copy()

	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None
	# Initialize history
	history = {'compliance': [], 'volume': [], 'change': []}
	# OC parameters
	xmin = 0.001  # Minimum density
	xmax = 1.0    # Maximum density
	
	if isinstance(fe_solver.mat_prop, list):
		KE_list = [elem_stiff.hex8_stiffness_matrix_structural( mp,fe_solver.mesh.elem_size)
			 for mp in fe_solver.mat_prop]
		KE = KE_list[0]
		print("Density-OC: Assuming all elements have the same material properties")
	else:
		KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	success = True
	errorMsg = ""
	for iter in range(maxIterations):
		x = np.array(x)
		if (plot_progress):
			fe_solver.mesh.setPseudoDensity(x)
			fe_solver.plot_pseudo_density(auto_close = False, title = f"Iteration {iter}")
		obj,u = compliance(x, fe_solver,material_model_dict)
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		if material_model == MaterialModel.SIMP:
			# For SIMP material model: x**penal
			penal = material_model_dict['penal']
			grad_obj = (-penal * x ** (penal - 1)) * ce
		elif material_model == MaterialModel.SIMPPLUS:
			# Needed for body force
			# For material model: (alpha-1)/alpha * x ** penal + (1/alpha) * x
			alpha = material_model_dict['alpha']
			penal = material_model_dict['penal']
			d_elem_material_scaling_dx = (alpha - 1) / alpha * penal * x ** (penal - 1) + 1 / alpha
			grad_obj = -d_elem_material_scaling_dx * ce

		if (nodal_body_force is not None):
			ce_body_force = (u[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			grad_obj +=  2*ce_body_force
			
		grad_obj = (H * grad_obj)/Hs

		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj)

		if (to_params.ElemsToKeep is not None):
			grad_obj[to_params.ElemsToKeep] = min(grad_obj)

		cons = volume_fraction_upperlimit(x, to_params.DesiredVolFraction)
		# Optimality criteria update
		xold = x.copy()
		if  not directLagrangeMethod: # bisection method
			# Calculate Lagrange multiplier bounds
			l1 = 0
			l2 = 1e12
			lmid = 0.5 * (l2 + l1)
			# Bisection loop for volume constraint
			while (l2 - l1) > 1e-7:
				lmid = 0.5 * (l2 + l1)
				b = -grad_obj / lmid	
				# OC update with damping and bounds
				xnew = jnp.maximum(xmin,np.maximum(x - move,np.minimum(xmax, np.minimum(x + move, x * np.sqrt(b)))))
				if jnp.sum(xnew) - to_params.DesiredVolFraction * num_elems > 0:
					l1 = lmid
				else:
					l2 = lmid	
			x = xnew
			xPhys = x
		else: # direct method
			#Reference: https://link.springer.com/article/10.1007/s00158-020-02740-y
			setChange = True
			eta = 0.5
			varIn = np.ones(num_elems, dtype = bool)
			xMax = np.minimum(x+move, 1.0)
			xMin = np.maximum(x-move,0.001)
			volToDistribute = to_params.DesiredVolFraction*num_elems
			varTimesGrad = x*(abs(grad_obj))**eta
			while setChange:
				xnew = varTimesGrad/((np.sum(varTimesGrad[varIn])+1e-12) /(volToDistribute+1e-12)) 
				volToDistribute = to_params.DesiredVolFraction*num_elems -np.sum(xMax[xnew>=xMax]) -np.sum(xMin[xnew<=xMin])
				setChange = not np.array_equal((xnew<xMax) & (xnew>xMin), varIn)
				varIn = (xnew < xMax) & (xnew > xMin)
			
			xnew[xnew>xMax] = xMax[xnew>xMax]
			xnew[xnew<xMin] = xMin[xnew<xMin]
			x = xnew
			xPhys = xnew.copy()

		# Calculate change and update densities
		#change = jnp.linalg.norm(x - xold, np.inf)
		change = jnp.max(jnp.abs(x - xold))

		fe_solver.mesh.setPseudoDensity(np.asarray(xPhys))
	
		history['compliance'].append(obj)
		history['volume'].append(np.mean(xPhys))
		history['change'].append(change)
		# Estimate the percentage of grey elements
		grey_elements = np.sum((x > 0.05) & (x < 0.95))
		fraction_grey = (grey_elements / num_elems) 
		if (print_progress):
			print(f"it.: {iter+1:d}, obj.: {obj:.5g}, "
				  	f"vol.: {np.mean(xPhys):.3g}, grey: {fraction_grey:.3f}")
		if np.isnan(obj):
			print("Objective function became NaN. Exiting optimization.")
			errorMsg = "Objective is diverging"
			success = False
			break
		if (change < move_tol):# success
			break
		if (len(history['compliance'])) >= 2:
			dJ = abs((history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2])
			if (abs(dJ) < rel_conv_tol and abs(cons) < rel_conv_tol) and (fraction_grey < 0.1): # success
				break

	if iter == maxIterations - 1:
		errorMsg = "Maximum iterations reached"
		print(errorMsg)
		success = False
	totalTime = time.time() - tStart
	# extract binary topology while preserving volume fraction
	target_vf = to_params.DesiredVolFraction
	x_sorted = np.sort(x)
	threshold = x_sorted[int((1-target_vf)*len(x))]
	x = np.where(x < threshold, 0.0, 1.0)
	volfrac = np.mean(x)
	fe_solver.mesh.setPseudoDensity(x)
	obj,u = compliance(x, fe_solver, material_model_dict)
	history['compliance'].append(obj)
	history['volume'].append(volfrac)
	history['change'].append(change)
	if (obj > 2*history['compliance'][-2]):
		errorMsg = "Disconnected topology"
		success = False
	if (volfrac > 1.1*to_params.DesiredVolFraction):
		errorMsg =  f"vf {to_params.DesiredVolFraction:0.3f} not reached"
		success = False

	nFEAs = iter + 1
	print(f"Final objective: {obj:.4g}, vf: {np.mean(x):.3f}")
	print(f"Total Time: {totalTime:.2f} s")
	return np.asarray(u), history, success, errorMsg, nFEAs

	
if __name__ == "__main__":    
	jax.config.update("jax_enable_x64", True)
	from topopt_benchmarks import *

	print("-" * 50)
	to_problem = StructuralTOExamples.EdgeCantilever # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
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
	
	print("OptimizationMethod: OC")
	u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,
											to_params = to_params,
											plot_progress = True,
											debug = debug)
	timeTaken = time.time() - startTime
	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")

	title = f"OC: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

	
	# plot the optimized mesh
	fe_solver.plot_mesh(title = title, plot_bc = False, save_path = None)

	# plot other quantities over the optimized mesh
	fe_solver.plot_deformation()
	fe_solver.postprocess()
	fe_solver.plot_vonMisesStress()

	fig, ax1 = plt.subplots()

	# Plot compliance on left y-axis
	ax1.set_xlabel('Iterations')
	ax1.set_ylabel('Compliance', color='tab:blue')
	ax1.plot(history['compliance'], color='tab:blue', label='Compliance')
	ax1.tick_params(axis='y', labelcolor='tab:blue')

	# Plot volume fraction on right y-axis with dotted line
	ax2 = ax1.twinx()
	ax2.set_ylabel('Volume Fraction', color='tab:orange')
	ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
	ax2.tick_params(axis='y', labelcolor='tab:orange')
	ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

	plt.title('OC: Volume and Compliance vs. Iterations')

	# Add legend
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2)

	plt.grid(True)
	plt.show()