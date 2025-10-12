
from topopt_common import *
from topopt_material_model import *
from topopt_obj_cons_sensitivities import *
import time

def topopt_optimality_criteria(
							fe_solver, #hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
							to_params,
			  				maxIterations: int = 250,
							move: float = 0.2,
							move_tol: float = 0.05,
							rel_conv_tol: float = 1.e-4,
							print_progress: bool = True,
							plot_progress: bool = False,
							debug: bool = False,
							binarize_topology: bool = True,
							progress_callback=None, 
                            plotter=None  
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
	def log_message(msg): # This is a helper function to log messages in GUI or console
		if progress_callback:
			progress_callback(str(msg))
		else:
			print(msg)   
	nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
	material_model = MaterialModel.SIMP 
	tStart = time.time()
	elem_body_force = fe_solver.elem_body_force


	num_elems = fe_solver.mesh.num_elems
	if (print_progress):
		log_message("Computing Filters ...")
	[H,Hs] = createFilters(fe_solver, to_params)
	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,nDOFPerNode)

	# Initialize design variables
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		raise ValueError(f"Unknown constraint type: {constraintType}")
	
	x = volFractionConstraint * np.ones(num_elems, dtype = float)
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
	history = {'objective': [], 'volfrac': [], 'change': []}
	# OC parameters
	xmin = 0.001  # Minimum density
	xmax = 1.0    # Maximum density
	
	if isinstance(fe_solver.mat_prop, list): # multiple materials
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural( mp.youngs_modulus,mp.poissons_ratio,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal( mp.thermal_conductivity,fe_solver.mesh.elem_size)
				for mp in fe_solver.mat_prop]
			KE = KE_list[0]	
		log_message("Assuming all elements have the same material properties")
	else: # single material
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
	success = True
	errorMsg = "None"
	
	for iter in range(maxIterations):
		x = np.array(x)
		if (plot_progress):
			if progress_callback is not None:
				progress_callback()	
			fe_solver.mesh.setPseudoDensity(x)
			fe_solver.plot_pseudo_density(plotter=plotter,auto_close = False, title = f"Iteration {iter}")
		
		
		sol = fe_solver.solve(x, material_model)
		fe_solver.postprocess()
		obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)

		if (len(history['objective']) == 0):
			objScaling = abs(obj)
		obj = obj/objScaling # Scale the objective function to avoid numerical issues
		grad_obj /= objScaling
	
		if (nodal_body_force is not None): #additional body force term
			ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			grad_obj +=  2*ce_body_force
			
		if (to_params.APPLY_FILTER_TO_SENSITIVITY):
			grad_obj = (H *(x* grad_obj))/Hs/x # apply filter

		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj) # retain elements that have nodes with external forces

		if (to_params.ElemsToKeep is not None):
			grad_obj[to_params.ElemsToKeep] = min(grad_obj) # also retain elements that are in the keep list

	
		volConstraint, volConstraint_gradient = compute_volume_constraint_and_gradient(x, volFractionConstraint)

		# Optimality criteria update
		xold = x.copy()
		
		# Calculate Lagrange multiplier bounds
		l1 = 0
		l2 = 1e12
		lmid = 0.5 * (l2 + l1)
		# Bisection loop for volume constraint
		while (l2 - l1)/((l1+l2)/2+1e-10) > 1e-7:
			lmid = 0.5 * (l2 + l1)
			D = -grad_obj / (volConstraint_gradient*lmid)	
			D = np.clip(D, 0.1,10) # 
			# OC update with damping and bounds
			xnew = np.maximum(xmin,np.maximum(x - move,np.minimum(xmax, np.minimum(x + move, x * np.sqrt(D)))))
			volConstraint, _ = compute_volume_constraint_and_gradient(xnew, volFractionConstraint)
			if volConstraint > 0:
				l1 = lmid
			else:
				l2 = lmid	
		x = xnew.copy()
		xPhys = x.copy()
		
		if (to_params.APPLY_FILTER_TO_DENSITY):
			x = H *x/Hs # apply filter
		# Calculate change and update densities
		change = np.max(np.abs(x - xold))

		fe_solver.mesh.setPseudoDensity(np.asarray(xPhys))
	
		history['objective'].append(obj*objScaling)
		history['volfrac'].append(np.mean(xPhys))
		history['change'].append(change)
		# Estimate the percentage of grey elements
		grey_elements = np.sum((x > 0.05) & (x < 0.95))
		fraction_grey = (grey_elements / num_elems) 

		if (print_progress):
			log_message(f"it.: {iter+1:d}, obj.: {(obj*objScaling):.3g}, "
				  	f"vol.: {np.mean(xPhys):.3g}, grey: {fraction_grey:.3f}")
		if np.isnan(obj):
			log_message("Objective function became NaN. Exiting optimization.")
			errorMsg = "Objective is diverging"
			success = False
			break
		
		if (len(history['objective'])) >= 2:
			dJ = abs((history['objective'][-1] - history['objective'][-2]) / history['objective'][-2])
			# we need multiple checks else it will terminate too early 
			if (dJ < rel_conv_tol) and (volConstraint < rel_conv_tol) and (change < 0.2) and (fraction_grey < 0.1): # success
				log_message("OC optimization converged.")
				break

			# Also this check for stalling
			if (dJ < rel_conv_tol) and (volConstraint < rel_conv_tol) and (change < move_tol): # success
				log_message("OC optimization converged.")
				break

	if iter == maxIterations - 1:
		errorMsg = "Maximum iterations reached"
		log_message(errorMsg)
		success = False
	totalTime = time.time() - tStart
	if (binarize_topology):
		# extract binary topology while preserving volume fraction
		x_sorted = np.sort(x)
		threshold = x_sorted[int((1-np.mean(x))*len(x))]
		x = np.where(x < threshold, 0.0, 1.0)
	volfrac = np.mean(x)
	fe_solver.mesh.setPseudoDensity(x)
	meshComponents = fe_solver.mesh.find_connected_components()
	if (len(meshComponents) > 1):
		if (print_progress):
			log_message("Disconnected topology detected. Removing hanging elements.")
		# Find the largest connected component and its size
		largest_component = max(meshComponents, key=len)
		# Set density to 1 for elements in largest component
		x[:] = 0.0
		x[list(largest_component)] = 1.0
		fe_solver.mesh.setPseudoDensity(x.flatten())
		volfrac = np.mean(x)
	sol = fe_solver.solve(x, material_model)
	obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)

	history['objective'].append(obj)
	history['volfrac'].append(volfrac)
	history['change'].append(change)

	if (volfrac > 1.1*volFractionConstraint):
		errorMsg =  f"vf {volFractionConstraint:0.3f} not reached"
		success = False

	nFEAs = iter + 1
	log_message(f"Final objective: {obj:.4g}, vf: {np.mean(x):.3f}")
	log_message(f"Total Time: {totalTime:.2f} s")
	log_message(f"Error: {errorMsg}")
	return sol, history, success, errorMsg, nFEAs

	 
if __name__ == "__main__":    
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *

	print("-" * 50)
	to_problem = StructuralTOExamples.EdgeCantilever # Choose the TO problem

	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False


	dsolver = deflation.DeflationSolver()
	if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):#  # Choose solver. Typically PARDISO, but DPCG for large DOF problems
		solver = lin_solv.Solvers.DPCG
		nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_deflation_matrix(mesh)
		dsolver.W = dsolver.W[bc.free_dofs, :]

	if (to_problem in StructuralTOExamples):
		fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					rtol = 1e-8,
					elem_body_force = elem_body_force)
	elif (to_problem in ThermalTOExamples):
		fe_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					rtol = 1e-8,
					elem_body_force = elem_body_force)
	

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	#print("Close the plot to continue...")
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#fe_solver.plot_mesh(title = title, save_path = None)
	
	startTime = time.time()		
	
	print("OptimizationMethod: OC")
	sol, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,
											to_params = to_params,
											plot_progress = True,
											debug = debug)
	timeTaken = time.time() - startTime
	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")

	title = f"OC: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"

	
	# plot the optimized mesh
	fe_solver.plot_mesh(title = title, plot_bc = False, save_path = None)

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
