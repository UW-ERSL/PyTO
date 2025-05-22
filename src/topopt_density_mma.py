from topopt_common import *
from topopt_material_model import *
import time
import mma
import matplotlib.pyplot as plt
def topopt_mma(fe_solver, #hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
			   			to_params,
			   			minMMAIterations: int = 5,
			   			 maxMMAIterations: int = 250, 
							timeLimit: float =3600, #1 hour
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-6,
							 move_tol: float = 0.05,
							 rel_conv_tol: float = 1.e-3,
							 print_progress: bool = True,
							plot_progress: bool = False,
							 debug: bool = False,
							 ) -> tuple[np.ndarray, dict]:
	"""MMA based topology optimization for minimum compliance.

	Args:
		fe_solver: The structural FEA solver object.
		maxMMAIterations: Maximum number of MMA iterations.
		volfrac: The target volume fraction.
		penal: The penalization factor for the SIMP method.
		move_limit: The maximum change allowed for the design variables in each
			iteration.
		kkt_tol: The tolerance for the KKT conditions.
		step_tol: The tolerance for the step size.

	Returns: The displacement field of the optimized structure.
	"""

	nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
	material_model = MaterialModel.SIMP 

	elem_body_force = fe_solver.elem_body_force
	tStart = time.time()
	num_elems= fe_solver.mesh.num_elems
	history = {'objective': [], 'volume': [], 'change': []}
	if (print_progress):
		print("Computing Filters ...")
	[H,Hs] = createFilters(fe_solver, to_params)

	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force,nDOFPerNode)
	nConstraints = len(to_params.Constraints)

	xmin = 0 # Minimum density
	mma_params = mma.MMAParams(max_iter=maxMMAIterations,
														kkt_tol = kkt_tol,
														step_tol = move_tol,
														move_limit = move_limit,
														num_design_var = num_elems,
														num_cons = nConstraints,
														lower_bound = xmin*np.ones((num_elems, 1)),
														upper_bound = np.ones((num_elems, 1)),
														)
	
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
	
	
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		volFractionConstraint =1 # default value
	
	x0 = volFractionConstraint* np.ones(num_elems, dtype = float)
	x0 = x0.reshape(-1, 1)
	mma_state = mma.init_mma(x0, mma_params)
	
	x_old = mma_state.x.reshape(-1)
	timeFEA = 0
	timeMMA = 0
	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None

	success = True
	errorMsg = "None"
	nFEAs = 0

	while True:
		x = mma_state.x.reshape(-1)
		if (to_params.APPLY_FILTER_TO_DENSITY):
			x = H*x/Hs
			mma_state.x = x.reshape(-1, 1)

		if (plot_progress):
			fe_solver.mesh.setPseudoDensity(x)
			fe_solver.plot_pseudo_density(auto_close = False, title = f"Iteration {mma_state.epoch+1}")
		timeFEAStart = time.time()

		sol = fe_solver.solve(x, material_model)
		nFEAs += 1
		timeFEA += time.time() - timeFEAStart

		obj, grad_obj = compute_objective_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
		
		if (len(history['objective']) == 0):
			objScaling = abs(0.1*obj)   # Scale the objective function to be in the range of 10
		obj = obj/objScaling
		grad_obj /=objScaling

		if (nodal_body_force is not None): # additional body force term
			ce_body_force = (sol[fe_solver.mesh.edofMat].reshape(num_elems, 24) * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			grad_obj +=  2*ce_body_force*get_material_model_rho_sensitivity(x,material_model)

		if (to_params.APPLY_FILTER_TO_SENSITIVITY) and (to_params.Objective is not TO_QOI.VOLUME_FRACTION):
			grad_obj = (H * grad_obj)/Hs # apply filter
		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj) # retain elements that have nodes with external forces

		if (to_params.ElemsToKeep is not None):
			grad_obj[to_params.ElemsToKeep] = min(grad_obj) # also retain elements that are in the keep list

		c, dcdx = compute_constraint_and_gradient(to_params,sol,x, fe_solver,KE, material_model)
		if (to_params.APPLY_FILTER_TO_SENSITIVITY):
			for m in range(len(to_params.Constraints)):
				if (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
					dcdx[m] = ((H @ dcdx[m])/Hs) # apply filter
	


		timeMMAStart = time.time()
		mma_state = mma.update_mma(mma_state,
										mma_params,
										np.array([obj]),
										np.array([grad_obj]).reshape((num_elems, 1)),
										c,
										dcdx
										)
			

		timeMMA += time.time() - timeMMAStart
		
		change = np.max(np.abs(x - x_old))
		x_old = x.copy()
		# Estimate the percentage of grey elements
		grey_elements = np.sum((x > 0.05) & (x < 0.95))
		fraction_grey = (grey_elements / num_elems) 

		if (print_progress):
			print(f"it.: {mma_state.epoch}, obj.: {obj*objScaling:.4g}, vf: {np.mean(x):.3f}, change: {change: 0.3f}, grey: {fraction_grey:.3f}")
		history['objective'].append(obj*objScaling)
		history['volume'].append(np.mean(x))
		history['change'].append(change)

		if (len(history['objective'])) >= minMMAIterations:
			dJ = abs((history['objective'][-1] - history['objective'][-2]) / abs(history['objective'][-2]))
			if (debug):
				print(f"relative Change in Objective: {dJ:.4g}")	

			# From experiments,  multiple checks were needed to ensure convergence
			if (dJ < rel_conv_tol) and (c[0] < rel_conv_tol) and (change < 0.2) and (fraction_grey < 0.1): # success
				print("MMA optimization converged.")
				break

			# Also this check for stalling
			if (dJ < rel_conv_tol) and (c[0] < rel_conv_tol) and (change < move_tol): # success
				print("MMA optimization converged.")
				break

			
		if time.time() - tStart > timeLimit:
			success = False
			errorMsg = "Time limit exceeded."
			print("MMA optimization terminated due to time limit.")
			break
		if len(history['objective']) >= maxMMAIterations:
			success = False
			errorMsg = "Maximum iterations reached."
			print("MMA optimization terminated due to maximum iterations.")
			break

	# Find threshold that preserves volume fraction
	x_sorted = np.sort(x)
	threshold = x_sorted[int((1-np.mean(x))*len(x))]
	x = np.where(x < threshold, 0.0, 1.0)
	volfrac = np.mean(x)
	fe_solver.mesh.setPseudoDensity(x)
	meshComponents = fe_solver.mesh.find_connected_components()
	
	if (len(meshComponents) > 1):
		if (print_progress):
			print("Disconnected topology detected. Removing hanging elements.")
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
	history['volume'].append(volfrac)
	history['change'].append(change)
	if (obj > 2*history['objective'][-2]):
		errorMsg = "Disconnected topology"
		success = False
	if (volfrac > 1.1*volFractionConstraint):
		errorMsg = f"vf {volFractionConstraint:0.3f} not reached"
		success = False 
	grey_elements = np.sum((x > 0.1) & (x < 0.9))
	fraction_grey = (grey_elements / num_elems) 

	print(f"Final objective: {obj:.4g}, vf: {np.mean(x):.3f}, grey: {fraction_grey:.3f}")
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	print(f"Total Time: {timeFEA+timeMMA:.2f} s")
	print("Error: ", errorMsg)
	return np.asarray(sol), history,success,errorMsg,nFEAs
	
if __name__ == "__main__":    
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
 
	print("-" * 50)

	to_problem = StructuralTOExamples.Mitchell_1 # Choose the TO problem

	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

	
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	dsolver = deflation.DeflationSolver()
	if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):# Typically PARDISO, but DPCG for large DOF problems
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
	print("nNodes: ", fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#fe_solver.plot_mesh(title = title, save_path = None)
	
	startTime = time.time()
	print("OptimizationMethod: MMA")
	u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,
								to_params = to_params,
								plot_progress = True,
								debug = debug)
	timeTaken = time.time() - startTime


	title = f"MMA: vol: {history['volume'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
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
	ax2.plot(history['volume'], color='tab:orange', linestyle=':', label='Volume Fraction')
	ax2.tick_params(axis='y', labelcolor='tab:orange')
	ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

	plt.title('MMA: Volume and Compliance vs. Iterations')

	# Add legend
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2)

	plt.grid(True)
	plt.show()