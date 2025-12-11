from topopt_common import *
import time
import numpy as np


#################################################################

def compute_objective_topological_sensitivity_compliance(feaMode: FEA_MODE,to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None):
	
	
	# Compute the compliance independent of objective
	if (feaMode == FEA_MODE.STRUCTURAL):
		dofMat = fe_solver.mesh.edofMatStructural
	elif (feaMode == FEA_MODE.THERMAL):
		dofMat = fe_solver.mesh.edofMatThermal
	
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)
	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)
	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")
	compliance = np.sum(materialScaling * ce)		

	# depending on the objective type, compute the topological sensitivity
	objectiveType  = to_params.Objective[0]	# first entry is the type of objective	
	if (objectiveType == TO_QOI.COMPLIANCE): 
		if (nRows == 24): # structural hex
			T = computeStructuralTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio,fe_solver.strainComponents,fe_solver.stressComponents,x)
		elif (nRows == 8): # thermal hex
			T = computeThermalTopologicalSensitivity(fe_solver.mat_prop.thermal_conductivity,fe_solver.strain,x)
		else:
			raise ValueError("Invalid number of rows in element stiffness matrix.")
		obj = compliance
		return obj,T,compliance
	elif (objectiveType == TO_QOI.PNORM_STRESS):
		pNormValue = to_params.Objective[1] or 16
		[stressObj, T] = compute_pnorm_stress_and_TS(sol, x, fe_solver,KE,material_model,pNormValue)
		return stressObj, T, compliance
	elif (objectiveType == TO_QOI.GVECTOR):
		g = to_params.Objective[1]
		obj = np.dot(fe_solver.sol, g)
	
		print("Not implemented yet")
		adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
		return obj,T, compliance
	
#################################################################
def computeStructuralTopologicalSensitivity(poissons_ratio,strains,stresses,x):
	stress_tensor = x[:, None, None] * np.array([
		[stresses[:, 0], stresses[:, 3], stresses[:, 4]],
		[stresses[:, 3], stresses[:, 1], stresses[:, 5]],
		[stresses[:, 4], stresses[:, 5], stresses[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
	strain_tensor = np.array([
		[strains[:, 0], strains[:, 3]/2, strains[:, 4]/2],
		[strains[:, 3]/2, strains[:, 1], strains[:, 5]/2],
		[strains[:, 4]/2, strains[:, 5]/2, strains[:, 2]]
	]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
	
	# Compute topological sensitivity
	trace_stress = np.trace(stress_tensor, axis1=1, axis2=2)
	trace_strain = np.trace(strain_tensor, axis1=1, axis2=2)
	if isinstance(poissons_ratio, list):
		# Handle multiple materials based on element component ID
		
		# This needs to be fixed to handle different nu values
		nu = poissons_ratio[0]
		
		T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1,2)) -
			 (1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)
	else:
		# Single material case
		nu = poissons_ratio
		T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1, 2)) -
			(1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)
	return T

#################################################################
def compute_pnorm_stress_and_TS(sol: np.ndarray, x,
										  fe_solver,KE,material_model, p=6):
	"""
    Compute von Mises stress and topological sensitivity for p-norm stress.
    """
	# "An efficient 146-line 3D sensitivity analysis code of 
	# stress based topology optimization written in MATLAB"
	# Optimization and Engineering (2022) 23:1733–1757
	# The sensitivity of pnorm von mises stress with respect to x has 2 terms: T1 and T2
	# T1 arises due to the stress relaxation: x**STRESS_RELAXATION
	# T2 arises indirectly via the solution sensitivity via the adjoint
	# T1 is small and can be ignored for large p, so we can use the adjoint sensitivity
	# to compute the topological sensitivity
	mesh = fe_solver.mesh
	nelems = mesh.num_elems
	q = 2 # STRESS_RELAXATION factor
	
	E = fe_solver.mat_prop.youngs_modulus 
	nu = fe_solver.mat_prop.poissons_ratio
	D = E / ((1 + nu) * (1 - 2*nu)) * np.array([
		[1-nu, nu, nu, 0, 0, 0],
		[nu, 1-nu, nu, 0, 0, 0],
		[nu, nu, 1-nu, 0, 0, 0],
		[0, 0, 0, (1-2*nu)/2, 0, 0],
		[0, 0, 0, 0, (1-2*nu)/2, 0],
		[0, 0, 0, 0, 0, (1-2*nu)/2]
	])
	gradN = (1 / 8) * np.array([
		[-1, 1, 1, -1, -1, 1, 1, -1],
		[-1, -1, 1, 1, -1, -1, 1, 1],
		[-1, -1, -1, -1, 1, 1, 1, 1]
	])
	for i in range(3):
		gradN[i, :] = 2*gradN[i,:] / fe_solver.mesh.elem_size[i]

	# Define the B matrix (strain-displacement matrix) for a hexahedral element at the center (xi=0, eta=0, zeta=0)
	B = np.zeros((6, 24))
	# Vectorized construction of B matrix for all 8 nodes at once
	Bi = np.zeros((6, 3, 8))
	Bi[0, 0, :] = gradN[0, :]
	Bi[1, 1, :] = gradN[1, :]
	Bi[2, 2, :] = gradN[2, :]
	Bi[3, 0, :] = gradN[1, :]
	Bi[3, 1, :] = gradN[0, :]
	Bi[4, 0, :] = gradN[2, :]
	Bi[4, 2, :] = gradN[0, :]
	Bi[5, 1, :] = gradN[2, :]
	Bi[5, 2, :] = gradN[1, :]
	# Vectorized assignment to B
	idx = np.arange(8)
	B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)
	F = D @ B  # shape (6, 24)
	g_elem = np.zeros((nelems, 24))
	vm_elems = np.zeros(nelems)

	for e in range(nelems):
		#  compute the stress with relaxation for sensitivity term T2 and pnorm stress
		stress_elem = (x[e]**q)* fe_solver.stressComponents[e]
		sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem
		vm_elems[e] = np.sqrt(0.5*((sigma11 - sigma22)**2 +(sigma22-sigma33)**2 + (sigma33-sigma11)**2) +
                3*(sigma12**2 + sigma13**2 + sigma23**2))

		g_e = ((sigma11 - sigma22) * (F[0] - F[1]) +
    	(sigma11 - sigma33) * (F[0] - F[2]) +
    	(sigma22 - sigma33) * (F[1] - F[2]) +
    	6 * sigma12 * F[3] + 6 * sigma13 * F[4] + 6 * sigma23 * F[5]) / np.sqrt(2)
		g_elem[e] = p * vm_elems[e] ** (p - 2) * g_e
	
	# Note that we are using the relaxed von Mises below
	vm_pnorm = np.sum(vm_elems**p)**(1/p)
	

	# Now compute the rhs of adjoint eqn 
	g = np.zeros(fe_solver.bc.num_dofs)
	for e in range(nelems): # assemble  g vector
		edof = mesh.edofMat[e]
		g[edof] += g_elem[e]
	g *= -(1 / p) * (np.sum(vm_elems ** p) ** (1/p - 1) )

    # Solve the adjoint	
	adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)
	
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	strain_adj = np.zeros((num_elems, 6))
	for e in range(num_elems):
		edof = dofMat[e]
		u_e = adjointSol[edof]
		strain_adj[e] = np.dot(B, u_e)
	
	T = computeStructuralTopologicalSensitivity(nu,strain_adj,fe_solver.stressComponents,x)
	return vm_pnorm,T
#################################################################
def computeThermalTopologicalSensitivity(conductivity,strains,x):
	# For thermal problems, topological sensitivity is related to conductivity * gradient^2
	# Multiply by density (x) to scale based on material distribution
	if isinstance(conductivity, list):
		# Handle multiple materials
		# Using the first conductivity value for now - this would need to be updated
		# to properly handle multiple materials
		k = conductivity[0]
	else:
		# Single material case
		k = conductivity
		
	T = x * k * np.sum(strains**2, axis=0)

	return T


def run_pareto_topopt(to_problem):
	if (to_problem in StructuralTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
		feaMode = FEA_MODE.STRUCTURAL
	elif (to_problem in ThermalTOExamples):
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
		feaMode = FEA_MODE.THERMAL

	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	
	plot_progress = True
	print_progress = True
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
	sol, history, success,errorMsg,nFEAs = topopt_pareto(feaMode,fe_solver = fe_solver,
									to_params = to_params,
									plot_progress= plot_progress,
									debug = debug)
	
	timeTaken = time.time() - startTime
	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")

	title = f"Pareto: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {nFEAs:3d}, time: {timeTaken:.0f} s"
	fe_solver.plot_mesh(title = title, save_path = None)

		
	# Plot volume vs compliance history
	plt.figure()
	plt.plot(history['volfrac'], history['objective'], marker='o')
	plt.xlabel('Volume Fraction')
	plt.ylabel('objective')
	plt.title('Pareto: Volume vs Compliance History')
	plt.grid(True)
	plt.show()
	
	


def topopt_pareto(feaMode,fe_solver,
				  to_params,
							rel_err: float = 0.025,
							vol_decr_max: float = 0.05,
							vol_decr_min: float = 0.0025,
							min_local_iters: int = 2,
							max_local_iters: int = 5,
							xVoid: float = 0,
							print_progress: bool = True,
							plot_progress: bool = False,
							debug: bool = False,
							progress_callback=None, 
                            plotter=None  
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
	objectiveType = to_params.Objective[0]
	if objectiveType != TO_QOI.COMPLIANCE:
		raise ValueError(f"Unsupported objective type: {objectiveType}")
    # Extract volume constraint
	constraintType = to_params.Constraints[0][0]
	if constraintType == TO_QOI.VOLUME_FRACTION:
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		raise ValueError(f"Unsupported constraint type: {constraintType}")
	def log_message(msg): # This is a helper function to log messages in GUI or console
		if progress_callback:
			progress_callback(str(msg))
		else:
			print(msg)  
	nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
	tStart = time.time()
	material_model = MaterialModel.SIMP 
	
	removeHangingElems = to_params.RemoveHangingElems
	if fe_solver.elem_body_force is not None and (np.linalg.norm(fe_solver.elem_body_force) > 0) and not removeHangingElems:
		removeHangingElems = True #For body forces, must remove hanging elements in Pareto



	# Initialize design field
	x = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'objective': [],'volfrac': []}
	if (print_progress):
		log_message("Computing Filters ...")
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
		log_message("Assuming all elements have the same material properties")
	else: # single material
		if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_structural( fe_solver.mat_prop.youngs_modulus,
															    fe_solver.mat_prop.poissons_ratio,
																fe_solver.mesh.elem_size)
		elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
			KE = hex_element_stiffness.hex8_stiffness_matrix_thermal( fe_solver.mat_prop.thermal_conductivity,fe_solver.mesh.elem_size)
	
	fe_solver.mesh.setPseudoDensity(x.flatten())
	sol = fe_solver.solve(x)
	fe_solver.postprocess()
	nFEAs = 1
	

	obj, T,compliance0 = compute_objective_topological_sensitivity_compliance(feaMode,to_params,sol,x, fe_solver,KE)
	obj0 = obj
	T = T/obj0
	J = obj

	history['objective'].append(obj)# may be the same as compliance
	history['volfrac'].append(volfrac)
	
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
	iteration = 0
	if (print_progress):
		log_message(f"Iteration: {iteration}, vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g},  #FEA={nFEAs:2d}")
	vol_decr = vol_decr_max

	success = True
	terminatePareto = False
	errorMsg = "No errors."
	
	nSmoothSteps = 2 # Number of smoothing steps to apply
	constraintType = to_params.Constraints[0][0] # assume this is the first constraint
	if (constraintType == TO_QOI.VOLUME_FRACTION):
		volFractionConstraint = to_params.Constraints[0][2]
	else:
		raise ValueError(f"Unknown constraint type: {constraintType}")
	
	while volfrac > volFractionConstraint:
		# Observation: Damping using the previous sensitivity values avoids getting trapped in local minima
		wtDamping = 0.9 # 0 means full wt to current T values, else previous T values are damped in
		fe_solver.mesh.setPseudoDensity(x)
		if progress_callback is not None:
			progress_callback()
		if (plot_progress):
			fe_solver.plot_pseudo_density_realtime(
                   title=f"Iter {iteration}",
				   iteration=iteration,
                   external_plotter=plotter  # Pass GUI plotter if available
               )
		# Move to next volume fraction
		volfrac = max(volFractionConstraint, volfrac - vol_decr)
		if (debug):
			log_message("-" * 50)
			log_message(f"Attempting v={volfrac:.3f}")
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
				log_message(f"Local Iteration: {localIter}/{max_local_iters}, JTemp: {JTemp:.3g}, JPrev: {JPrev:.3g}")
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
					log_message("**Failed to converge, restoring previous design")
					log_message(f"Previous successful vol_frac: {vol_frac_success:.5g}")
					log_message(f"Reducing vol_decr to: {vol_decr:.5g}")
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

			sol = fe_solver.solve(x, material_model)
			fe_solver.postprocess()

			nFEAs += 1
			obj, TTemp,compliance = compute_objective_topological_sensitivity_compliance(feaMode,to_params,sol,x, fe_solver,KE)
			TTemp = TTemp/obj0
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

			T = ((1-wtDamping)*T + wtDamping*TPrev)  # Damping
			wtDamping *= 0.5
			if (elemsWithForces.size > 0):
				T[elemsWithForces] = np.max(T)

			if (to_params.ElemsToKeep is not None):
				T[to_params.ElemsToKeep] = np.max(T)

			localIter += 1
			iteration += 1


		if terminatePareto:
			if (volfrac > 1.1*volFractionConstraint):
				success = False
				errorMsg =  f"vf {volFractionConstraint:0.3f} not reached"
				log_message("-" * 50)
				log_message("Pareto: Failed to reach volume fraction.")
				log_message("1. Check for incorrect symmetry constraints")
				log_message("2. Increase mesh size")
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
			history['volfrac'].append(volfrac)
			scale = (compliance / compliance0)**2
			vol_decr = max(vol_decr_min,min(vol_decr,vol_decr_max/scale)) # Reduce volume increment for steep increase in compliance
			if (print_progress):
				log_message(f"Iteration: {iteration}, vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g},  #FEA={nFEAs:2d}")
			fe_solver.mesh.setPseudoDensity(x.flatten())
		

	totalTime = time.time() - tStart

	log_message(f"Final: vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g},#FEA={nFEAs:2d}")
	log_message(f"Total Time: {totalTime:.2f} s")
	log_message(f"Error: {errorMsg}")
	return sol, history, success,errorMsg,nFEAs


if __name__ == "__main__":    
	from topopt_structural_benchmarks import *
	from topopt_thermal_benchmarks import *
	
	print("-" * 50)
	to_problem = StructuralTOExamples.MBBBeam # Choose the TO problem
	#to_problem = ThermalTOExamples.FourCornersThermal # Choose the TO problem

	run_pareto_topopt(to_problem)