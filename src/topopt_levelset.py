"""Optimization routines for topology optimization."""

from topopt_common import *


def topopt_levelset(fe_solver: sfea.StructFEA,
					 to_params,
						 maxIterations: int = 250,
						 volfrac: float = 0.5,
						 time_step: float = 0.1,
						 epsilon: float = 1.0,
						 rel_conv_tol: float = 1e-4,
						plotIntermediateTopologies: bool = False,
						 debug: bool = False) -> tuple[np.ndarray, dict]:
		"""Level Set Method for Topology Optimization using Hamilton-Jacobi equation.

		Args:
			fe_solver: The structural FEA solver object.
			maxIterations: Maximum number of iterations.
			volfrac: The target volume fraction.
			time_step: Time step for the Hamilton-Jacobi update.
			epsilon: Regularization parameter for the Heaviside function.
			rel_conv_tol: Relative convergence tolerance.
			to_params: Topology optimization constraints.
			debug: If True, prints debug information.

		Returns:
			A tuple containing the displacement field of the optimized structure
			and a dictionary containing the optimization history.
		"""
		def heaviside(phi, epsilon):
			"""Smooth Heaviside function."""
			return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

		def heaviside_derivative(phi, epsilon):
			"""Derivative of the smooth Heaviside function."""
			return (1 / (np.pi * epsilon)) * (1 / (1 + (phi / epsilon) ** 2))

		def compute_velocity_field(fe_solver, phi, material_model_dict):
			"""Compute the velocity field for the level set update."""
			density = heaviside(phi, epsilon)
			obj, u = compliance(density, fe_solver, material_model_dict)
			ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			velocity = -ce * heaviside_derivative(phi, epsilon)
			return velocity, obj

		# Initialize level set field (phi)
		num_elems = fe_solver.mesh.num_elems
		phi = np.ones(num_elems)

		# History for compliance and volume
		history = {'compliance': [], 'volume': [], 'change': []}

		# Material model dictionary
		material_model_dict = {'name': 'SIMP', 'penal': 3.0}

		# Element stiffness matrix
		KE = elem_stiff.hex8_stiffness_matrix_structural(fe_solver.mat_prop, fe_solver.mesh.elem_size)
		[H,Hs] = createFilters(fe_solver, to_params)
		success = True
		errorMsg = ""
		for iter in range(maxIterations):
			# Compute velocity field and objective
			velocity, obj = compute_velocity_field(fe_solver, phi, material_model_dict)

			# Update level set field using Hamilton-Jacobi equation
			phi_old = phi.copy()
			phi += time_step * velocity

			# Apply volume constraint
			density = heaviside(phi, epsilon)
			current_volfrac = np.mean(density)
			if current_volfrac > volfrac:
				phi -= time_step * (current_volfrac - volfrac)

			# Update pseudo-density in the mesh
			fe_solver.mesh.setPseudoDensity(density)

			# Compute change and update history
			change = np.max(np.abs(phi - phi_old))
			history['compliance'].append(obj)
			history['volume'].append(np.mean(density))
			history['change'].append(change)

			print(f"it.: {iter + 1}, vol.: {np.mean(density):.3f}, obj.: {obj:.6g},  ch.: {change:.3g}")

			# Check for convergence
			if iter > 1 and change < rel_conv_tol:
				break

		if iter == maxIterations - 1:
			print("Maximum iterations reached without convergence.")
			success = False
		# Solve for final displacement field
		u = np.asarray(fe_solver.solve(density))
		nFEAs = iter
		return u, history, success,errorMsg, nFEAs


if __name__ == "__main__":    
	
	from topopt_examples import *
	jax.config.update("jax_enable_x64", True)
	
	print("-" * 50)
	to_problem = StructuralTOExamples.MidCantilever # Choose the TO problem
	print(f"Running {to_problem.name}...") 
	print("-" * 50)
	solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
	debug = False

	# Get the structural problem
	mesh, mat_prop, bc,elem_body_force, to_params,nFEAs = getStructuralTOProblem(to_problem)

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
	#plots.plotMesh(mesh, bc,title = title)


	startTime = time.time()
	
	
	print("OptimizationMethod: Level Set")
	u, history, success,errorMsg = topopt_levelset(fe_solver = fe_solver,
									to_params = to_params,
									maxIterations = 100,
									time_step = 0.1,
									epsilon = 1.0,
									rel_conv_tol = 1e-4,
									debug = debug)
	timeTaken = time.time() - startTime
	title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"	

	print(f"Time taken: {timeTaken:.0f} s")
	if not success:
		print(f"Error: {errorMsg}")
	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title)

	#plots.plotIsocontour(fe_solver.mesh, title = title, save_path = None)
	# Save the mesh and results