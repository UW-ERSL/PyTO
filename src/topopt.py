"""Optimization routines for topology optimization."""

import functools
import enum
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import element_stiffness as elem_stiff
import mesher
import mat_lib
import struct_fea as sfea
import mma


_LARGE_NUMBER = 1.e9


class Optimizers(enum.Enum):
	MMA = enum.auto()
	OC = enum.auto()
	PARETO = enum.auto()


def createFilter(mesh: mesher.Mesher):
	## Prepare filter
	nfilter = int(27 * mesh.num_elems)
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
	cc = 0

	elemNeighborsArray = mesh.elemNeighborsArray
	for elem in range(mesh.num_elems):
		elemNeighbors = elemNeighborsArray[elem]
		for neighbor in elemNeighbors:
			if neighbor >= 0:
				r = np.linalg.norm(mesh.elem_centers[elem, :] -
											 		 mesh.elem_centers[int(neighbor), :])
				weight = np.exp(-1*r**2)
				iH[cc] = elem
				jH[cc] = neighbor
				sH[cc] = weight
				cc = cc + 1
	# Finalize assembly and convert to csc format
	H = coo_matrix((sH, (iH, jH)), shape = (mesh.num_elems, mesh.num_elems)).tocsc()
	Hs = np.array(H.sum(1)).squeeze()
	return H, Hs


def _volume_constraint(density: jnp.ndarray,
											 volfrac: float,
											 )-> jnp.ndarray:
	"""Compute the volume constraint.
	
	Args:
		density: Array of (num_elems,) containing the element densities.
		volfrac: The target volume fraction.
	
	Returns: The volume constraint. The constraint is satisfied when the
		returned value is zero. The constraint is inactive when the returned
		value is negative.
	"""
	return (jnp.mean(density)/volfrac) - 1.0


def _compliance_objective(density: jnp.ndarray,
													fe_solver: sfea.StructFEA,
													penal: float = 3.0,
													) -> jnp.ndarray:
	"""Compute the structural compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	penal_dens = density ** penal
	u = fe_solver.solve(penal_dens)
	return jnp.einsum('i, i -> ', fe_solver.bc.force, u), u


def topopt_mma(fe_solver: sfea.StructFEA,
			   			 maxMMAIterations: int = 500, 
			   			 volfrac: float = 0.5,
							 penal: float = 3.,
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-5,
							 step_tol: float = 1.e-2,
							 ) -> tuple[np.ndarray, dict]:
	"""Optimality Criteria based topology optimization for minimum compliance.

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
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	H, Hs = createFilter(fe_solver.mesh)

	mma_params = mma.MMAParams(max_iter=maxMMAIterations,
														kkt_tol = kkt_tol,
														step_tol = step_tol,
														move_limit = move_limit,
														num_design_var = num_elems,
														num_cons = 1,
														lower_bound = np.zeros((num_elems, 1)),
														upper_bound = np.ones((num_elems, 1)),
														)
	mma_state = mma.init_mma(volfrac * np.ones((num_elems, 1)), mma_params)
	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	x_old = volfrac*np.ones(num_elems, dtype = float)
	while not mma_state.is_converged:
		x = mma_state.x.reshape(-1)
		obj,u = _compliance_objective(x, fe_solver, penal)
		obj = np.array([obj])
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		grad_obj = (-penal * x ** (penal - 1)) * ce
		grad_obj = (H * grad_obj)/Hs

		cons = _volume_constraint(x, volfrac)
		grad_cons = np.ones(num_elems)/volfrac

		mma_state = mma.update_mma(mma_state,
														   mma_params,
														 	 obj,
															 np.array([grad_obj]).reshape((num_elems, 1)),
														 	 jnp.array([cons]).reshape((1, 1)),
															 grad_cons.reshape((1, num_elems))
															 )

		change = np.max(np.abs(x - x_old))
		x_old = x
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.3f} vc: {cons:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
		history['change'].append(change)

	fe_solver.mesh.setPseudoDensity(x)
	return np.asarray(u), history


def topopt_optimality_criteria(
							fe_solver: sfea.StructFEA,
			  			maxIterations: int = 500,
							volfrac: float = 0.5,
							penal: float = 3,
							move: float = 0.2,
							verbose: bool = True
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
	num_elems = fe_solver.mesh.num_elems

	H, Hs = createFilter(fe_solver.mesh)

	# Initialize design variables
	x = volfrac * jnp.ones(num_elems)
	xPhys = x.copy()

	# Initialize history
	history = {'compliance': [], 'volume': [], 'change': []}

	# OC parameters
	xmin = 0.001  # Minimum density
	xmax = 1.0    # Maximum density
	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	for iter in range(maxIterations):
		obj,u = _compliance_objective(x, fe_solver, penal)
		
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		grad_obj = (-penal * x ** (penal - 1)) * ce
		grad_obj = (H * grad_obj)/Hs

		# Optimality criteria update
		xold = x.copy()

		# Calculate Lagrange multiplier bounds
		l1 = 0
		l2 = _LARGE_NUMBER

		# Bisection loop for volume constraint
		while (l2 - l1) > 1e-9:
			lmid = 0.5 * (l2 + l1)
			b = -grad_obj / lmid	
			# OC update with damping and bounds
			xnew = jnp.maximum(xmin,jnp.maximum(x - move,jnp.minimum(xmax, jnp.minimum(x + move, x * np.sqrt(b)))))

			if jnp.sum(xnew) - volfrac * num_elems > 0:
				l1 = lmid
			else:
				l2 = lmid
		
		x = xnew
		xPhys = x
	
		# Calculate change and update densities
		change = jnp.linalg.norm(x - xold, np.inf)
		fe_solver.mesh.setPseudoDensity(np.asarray(xPhys))
	
		history['compliance'].append(obj)
		history['volume'].append(np.mean(xPhys))
		history['change'].append(change)

		if verbose:
			print(f"it.: {iter+1:d}, obj.: {obj:.3e}, "
				  	f"vol.: {np.mean(xPhys):.3f}, ch.: {change:.3f}")

		if change < 0.025:
			break

	return np.asarray(u), history


def topopt_pareto(fe_solver: sfea.StructFEA,
							desiredVolFrac: float = 0.5,
							rel_err: float = 0.025,
							vol_decr_max: float = 0.05,
							min_local_iters: int = 2,
							max_local_iters: int = 10,
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
	totalIter = 1

	# Initialize design field
	rho = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'compliance': [], 'volume': []}

	# Create filter
	H, Hs = createFilter(fe_solver.mesh)

	u = np.asarray(fe_solver.solve(rho))

	# Store initial compliance
	history['compliance'].append(fe_solver.bc.force.T @ u)
	history['volume'].append(volfrac)

	# Compute initial topological sensitivity
	T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
	T = (H * T) / Hs
	print(f"v={volfrac:.2f}; J={history['compliance'][-1]:.2e}; #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	while volfrac > desiredVolFrac:
		# Move to next volume fraction
		volfrac = max(desiredVolFrac, volfrac * (1 - vol_decr))

		localIter = 0
		success = False
		JTemp = history['compliance'][-1]  # Store previous value
		JPrev = float('inf')  # Initialize JPrev
		JPrevPrev = float('inf')  # Initialize JPrev
		while localIter < max_local_iters:
			if JTemp > 10 * history['compliance'][0]:  # Divergence check
				break
			# Check convergence, and break if converged
			if localIter >= min_local_iters and abs(min(JPrev,JPrevPrev) - JTemp)/JTemp < rel_err:
				success = True
				break

			# Find cutoff value and update design
			value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
			rho = np.ones((fe_solver.mesh.num_elems))
			rho[T < value] = 0.001
			JPrevPrev = JPrev  # Store previous value
			JPrev = JTemp  # Store previous value
			
			u = np.asarray(fe_solver.solve(rho))
			JTemp = float(fe_solver.bc.force.T @ u)
			
			# Update sensitivity
			T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
			T = (H * T) / Hs

			localIter += 1
			totalIter += 1

		if not success:
			break
		
		vol_decr = max(0.01,0.95*vol_decr)  # Decrease volume decrease factor

		history['compliance'].append(JTemp)
		history['volume'].append(volfrac)
		print(f"v={history['volume'][-1]:.2f}; J={history['compliance'][-1]:.2e}; #FEA={totalIter:2d}")
		fe_solver.mesh.setPseudoDensity(rho.flatten())

	return u, history


def computeTopologicalSensitivity(mesh: mesher.Mesher,
																	mat_prop: mat_lib.StructuralMaterial,
																	u, rho):
	"""Compute topological sensitivity field."""

	T = np.zeros((mesh.num_elems))
	e, nu = mat_prop.youngs_modulus, mat_prop.poissons_ratio
	# Create constitutive matrix
	v1 = 2*nu**2 + nu - 1
	v2 = 2*nu + 2
	D = e * np.array([
										[(nu - 1)/v1, -nu/v1, -nu/v1, 0, 0, 0],
										[-nu/v1, (nu - 1)/v1, -nu/v1, 0, 0, 0],
										[-nu/v1, -nu/v1, (nu - 1)/v1, 0, 0, 0],
										[0, 0, 0, 1/v2, 0, 0],
										[0, 0, 0, 0, 1/v2, 0],
										[0, 0, 0, 0, 0, 1/v2]
									])

	# Shape function gradients at center
	gradN = 1/8 * np.array([
													[-1, 1, 1, -1, -1, 1, 1, -1],
													[-1, -1, 1, 1, -1, -1, 1, 1],
													[-1, -1, -1, -1, 1, 1, 1, 1]
												])

	for elem in range(mesh.num_elems):
		edof = mesh.edofMat[elem]
	
		# Get displacement gradients
		uGrad = gradN @ u[edof[::3]]
		vGrad = gradN @ u[edof[1::3]]
		wGrad = gradN @ u[edof[2::3]]

		# Compute strains
		strains = np.array([
			uGrad[0], vGrad[1], wGrad[2],
			uGrad[1] + vGrad[0],
			uGrad[2] + wGrad[0],
			vGrad[2] + wGrad[1]
		])

		# Compute stresses
		stresses = rho[elem] * D @ strains

		# Create tensors
		stress_tensor = np.array([
															[stresses[0], stresses[3], stresses[4]],
															[stresses[3], stresses[1], stresses[5]],
															[stresses[4], stresses[5], stresses[2]]
														])
		
		strain_tensor = np.array([
															[strains[0], strains[3], strains[4]],
															[strains[3], strains[1], strains[5]],
															[strains[4], strains[5], strains[2]]
														])

		# Compute topological sensitivity
		T[elem] = (4/(1+nu) * np.sum(stress_tensor * strain_tensor) - 
						  (1-3*nu)/(1-nu**2) * np.trace(stress_tensor) * np.trace(strain_tensor))

	return T

if __name__ == "__main__":    
	from examples_structural import createCantileverProblem, createLBracketProblem
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time
	import plots	
	jax.config.update("jax_enable_x64", True)


	example = 2
	nDOFDesired = 10000
	if example == 1:
		mesh, mat_prop, bc = createCantileverProblem(nDOFDesired = nDOFDesired)
	elif example == 2:
		mesh, mat_prop, bc = createLBracketProblem(nDOFDesired = nDOFDesired)    

	fe_solver = fea.StructFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = lin_solv.Solvers.PARDISO)

	youngs_modulus = np.ones((fe_solver.mesh.num_elems,))

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	
	
	volfrac = 0.5
	num_iter = 100

	optimizationMethod = 1 # 1: MMA, 2: OC, 3: Pareto	

	startTime = time.time()
	if optimizationMethod == 1:
		print("optimizationMethod: MMA")
		u, history = topopt_mma(fe_solver = fe_solver,
														maxMMAIterations = num_iter,
														volfrac = volfrac
														)
		timeTaken = time.time() - startTime

		title = f'MMA: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3e}, time: {timeTaken:.2f} s'
	elif optimizationMethod == 2:
		print("optimizationMethod: OC")
		u, history = topopt_optimality_criteria(fe_solver = fe_solver,
												maxIterations= num_iter,
												volfrac = volfrac
												)
		timeTaken = time.time() - startTime
		title = f'OC: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3e}, time: {timeTaken:.2f} s'

	elif optimizationMethod == 3:
		print("optimizationMethod: Pareto")
		u, history = topopt_pareto(fe_solver = fe_solver,
										desiredVolFrac =  volfrac)
		
		timeTaken = time.time() - startTime
		title = f'Pareto: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3e}, time: {timeTaken:.2f} s'

	plots.plotMesh(fe_solver.mesh, fe_solver.bc, u, title = title)
