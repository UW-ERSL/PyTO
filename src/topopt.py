"""Optimization routines for topology optimization."""

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
import deflation
from scipy.optimize import minimize
import nlopt


_LARGE_NUMBER = 1.e9


class Optimizers(enum.Enum):
	MMA = enum.auto()
	OC = enum.auto()
	PARETO = enum.auto()
	SCIPY = enum.auto()
	NLOPT = enum.auto()

def find_elements_with_forces(mesh: mesher.Mesher, force) -> np.ndarray:
	"""Find all elements that have nodes on which force has been applied.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with applied forces.
	"""
	force_dofs = np.where(force != 0)[0]
	forced_nodes = set(force_dofs // 3)  # Convert DOFs to node indices
	elements_with_forces = []

	for elem in range(mesh.num_elems):
		nodes = mesh.elemArray[elem]
		if any(node in forced_nodes for node in nodes):
			elements_with_forces.append(elem)

	return np.array(elements_with_forces)

def createSmoothingFilter(mesh: mesher.Mesher):
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

def createSymmetryFilterXMidPlane(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about X mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HX: Sparse matrix that when multiplied with density vector enforces X mid-plane symmetry
			HXs: Array of row sums of HX matrix
	"""
	num_elems = mesh.num_elems
	x_mid = (mesh.elem_centers[:, 0].max() + mesh.elem_centers[:, 0].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		mirror_x = 2 * x_mid - mesh.elem_centers[i, 0]
		mirror_idx = np.argmin(np.abs(mesh.elem_centers[:, 0] - mirror_x))
		if (mirror_idx == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idx)
			data.append(0.5)

	HX = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	HXs = np.array(HX.sum(1)).squeeze()
	return HX, HXs

def createSymmetryFilterYMidPlane(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Y mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HY: Sparse matrix that when multiplied with density vector enforces Y mid-plane symmetry
			HYs: Array of row sums of HY matrix
	"""
	num_elems = mesh.num_elems
	y_mid = (mesh.elem_centers[:, 1].max() + mesh.elem_centers[:, 1].min()) / 2
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		mirror_y = 2 * y_mid - mesh.elem_centers[i, 1]
		mirror_idy = np.argmin(np.abs(mesh.elem_centers[:, 1] - mirror_y))
		if (mirror_idy == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idy)
			data.append(0.5)

	HY = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	HYs = np.array(HY.sum(1)).squeeze()
	return HY, HYs
	

def createSymmetryFilterZMidPlane(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Z mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZ: Sparse matrix that when multiplied with density vector enforces Z mid-plane symmetry
			HZs: Array of row sums of HZ matrix
	"""
	num_elems = mesh.num_elems
	z_mid = (mesh.elem_centers[:, 2].max() + mesh.elem_centers[:, 2].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		mirror_z = 2 * z_mid - mesh.elem_centers[i, 2]
		mirror_idz = np.argmin(np.abs(mesh.elem_centers[:, 2] - mirror_z))
		if (mirror_idz == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idz)
			data.append(0.5)

	HZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	HZs = np.array(HZ.sum(1)).squeeze()
	return HZ, HZs

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
							 move_limit: float = 0.3,
							 kkt_tol: float = 1.e-5,
							 step_tol: float = 0.025,
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
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	H, Hs = createSmoothingFilter(fe_solver.mesh)

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
	timeFEA = 0
	timeMMA = 0
	while not mma_state.is_converged:
		x = mma_state.x.reshape(-1)
		timeFEAStart = time.time()
		obj,u = _compliance_objective(x, fe_solver, penal)
		timeFEA += time.time() - timeFEAStart
		obj = np.array([obj])
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		grad_obj = (-penal * x ** (penal - 1)) * ce
		grad_obj = (H * grad_obj)/Hs

		cons = _volume_constraint(x, volfrac)
		grad_cons = np.ones(num_elems)/volfrac/num_elems

		timeMMAStart = time.time()
		mma_state = mma.update_mma(mma_state,
														   mma_params,
														 	 obj,
															 np.array([grad_obj]).reshape((num_elems, 1)),
														 	 jnp.array([cons]).reshape((1, 1)),
															 grad_cons.reshape((1, num_elems))
															 )
		timeMMA += time.time() - timeMMAStart

		change = np.max(np.abs(x - x_old))
		x_old = x
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.3f} vc: {cons:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
		history['change'].append(change)

	fe_solver.mesh.setPseudoDensity(x)
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	return np.asarray(u), history


def topopt_optimality_criteria(
							fe_solver: sfea.StructFEA,
			  				maxIterations: int = 500,
							volfrac: float = 0.5,
							penal: float = 3,
							move: float = 0.2,
							conv_tol: float = 0.025,
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

	H, Hs = createSmoothingFilter(fe_solver.mesh)

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
		lmid = 0.5 * (l2 + l1)
		# Bisection loop for volume constraint
		
		while (l2 - l1) > 1e-7:
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
		#change = jnp.linalg.norm(x - xold, np.inf)
		change = jnp.max(jnp.abs(x - xold))
		fe_solver.mesh.setPseudoDensity(np.asarray(xPhys))
	
		history['compliance'].append(obj)
		history['volume'].append(np.mean(xPhys))
		history['change'].append(change)

		if verbose:
			print(f"it.: {iter+1:d}, obj.: {obj:.3e}, "
				  	f"vol.: {np.mean(xPhys):.3f}, ch.: {change:.3f}")

		if change < conv_tol:
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
	totalIter = 1

	# Initialize design field
	rho = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'compliance': [], 'volume': []}

	# Create filter
	H, Hs = createSmoothingFilter(fe_solver.mesh)
	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	u = np.asarray(fe_solver.solve(rho))

	# Store initial compliance
	history['compliance'].append(fe_solver.bc.force.T @ u)
	history['volume'].append(volfrac)

	# Compute initial topological sensitivity
	T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
	T[elemsWithForces] = np.max(T)
	T = (H * T) / Hs
	
	print(f"v={volfrac:.2f}; J={history['compliance'][-1]:.2e}; #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	while volfrac > desiredVolFrac:
		# Move to next volume fraction
		volfrac = max(desiredVolFrac, volfrac - vol_decr)

		localIter = 0
		success = False
		JTemp = history['compliance'][-1]  # Store previous value
		JPrev = float('inf')  # Initialize JPrev
		JPrevPrev = float('inf')  # Initialize JPrev
		while localIter < max_local_iters:
			if JTemp > 10 * history['compliance'][-1]:  # Divergence check
				break
			# Check convergence, and break if converged
			if localIter >= min_local_iters and abs(min(JPrev,JPrevPrev) - JTemp)/JTemp < rel_err:
				success = True
				break

			# Find cutoff value and update design
			value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
			rho = np.ones((fe_solver.mesh.num_elems))

			rho = (T - min(T))/(max(T) - min(T))
			rho = volfrac*rho/np.mean(rho)+0.01
			#rho[T < value] = (0.01)
			
			JPrevPrev = JPrev  # Store previous value
			JPrev = JTemp  # Store previous value
			
			u = np.asarray(fe_solver.solve(rho))
			JTemp = float(fe_solver.bc.force.T @ u)
			
			# Update sensitivity
			T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
			T[elemsWithForces] = np.max(T)
			T = (H * T) / Hs
			localIter += 1
			totalIter += 1

		if not success:
			print("Pareto: Failed to converge in local iterations.")
			print("Forcing symmetry, if applicable, can help.")
			break
		
		history['compliance'].append(JTemp)
		history['volume'].append(volfrac)
		dJdvNormalized = 0
		if (len(history['compliance'])) >= 2:
			dJdv = (history['compliance'][-1] - history['compliance'][-2]) / (history['volume'][-1] - history['volume'][-2])
			dJdvNormalized = abs(dJdv / history['compliance'][0])
		vol_decr = vol_decr_max/np.sqrt(1 + dJdvNormalized)  # Adjust volume decrease factor
		vol_decr = max(	0.01, min(vol_decr, vol_decr_max))  # Limit volume decrease factor
		print(f"v={history['volume'][-1]:.3f}; J={history['compliance'][-1]:.2e};  #FEA={totalIter:2d}")
		
		fe_solver.mesh.setPseudoDensity(rho.flatten())

	return u, history

def topopt_scipy(fe_solver: sfea.StructFEA,
			  				maxIterations: int = 500,
							volfrac: float = 0.5,
							penal: float = 3,
							verbose: bool = True) -> tuple[np.ndarray, dict]:
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
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	H, Hs = createSmoothingFilter(fe_solver.mesh)

	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)

	x0 = volfrac*np.ones(num_elems, dtype = float)


	def volume_constraint(x):
		return _volume_constraint(x, volfrac)

	def objective_with_grad(x):
		print(np.min(x), np.max(x))
		obj, u = _compliance_objective(x, fe_solver, penal)
		print(obj)
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * 
			  u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		grad = (-penal * x ** (penal - 1)) * ce
		grad = (H * grad)/Hs
		return float(obj), grad

	def constraint_grad(x):
		return np.ones(num_elems)/volfrac/num_elems

	bounds = [(0.001, 1) for _ in range(num_elems)]
	result = minimize(
		objective_with_grad,
		x0,
		method='trust-constr',
		jac=True,
		constraints={'type': 'eq', 'fun': volume_constraint, 'jac': constraint_grad},
		bounds=bounds,
		options={'maxiter': maxIterations, 'disp': True}
	)

	u = fe_solver.solve(result.x)
	fe_solver.mesh.setPseudoDensity(result.x)

	return np.asarray(u), history
	
def topopt_nlopt(fe_solver: sfea.StructFEA,
			  				maxIterations: int = 500,
							volfrac: float = 0.5,
							penal: float = 3,
							verbose: bool = True) -> tuple[np.ndarray, dict]:
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
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	H, Hs = createSmoothingFilter(fe_solver.mesh)
	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)

	def objective_with_grad(x,grad):
		print(np.min(x), np.max(x))
		obj, u = _compliance_objective(x, fe_solver, penal)
		print(obj)
		if (grad.size > 0):
			ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * 
			  u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
			grad = (-penal * x ** (penal - 1)) * ce
			grad = (H * grad)/Hs
		return float(obj)

	def constraint_grad(x, grad):
		if (grad.size >0):
			grad[:] = np.ones(num_elems)/volfrac/num_elems
		constraint = (np.mean(x)/volfrac) - 1.0
		print(constraint)
		return constraint

	opt = nlopt.opt(nlopt.LD_SLSQP, num_elems)  # LD_SLSQP algorithm
	opt.set_min_objective(objective_with_grad)
	opt.add_equality_constraint(constraint_grad)
	opt.set_lower_bounds(0.001*np.ones(num_elems))
	opt.set_upper_bounds(np.ones(num_elems))
	opt.set_maxeval(200)   # Stop after iterations
	opt.set_maxtime(400)  # Stop after seconds

	x0 = volfrac*np.ones(num_elems, dtype = float)
	x_opt = opt.optimize(x0)
	u = fe_solver.solve(x_opt)
	fe_solver.mesh.setPseudoDensity(x_opt)
	return np.asarray(u)
	
if __name__ == "__main__":    
	from examples_structural import createCantileverProblem, createLBracketProblem
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time

	import plots	
	jax.config.update("jax_enable_x64", True)
	dsolver = deflation.DeflationSolver()

	example = 2
	nDOFDesired = 20000
	if example == 1:
		mesh, mat_prop, bc = createCantileverProblem(nDOFDesired = nDOFDesired,L = [0.4,0.2,0.1])
	elif example == 2:
		mesh, mat_prop, bc = createLBracketProblem(nDOFDesired = nDOFDesired)    

	#plots.plotMesh(mesh, bc)

	solver = lin_solv.Solvers.PARDISO
		
	fe_solver = fea.StructFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = solver)
	
	youngs_modulus = np.ones((fe_solver.mesh.num_elems,))

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	
	
	volfrac = 0.5
	num_iter = 200

	optimizationMethod = 2 # 1: MMA, 2: OC, 3: Pareto, 4: Scipy, 5: NLOPT

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
	elif optimizationMethod == 4:
		print("optimizationMethod: Scipy")
		u, history = topopt_scipy(fe_solver = fe_solver,
										volfrac =  volfrac)
		
		timeTaken = time.time() - startTime
		title = f'Pareto: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3e}, time: {timeTaken:.2f} s'
	elif optimizationMethod == 5:
		print("optimizationMethod: NLOPT")
		u = topopt_nlopt(fe_solver = fe_solver,
										volfrac =  volfrac)
		
		timeTaken = time.time() - startTime
		title = '' 

	plots.plotMesh(fe_solver.mesh, fe_solver.bc, u, title = title)
