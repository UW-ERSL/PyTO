"""Optimization routines for topology optimization."""

import enum
import numpy as np
import jax
import jax.numpy as jnp
import element_stiffness as elem_stiff
import mesher
import mat_lib
import struct_fea as sfea
import mma
import deflation
from TOfilters import createXSymmetryFilter, createYSymmetryFilter, createZSymmetryFilter, createSmoothingFilter

_LARGE_NUMBER = 1.e9


class Optimizers(enum.Enum):
	MMA = enum.auto()
	OC = enum.auto()
	PARETO = enum.auto()

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
							 kkt_tol: float = 1.e-6,
							 step_tol: float = 0.025,
							 exitOnConvergence: bool = True
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
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.3g} vc: {cons:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
		history['change'].append(change)
		if exitOnConvergence and (len(history['compliance'])) >= 2:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if abs(dJ) < 1e-5 and abs(cons) < 1e-5:
				break


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
							verbose: bool = True,
							exitOnConvergence: bool = True
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

		cons = _volume_constraint(x, volfrac)
		fe_solver.mesh.setPseudoDensity(np.asarray(xPhys))
	
		history['compliance'].append(obj)
		history['volume'].append(np.mean(xPhys))
		history['change'].append(change)

		if verbose:
			print(f"it.: {iter+1:d}, obj.: {obj:.3g}, "
				  	f"vol.: {np.mean(xPhys):.3g}, ch.: {change:.3f}")
			

		if change < conv_tol:
			break
		if exitOnConvergence and (len(history['compliance'])) >= 2:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if abs(dJ) < 1e-4 and abs(cons) < 1e-3:
				break


	return np.asarray(u), history


def topopt_pareto(fe_solver: sfea.StructFEA,
							desiredVolFrac: float = 0.5,
							rel_err: float = 0.025,
							vol_decr_max: float = 0.05,
							min_local_iters: int = 1,
							max_local_iters: int = 10,
							rhoVoid: float = 0,
							imposeXSymmetry: bool = False,
							imposeYSymmetry: bool = False,
							imposeZSymmetry: bool = False
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
			stresses =  D @ strains

			# Create tensors
			stress_tensor = rho[elem]*np.array([
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
	if imposeXSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
	if imposeYSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
	if imposeZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)

	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	u = np.asarray(fe_solver.solve(rho))

	# Store initial compliance
	history['compliance'].append(fe_solver.bc.force.T @ u)
	history['volume'].append(volfrac)

	# Compute initial topological sensitivity
	T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
	if imposeXSymmetry:
		T = (HX * T)	
	if imposeYSymmetry:
		T = (HY * T)
	if imposeZSymmetry:
		T = (HZ * T)	
	T[elemsWithForces] = np.max(T)
	T = (H * T) / Hs
	print(f"v={volfrac:.2f}; J={history['compliance'][-1]:.2g}; #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	wtDamping = 0.5
	while volfrac > desiredVolFrac:
		# Move to next volume fraction
		volfrac = max(desiredVolFrac, volfrac - vol_decr)

		localIter = 0
		success = False
		JTemp = history['compliance'][-1]  # Store previous value
		JPrev = float('inf')  # Initialize JPrev
		JPrevPrev = float('inf')  # Initialize JPrev
		TPrev = T.copy()  # Store previous sensitivity
		while localIter < max_local_iters:
			#print(JTemp)
			if abs(JTemp) > 10 * history['compliance'][-1]:  # Divergence check	
				vol_decr_max /= 2 # Reduce max step size
				print(f"Pareto: Reducing vol_decr_max to {vol_decr_max:.3f}")
				if (vol_decr_max < 1e-3):
					print("Pareto: Failed to reach volume fraction.")
					print("Recommendations:")
					print("1. Check for incorrect symmetry constraints")
					print("2. Decrease vol_decr_max parameter")
					print("3. Increase mesh size")
					success = False
					break
				
				volfrac = volfrac + vol_decr # go back to previous step
				vol_decr = vol_decr_max/scale**2
				# Need to revert changes and try again with smaller step
				rho = np.ones((fe_solver.mesh.num_elems))
				T = TPrev.copy()
				JTemp = JPrev
				localIter = 0
				continue
			# Check convergence, and break if converged
			if localIter >= min_local_iters:
				if abs(JPrev - JTemp)/JTemp < rel_err or abs(min(JPrev,JPrevPrev) - JTemp)/JTemp < rel_err:
					success = True
					break

			# Find cutoff value and update design
			value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
			rho = np.ones((fe_solver.mesh.num_elems))
			rho[T < value] = rhoVoid
			
			JPrevPrev = JPrev  # Store previous to previous value
			JPrev = JTemp  # Store previous value
			
			u = np.asarray(fe_solver.solve(rho))
			JTemp = float(fe_solver.bc.force.T @ u)
			
			# Update sensitivity
			T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
			T = (H * T) / Hs
			T = ((1-wtDamping)*T + wtDamping*TPrev)  # Damping
			if imposeXSymmetry:
				T = (HX * T)	
			if imposeYSymmetry:
				T = (HY * T)
			if imposeZSymmetry:
				T = (HZ * T)	
			T[elemsWithForces] = np.max(T)

			localIter += 1
			totalIter += 1

		if not success:
			break
		
		history['compliance'].append(JTemp)
		history['volume'].append(volfrac)
		scale = history['compliance'][-1] / history['compliance'][0]
		vol_decr = vol_decr_max/scale**2  # Adjust volume decrease factor for steep increase in compliance
		print(f"v={history['volume'][-1]:.3f}; J={history['compliance'][-1]:.3g};  #FEA={totalIter:2d}")
		
		fe_solver.mesh.setPseudoDensity(rho.flatten())
		
			
	return u, history

if __name__ == "__main__":    
	from examples_structural import *
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time
	import matplotlib.pyplot as plt

	import plots	
	jax.config.update("jax_enable_x64", True)
	dsolver = deflation.DeflationSolver()

	example = 1
	nDOFDesired = 20000
	if example == 1:
		mesh, mat_prop, bc = createCantileverProblem(nDOFDesired = nDOFDesired,L = [0.4,0.2,0.1])
		imposeXSymmetry = False
		imposeYSymmetry = True
		imposeZSymmetry = False
	elif example == 2:
		mesh, mat_prop, bc = createMBBProblem(nDOFDesired = nDOFDesired)
		imposeXSymmetry = False
		imposeYSymmetry = False
		imposeZSymmetry = False
	elif example == 3:
		mesh, mat_prop, bc = createDistributedLoadProblem(nDOFDesired = nDOFDesired)
		imposeXSymmetry = True
		imposeYSymmetry = False
		imposeZSymmetry = False
	elif example == 4:
		mesh, mat_prop, bc = createMultiloadProblem(nDOFDesired = nDOFDesired)
		imposeXSymmetry = False
		imposeYSymmetry = False
		imposeZSymmetry = True
	elif example == 5:
		mesh, mat_prop, bc = createLBracketProblem(nDOFDesired = nDOFDesired)    
		imposeXSymmetry = False
		imposeYSymmetry = False
		imposeZSymmetry = False

	solver = lin_solv.Solvers.PARDISO
		
	fe_solver = fea.StructFEA(mesh = mesh,
				mat_prop = mat_prop,
				bc = bc,
				solver = solver)
	

	print('Solver: ', fe_solver.solver.name)
	print("nDof: ", 3*fe_solver.mesh.num_nodes)
	print("nElem: ", fe_solver.mesh.num_elems)	
	
	title = f'nDOF: {3*fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
	#plots.plotMesh(mesh, bc,title = title)
	volfrac = 0.2
	num_iter = 250

	optimizationMethod = 3 # 1: MMA, 2: OC, 3: Pareto

	startTime = time.time()
	if optimizationMethod == 1:
		print("optimizationMethod: MMA")
		u, history = topopt_mma(fe_solver = fe_solver,
									maxMMAIterations = num_iter,
														volfrac = volfrac,
														exitOnConvergence=True)
		timeTaken = time.time() - startTime
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

		plt.title('MMA: Volume and Compliance vs. Iterations')

		# Add legend
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2)

		plt.grid(True)
		plt.show()

		title = f'MMA: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s'
	elif optimizationMethod == 2:
		print("optimizationMethod: OC")
		u, history = topopt_optimality_criteria(fe_solver = fe_solver,
												maxIterations= num_iter,
												volfrac = volfrac,
												exitOnConvergence=True
												)
		timeTaken = time.time() - startTime
		title = f'OC: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s'

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

		plt.title('MMA: Volume and Compliance vs. Iterations')

		# Add legend
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2)

		plt.grid(True)
		plt.show()

	elif optimizationMethod == 3:
		print("optimizationMethod: Pareto")
		u, history = topopt_pareto(fe_solver = fe_solver,
										desiredVolFrac =  volfrac,imposeXSymmetry=imposeXSymmetry,
										imposeYSymmetry=imposeYSymmetry,imposeZSymmetry=imposeZSymmetry)
		
		timeTaken = time.time() - startTime
		title = f'Pareto: vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s'
		
		# Plot volume vs compliance history
		plt.figure()
		plt.plot(history['volume'], history['compliance'], marker='o')
		plt.xlabel('Volume Fraction')
		plt.ylabel('Compliance')
		plt.title('Pareto: Volume vs Compliance History')
		plt.grid(True)
		plt.show()

	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title)
