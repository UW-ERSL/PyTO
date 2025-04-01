"""Optimization routines for topology optimization."""

import enum
import numpy as np
import jax
import jax.numpy as jnp
import element_stiffness as elem_stiff
import mesher
import struct_fea as sfea
import mma
import deflation
from to_filters import *
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
_LARGE_NUMBER = 1.e9


class TO_METHODS(enum.Enum):
	DENSITYMMA = enum.auto()
	DENSITYOC = enum.auto()
	PARETO = enum.auto()
	LEVELSET = enum.auto()

class MaterialModel(enum.Enum):
	SIMP = enum.auto()
	RAMP = enum.auto()
	SIMPPLUS = enum.auto()


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


def _compliance_objective(x: jnp.ndarray,
								fe_solver: sfea.StructFEA,
													material_model_dict = None,
													) -> jnp.ndarray:
	"""Compute the structural compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	u = fe_solver.solve(x, material_model_dict)
	return jnp.einsum('i, i -> ', fe_solver.total_force, u), u


def createFilters(fe_solver: sfea.StructFEA,to_params):
	# Create  filters
	print("Computing filters...")
	H, Hs = createSmoothingFilter(fe_solver.mesh, rel_filter_radius=to_params.RelativeFilterRadius)
	# Accumulate all other filters
	if to_params.XSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
		H = H*HX
	if to_params.YSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
		H = H*HY
	if to_params.ZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)
		H = H*HZ
	if to_params.ZAxisAngularSymmetry >	0:
		HAZ = createAngularSymmetryFilter(fe_solver.mesh, to_params.ZAxisAngularSymmetry)
		H = H*HAZ
	if (to_params.ExtrudeZ):
		HEZ = createZExtrudeFilter(fe_solver.mesh)
		H = H*HEZ
	if (to_params.AMBuildConstraint):
		HZAM = createAMBuildFilter(fe_solver.mesh)
		H = H*HZAM

	return H, Hs

def topopt_mma(fe_solver: sfea.StructFEA,
			   			to_params,
			   			minMMAIterations: int = 5,
			   			 maxMMAIterations: int = 250, 
							timeLimit: float =3600, #1 hour
						   penal: float = 3.0,
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-6,
							 move_tol: float = 0.025,
							 rel_conv_tol: float = 1.e-4,
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
	elem_body_force = fe_solver.elem_body_force
	if elem_body_force is None or (np.linalg.norm(elem_body_force) == 0):
		material_model = MaterialModel.SIMP #For no body forces, using SIMP material model
		material_model_dict = {'name': 'SIMP', 'penal': 3.0} # Default SIMP model
	else:
		material_model = MaterialModel.SIMPPLUS #For body forces, using SIMPPLUS material model
		material_model_dict = {'name': 'SIMPPLUS', 'penal': 3, 'alpha': 16} 
		#  body-force model from the papers here:
		#  https://doi.org/10.1002/nme.2499, https://doi.org/10.1016/j.cma.2017.04.021 


	tStart = time.time()
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	
	[H,Hs] = createFilters(fe_solver, to_params)

	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	xMin = 1e-10
	mma_params = mma.MMAParams(max_iter=maxMMAIterations,
														kkt_tol = kkt_tol,
														step_tol = move_tol,
														move_limit = move_limit,
														num_design_var = num_elems,
														num_cons = 1,
														lower_bound = xMin*np.ones((num_elems, 1)),
														upper_bound = np.ones((num_elems, 1)),
														)
	mma_state = mma.init_mma(to_params.DesiredVolFraction * np.ones((num_elems, 1)), mma_params)
	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	x_old = to_params.DesiredVolFraction *np.ones(num_elems, dtype = float)
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
	
	while not mma_state.is_converged:
		x = mma_state.x.reshape(-1)
		timeFEAStart = time.time()
		obj,u = _compliance_objective(x, fe_solver, material_model_dict)
		
		timeFEA += time.time() - timeFEAStart
		obj = np.array([obj])
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
			grad_obj +=  2*ce_body_force # Assumes body force is linear w.r.t. x : https://doi.org/10.1002/nme.2499 , https://doi.org/10.1016/j.cma.2017.04.021 
	
		grad_obj = (H * grad_obj)/Hs


		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj)

		if (to_params.ElemsToKeep is not None):
			grad_obj[to_params.ElemsToKeep] = min(grad_obj)
			#x[to_params.ElemsToKeep] = 1.0

		vf = np.mean(x)
		cons = _volume_constraint(x, to_params.DesiredVolFraction)
		grad_cons = np.ones(num_elems)/to_params.DesiredVolFraction/num_elems

		
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
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.6g} vf: {vf:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
		history['change'].append(change)

		if (len(history['compliance'])) >= minMMAIterations:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if abs(dJ) < rel_conv_tol and (cons) < rel_conv_tol:
				break
		if time.time() - tStart > timeLimit:
			success = False
			print("MMA optimization terminated due to time limit.")
			break
		# if (history['compliance'][-1] > 100*history['compliance'][0]):
		# 	print("Optimization terminated due to large compliance increase.")
		# 	success = False
		# 	break

	if mma_state.epoch >= maxMMAIterations:
		print("MMA optimization did not converge.")
		success = False
		
	fe_solver.mesh.setPseudoDensity(x)
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	print(f"Total Time: {timeFEA+timeMMA:.2f} s")
	return np.asarray(u), history,success


def topopt_optimality_criteria(
							fe_solver: sfea.StructFEA,
							to_params,
			  				maxIterations: int = 250,
							penal: float = 3,
							move: float = 0.2,
							move_tol: float = 0.025,
							rel_conv_tol: float = 1.e-4,
							directLagrangeMethod: bool = True,
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
	KE = elem_stiff.hex8_stiffness_matrix_structural( fe_solver.mat_prop,fe_solver.mesh.elem_size)
	
	success = True
	for iter in range(maxIterations):
		x = np.array(x)
		obj,u = _compliance_objective(x, fe_solver,material_model_dict)
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

		cons = _volume_constraint(x, to_params.DesiredVolFraction)
		# Optimality criteria update
		xold = x.copy()
		if  not directLagrangeMethod: # bisection method
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
				xnew = varTimesGrad/ (np.sum(varTimesGrad[varIn]) /volToDistribute)
				volToDistribute = to_params.DesiredVolFraction*num_elems -np.sum(xMax[xnew>=xMax]) -np.sum(xMin[xnew<=xMin])
				setChange = np.sum(xnew) - to_params.DesiredVolFraction * num_elems > 0
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
		
		print(f"it.: {iter+1:d}, obj.: {obj:.5g}, "
				  	f"vol.: {np.mean(xPhys):.3g}, ch.: {change:.3f}")
		if np.isnan(obj):
			print("Objective function became NaN. Exiting optimization.")
			success = False
			break
		if (change < move_tol):
			break
		if (len(history['compliance'])) >= 2:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if (abs(dJ) < rel_conv_tol and abs(cons) < rel_conv_tol):
				break

	if iter == maxIterations - 1:
		print("Maximum iterations reached without convergence.")
		success = False

	return np.asarray(u), history, success


def topopt_pareto(fe_solver: sfea.StructFEA,
				  to_params,
							rel_err: float = 0.025,
							vol_decr_max: float = 0.05,
							vol_decr_min: float = 0.001,
							min_local_iters: int = 2,
							max_local_iters: int = 10,
							rhoVoid: float = 0,
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
	import numpy as np

	def computeTopologicalSensitivity(mesh, mat_prop, u, rho):
		"""Compute topological sensitivity field. Vectorized version."""
		
		num_elems = mesh.num_elems
		T = np.zeros(num_elems)
		e, nu = mat_prop.youngs_modulus, mat_prop.poissons_ratio
		
		# Create constitutive matrix
		v1 = 2 * nu**2 + nu - 1
		v2 = 2 * nu + 2
		D = e * np.array([
			[(nu - 1) / v1, -nu / v1, -nu / v1, 0, 0, 0],
			[-nu / v1, (nu - 1) / v1, -nu / v1, 0, 0, 0],
			[-nu / v1, -nu / v1, (nu - 1) / v1, 0, 0, 0],
			[0, 0, 0, 1 / v2, 0, 0],
			[0, 0, 0, 0, 1 / v2, 0],
			[0, 0, 0, 0, 0, 1 / v2]
		])
		
		# Shape function gradients at center
		gradN = (1 / 8) * np.array([
			[-1, 1, 1, -1, -1, 1, 1, -1],
			[-1, -1, 1, 1, -1, -1, 1, 1],
			[-1, -1, -1, -1, 1, 1, 1, 1]
		])
		
		# Get element degrees of freedom
		edof = mesh.edofMat
		
		# Compute displacement gradients
		uGrad = gradN @ u[edof[:, ::3]].T
		vGrad = gradN @ u[edof[:, 1::3]].T
		wGrad = gradN @ u[edof[:, 2::3]].T
		
		# Compute strains
		strains = np.stack([
			uGrad[0], vGrad[1], wGrad[2],
			uGrad[1] + vGrad[0],
			uGrad[2] + wGrad[0],
			vGrad[2] + wGrad[1]
		], axis=1)  # Shape: (num_elems, 6)
		
		# Compute stresses
		stresses = strains @ D.T  # Shape: (num_elems, 6)
		
		# Create stress and strain tensors
		stress_tensor = rho[:, None, None] * np.array([
			[stresses[:, 0], stresses[:, 3], stresses[:, 4]],
			[stresses[:, 3], stresses[:, 1], stresses[:, 5]],
			[stresses[:, 4], stresses[:, 5], stresses[:, 2]]
		]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
		
		strain_tensor = np.array([
			[strains[:, 0], strains[:, 3], strains[:, 4]],
			[strains[:, 3], strains[:, 1], strains[:, 5]],
			[strains[:, 4], strains[:, 5], strains[:, 2]]
		]).transpose(2, 0, 1)  # Shape: (num_elems, 3, 3)
		
		# Compute topological sensitivity
		trace_stress = np.trace(stress_tensor, axis1=1, axis2=2)
		trace_strain = np.trace(strain_tensor, axis1=1, axis2=2)
		T = (4 / (1 + nu) * np.sum(stress_tensor * strain_tensor, axis=(1, 2)) -
			(1 - 3 * nu) / (1 - nu**2) * trace_stress * trace_strain)

		return T


	removeHangingElems = to_params.RemoveHangingElems
	if fe_solver.elem_body_force is not None and (np.linalg.norm(fe_solver.elem_body_force) > 0) and not removeHangingElems:
		removeHangingElems = True #For body forces, must remove hanging elements in Pareto

	totalIter = 1

	# Initialize design field
	rho = np.ones((fe_solver.mesh.num_elems))
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
	u = np.asarray(fe_solver.solve(rho))

	# Store initial compliance
	history['compliance'].append(fe_solver.total_force.T @ u)
	history['volume'].append(volfrac)

	# Compute initial topological sensitivity
	T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)

	# Add contribution from body force to topological sensitivity if present
	if (nodal_body_force is not None):
		T_body = np.zeros(fe_solver.mesh.num_elems)
		for elem in range(fe_solver.mesh.num_elems):
			edof = fe_solver.mesh.edofMat[elem]
			T_body[elem] =  (rho[elem]*u[edof] * nodal_body_force[edof]).sum()
		T += 2*T_body

	if (elemsWithForces.size > 0): #For pure body forces, this may be empty
		T[elemsWithForces] = np.max(T)
	if (to_params.ElemsToKeep is not None):
		T[to_params.ElemsToKeep] = np.max(T)
	T = (H * T) / Hs


	print(f"vf={history['volume'][-1]:.3f},  J={history['compliance'][-1]:.3g},  #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	wtDamping = 0.5 # 0 means full wt to current T values, else previous T values are damped in

	success = True
	terminatePareto = False
	while volfrac > to_params.DesiredVolFraction:
		# Move to next volume fraction
		volfrac = max(to_params.DesiredVolFraction, volfrac - vol_decr)
		if (debug):
			print("-" * 50)
			print(f"Attempting v={volfrac:.3f}")
		# Initialize local iteration variables
		localIter = 0
		JTemp = history['compliance'][-1]  # Store previous value
		JPrev = float('inf')  # Initialize JPrev
		JPrevPrev = float('inf')  # Initialize JPrev
		TPrev = T.copy()  # Store previous sensitivity
		rhoPrev = rho.copy()  # Store previous design
		innerLoopSuccess = True
		while True:
			if (debug):
				print(f"Local Iteration: {localIter}/{max_local_iters}, JTemp: {JTemp:.3g}")
			if (localIter >= max_local_iters) or abs(JTemp) > 10 * history['compliance'][-1]:  # large change in compliance	
				innerLoopSuccess = False
				rho = rhoPrev.copy()
				T = TPrev.copy()
				fe_solver.mesh.setPseudoDensity(rho.flatten())
				JTemp = JPrev
				volfrac = volfrac + vol_decr # Restore volume fraction
				vol_decr *= 0.75 # Reduce volume decrement
				if (debug):
					print(f"Decrementing vol_decr to: {vol_decr:.5g}")
				if vol_decr < vol_decr_min:
					terminatePareto = True
				break
			# Check convergence, and break if converged
			if localIter >= min_local_iters:
				if abs(JPrev - JTemp)/JTemp < rel_err or abs(min(JPrev,JPrevPrev) - JTemp)/JTemp < rel_err:
					innerLoopSuccess = True
					break

			# Find cutoff value and update design
			value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
			rho = np.ones((fe_solver.mesh.num_elems))
			rho[T < value] = rhoVoid
			fe_solver.mesh.setPseudoDensity(rho.flatten())
			if (removeHangingElems):
				meshComponents = fe_solver.mesh.find_connected_components()
				if (len(meshComponents) > 1):
					# Find the largest connected component and its size
					largest_component = max(meshComponents, key=len)
					# Set density to rhoVoid for all elements
					rho[:] = rhoVoid
					# Set density to 1 for elements in largest component
					rho[list(largest_component)] = 1.0
					fe_solver.mesh.setPseudoDensity(rho.flatten())
			
			JPrevPrev = JPrev  # Store previous to previous value
			JPrev = JTemp  # Store previous value
			u = np.asarray(fe_solver.solve(rho))
			JTemp = float(fe_solver.total_force.T @ u)
			#plots.plotMesh(fe_solver.mesh, bc = None, u=u, title = title)
			# Update sensitivity
			T = computeTopologicalSensitivity(fe_solver.mesh, fe_solver.mat_prop, u, rho)
			# Add contribution from body force to topological sensitivity if present
			if (nodal_body_force is not None):
				T_body = np.zeros(fe_solver.mesh.num_elems)
				for elem in range(fe_solver.mesh.num_elems):
					edof = fe_solver.mesh.edofMat[elem]
					T_body[elem] =  (rho[elem]*u[edof] * nodal_body_force[edof]).sum()
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
			success = False
			print("-" * 50)
			print("Pareto: Failed to reach volume fraction.")
			print("Recommendations:")	
			print("1. Check for incorrect symmetry constraints")
			print("2. Increase mesh size")
			print("3. Decrease vol_decr_max parameter")
			break
		if innerLoopSuccess:
			history['compliance'].append(JTemp)
			history['volume'].append(volfrac)
			#scale = history['compliance'][-1] / history['compliance'][0]
			#vol_decr = max(vol_decr_min,vol_decr/scale**2) # Adjust volume decrease factor for steep increase in compliance
			print(f"vf={history['volume'][-1]:.3f}, J={history['compliance'][-1]:.3g},   #FEA={totalIter:2d}")
			fe_solver.mesh.setPseudoDensity(rho.flatten())
			
	return u, history, success

def topopt_levelset(fe_solver: sfea.StructFEA,
					 to_params,
						 maxIterations: int = 250,
						 volfrac: float = 0.5,
						 time_step: float = 0.1,
						 epsilon: float = 1.0,
						 rel_conv_tol: float = 1e-4,
						 
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
			obj, u = _compliance_objective(density, fe_solver, material_model_dict)
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
		return u, history, success


def runTOTests():

	# Create a list to store results
	results_list = []
	dsolver = deflation.DeflationSolver()
	for to_problem in StructuralTOExamples:
		if (to_problem == StructuralTOExamples.BliskWithBlade):
			continue
		print("-" * 50)
		print(f"Running {to_problem.name}...")
		print("-" * 50)

		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

		if (to_params.nDOFDesired < 200000):# Typically PARDISO 
			print("Solver: Pardiso")
			solver = lin_solv.Solvers.PARDISO
		else: #DPCG for DOF > 200,000
			print("Solver: DPCG")
			solver = lin_solv.Solvers.DPCG 
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
		startTime = time.time()
		if optimizationMethod == TO_METHODS.DENSITYMMA:
			u, history,success = topopt_mma(fe_solver = fe_solver,
									to_params = to_params)
		elif optimizationMethod == TO_METHODS.DENSITYOC:
			u, history, success = topopt_optimality_criteria(fe_solver = fe_solver,
											to_params = to_params)
		elif optimizationMethod == TO_METHODS.PARETO:
			u, history, success = topopt_pareto(fe_solver = fe_solver,
													to_params = to_params)
		elif optimizationMethod == TO_METHODS.LEVELSET:
			u, history, success = topopt_levelset(fe_solver = fe_solver,
													to_params = to_params)
		timeTaken = time.time() - startTime

		# Create the directory if it does not exist
		output_dir = f"./Results_{time.strftime('%Y-%m-%d')}/{optimizationMethod.name}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		image_path = f"{output_dir}/{to_problem.name}.png"
		title = f"{optimizationMethod.name}: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
	
		plots.plotMesh(fe_solver.mesh, bc = None, u=None, save_path = image_path, title = title)
		
		
		results_list.append({
			'name': to_problem.name,
			'comment': to_params.Comment,  
			'ndof': 3*fe_solver.mesh.num_nodes,
			'volume': history['volume'][-1],
			'compliance': history['compliance'][-1],
			'time (s)': timeTaken,
			'success': success
		})
	

	# Convert results_list to a DataFrame for better visualization
	results_df = pd.DataFrame(results_list)

	# Format
	results_df['volume'] = results_df['volume'].map(lambda x: f"{x:.3g}")
	results_df['compliance'] = results_df['compliance'].map(lambda x: f"{x:.3g}")
	results_df['time (s)'] = results_df['time (s)'].map(lambda x: f"{x:.3g}")

	# Plot the results as a table
	fig, ax = plt.subplots(figsize=(10, len(results_list) * 0.5))
	ax.axis('tight')
	ax.axis('off')
	table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.auto_set_column_width(col=list(range(len(results_df.columns))))

	# Make the first row and column bold
	for key, cell in table.get_celld().items():
		if key[0] == 0 or key[1] == 0:  # Header row
			cell.set_text_props(weight='bold')
	
	# Save the table as an image
	results_path = f"{output_dir}/{optimizationMethod.name}_summary.png"
	plt.savefig(results_path, bbox_inches='tight')

	
if __name__ == "__main__":    
	from examples_topology_optimization import *
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time
	import matplotlib.pyplot as plt
	import deflation
	import os
	import pandas as pd
	import plots	
	
	jax.config.update("jax_enable_x64", True)
	optimizationMethod = TO_METHODS.DENSITYMMA # DENSITYMMA, DENSITYOC, PARETO, LEVELSET

	#runTOTests(); exit(0) # Run all tests for each example in the StructuralTOExamples enum
	
	# Choose the TO problem
	to_problem = StructuralTOExamples.GravityPlate
	solver = lin_solv.Solvers.PARDISO # Typically PARDISO, but DPCG for DOF > 200,000
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
	#plots.plotMesh(mesh, bc,title = title)


	startTime = time.time()
	if optimizationMethod == TO_METHODS.DENSITYMMA:
		print("OptimizationMethod: MMA")
		u, history,success = topopt_mma(fe_solver = fe_solver,
						  			to_params = to_params,
									debug = debug)
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
		plt.show(block=False)

		title = f"MMA: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

	elif optimizationMethod == TO_METHODS.DENSITYOC:
		print("OptimizationMethod: OC")
		u, history, success = topopt_optimality_criteria(fe_solver = fe_solver,
										  		to_params = to_params,
												debug = debug)
		timeTaken = time.time() - startTime
		title = f"OC: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"

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
		plt.show(block=False)
	
	elif optimizationMethod == TO_METHODS.PARETO:
		print("OptimizationMethod: Pareto")
		u, history, success = topopt_pareto(fe_solver = fe_solver,
										to_params = to_params,
										debug = debug)
		
		timeTaken = time.time() - startTime
		title = f"Pareto: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
		
		# Plot volume vs compliance history
		plt.figure()
		plt.plot(history['volume'], history['compliance'], marker='o')
		plt.xlabel('Volume Fraction')
		plt.ylabel('Compliance')
		plt.title('Pareto: Volume vs Compliance History')
		plt.grid(True)
		plt.show(block=False)
	elif optimizationMethod == TO_METHODS.LEVELSET:
		print("OptimizationMethod: Level Set")
		u, history, success = topopt_levelset(fe_solver = fe_solver,
										to_params = to_params,
										maxIterations = 100,
										time_step = 0.1,
										epsilon = 1.0,
										rel_conv_tol = 1e-4,
										debug = debug)
		timeTaken = time.time() - startTime
		title = f"Level Set: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"	

	print(f"Time taken: {timeTaken:.0f} s")
	
	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title)