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
from TOfilters import *
import sys
import time

_LARGE_NUMBER = 1.e9


class Optimizers(enum.Enum):
	MMA = enum.auto()
	OC = enum.auto()
	PARETO = enum.auto()

class MaterialModel(enum.Enum):
	SIMP = enum.auto()
	RAMP = enum.auto()
	SIMPPLUS = enum.auto()


def find_elements_with_fixedDOF(fe_solver: sfea.StructFEA,) -> np.ndarray:
	"""Find all elements that have nodes with fixed degrees of freedom.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with fixed degrees of freedom.
	"""
	fixed_dofs = fe_solver.bc.fixed_dofs
	fixed_nodes = set(fixed_dofs // 3)  # Convert DOFs to node indices
	elements_with_fixed_dofs = []

	for elem in range(fe_solver.mesh.num_elems):
		nodes = fe_solver.mesh.elemArray[elem]
		if any(node in fixed_nodes for node in nodes):
			elements_with_fixed_dofs.append(elem)

	return np.array(elements_with_fixed_dofs)
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


def topopt_mma(fe_solver: sfea.StructFEA,
			   			 maxMMAIterations: int = 250, 
							timeLimit: float =3600, #1 hour
			   			 volfrac: float = 0.5,
						   penal: float = 3.0,
							 move_limit: float = 0.2,
							 kkt_tol: float = 1.e-6,
							 move_tol: float = 0.025,
							 continuationScheme: bool = False,
							 imposeXSymmetry: bool = False,
							 imposeYSymmetry: bool = False,
							 imposeZSymmetry: bool = False,
							 keepFixedElems: bool = False,
							 rel_conv_tol: float = 1.e-4,
							 debug: bool = False,
							 material_model = MaterialModel.SIMP,
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

	tStart = time.time()
	num_elems= fe_solver.mesh.num_elems
	history = {'compliance': [], 'volume': [], 'change': []}
	if material_model == MaterialModel.SIMP:
		material_model_dict = {'name': 'SIMP', 'penal': 3.0} # Default SIMP model
	elif material_model == MaterialModel.RAMP:
		material_model_dict = {'name': 'RAMP', 'penal': 7.0}
	elif material_model == MaterialModel.SIMPPLUS:
		material_model_dict = {'name': 'SIMPPLUS', 'penal': 3.0, 'alpha': 16} 
		#  body-force model from the papers here:
		#  https://doi.org/10.1002/nme.2499, https://doi.org/10.1016/j.cma.2017.04.021 

	# Create  filters
	print("Computing filters...")
	H, Hs = createSmoothingFilter(fe_solver.mesh)
	if imposeXSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
	if imposeYSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
	if imposeZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)
	if imposeZAxisAngularSymmetry >	0:
		HAZ = createAngularSymmetryFilter(fe_solver.mesh, imposeZAxisAngularSymmetry)

	
	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	if (keepFixedElems):
		elemsWithFixedDOF = find_elements_with_fixedDOF(fe_solver)

	mma_params = mma.MMAParams(max_iter=maxMMAIterations,
														kkt_tol = kkt_tol,
														step_tol = move_tol,
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
	if (fe_solver.elem_body_force is not None):
		elem_force = fe_solver.elem_body_force.copy()
		nNodes = fe_solver.mesh.num_nodes
		nodal_body_force = np.zeros((nNodes * 3,))
		nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
		nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
		nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
	else:
		nodal_body_force = None
	if (continuationScheme):
		penal = 1.2
	
	
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
		elif material_model == MaterialModel.RAMP:
			penal = material_model_dict['penal']
			# For RAMP material model: x/(1+penal*(1-x))
			grad_obj = -((penal + 1)/(penal - penal*x + 1)**2)* ce
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

		if imposeXSymmetry:
			grad_obj = (HX * grad_obj)	
		if imposeYSymmetry:
			grad_obj= (HY * grad_obj)
		if imposeZSymmetry:
			grad_obj= (HZ * grad_obj)
		
		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj)

		if (keepFixedElems):
			grad_obj[elemsWithFixedDOF] = min(grad_obj)

		vf = np.mean(x)
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
		print(f"it.: {mma_state.epoch}, obj.: {obj[0]:.6g} vf: {vf:.3f}",
					f"ch: {change:.3f}")
		history['compliance'].append(obj[0])
		history['volume'].append(np.mean(x))
		history['change'].append(change)
		
		if (len(history['compliance'])) >= 2:
			if (change < move_tol):
				break
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if (debug):
				print(f"dJ: {abs(dJ):.7g}, cons: {abs(cons):.7g}")
			if abs(dJ) < rel_conv_tol and abs(cons) < rel_conv_tol:
				break
		if (continuationScheme):
			penal *= 1.1
			penal = min(penal, 3.0)
		if time.time() - tStart > timeLimit:
			break

	fe_solver.mesh.setPseudoDensity(x)
	print(f"Time FEA: {timeFEA:.2f} s, Time MMA: {timeMMA:.2f} s")
	return np.asarray(u), history


def topopt_optimality_criteria(
							fe_solver: sfea.StructFEA,
			  				maxIterations: int = 250,
							volfrac: float = 0.5,
							penal: float = 3,
							move: float = 0.2,
							move_tol: float = 0.025,
							rel_conv_tol: float = 1.e-4,
							directLagrangeMethod: bool = True,
							verbose: bool = True,
							keepFixedElems: bool = False,
							imposeXSymmetry: bool = False,
							imposeYSymmetry: bool = False,
							imposeZSymmetry: bool = False,
							material_model = MaterialModel.SIMP,
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
	# Create  filters
	H, Hs = createSmoothingFilter(fe_solver.mesh)
	if imposeXSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
	if imposeYSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
	if imposeZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)


	elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

	if (keepFixedElems):
		elemsWithFixedDOF = find_elements_with_fixedDOF(fe_solver)
	# Initialize design variables
	x = volfrac * np.ones(num_elems, dtype = float)
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
	obj0,u = _compliance_objective(x, fe_solver)
	
	if material_model == MaterialModel.SIMP:
		material_model_dict = {'name': 'SIMP', 'penal': 3.0} # Default SIMP model
	elif material_model == MaterialModel.RAMP:
		material_model_dict = {'name': 'RAMP', 'penal': 7.0}
	elif material_model == MaterialModel.SIMPPLUS:
		material_model_dict = {'name': 'SIMPPLUS', 'penal': 3.0, 'alpha': 16} 
		#  body-force model from the papers here: https://doi.org/10.1002/nme.2499, https://doi.org/10.1016/j.cma.2017.04.021 

	for iter in range(maxIterations):
		x = np.array(x)
		
		obj,u = _compliance_objective(x, fe_solver,material_model_dict)
	
		ce = (np.dot(u[fe_solver.mesh.edofMat].reshape(num_elems, 24), KE) * u[fe_solver.mesh.edofMat].reshape(num_elems, 24)).sum(1)
		if material_model == MaterialModel.SIMP:
			# For SIMP material model: x**penal
			penal = material_model_dict['penal']
			grad_obj = (-penal * x ** (penal - 1)) * ce
		elif material_model == MaterialModel.RAMP:
			penal = material_model_dict['penal']
			# For RAMP material model: x/(1+penal*(1-x))
			grad_obj = -((penal + 1)/(penal - penal*x + 1)**2)* ce
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
			
		grad_obj /= obj0
		grad_obj = (H * grad_obj)/Hs

		
		if imposeXSymmetry:
			grad_obj = (HX * grad_obj)	
		if imposeYSymmetry:
			grad_obj= (HY * grad_obj)
		if imposeZSymmetry:
			grad_obj= (HZ * grad_obj)
		
		if (elemsWithForces.size > 0):
			grad_obj[elemsWithForces] = min(grad_obj)

		if (keepFixedElems):
			grad_obj[elemsWithFixedDOF] = min(grad_obj)

		cons = _volume_constraint(x, volfrac)
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

				if jnp.sum(xnew) - volfrac * num_elems > 0:
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
			volToDistribute = volfrac*num_elems
			varTimesGrad = x*(-grad_obj)**eta
			while setChange:
				xnew = varTimesGrad/ (np.sum(varTimesGrad[varIn]) /volToDistribute)
				volToDistribute = volfrac*num_elems -np.sum(xMax[xnew>=xMax]) -np.sum(xMin[xnew<=xMin])
				setChange = np.sum(xnew) - volfrac * num_elems > 0
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

		if verbose:
			print(f"it.: {iter+1:d}, obj.: {obj:.5g}, "
				  	f"vol.: {np.mean(xPhys):.3g}, ch.: {change:.3f}")
		
		if (change < move_tol):
			break
		if (len(history['compliance'])) >= 2:
			dJ = (history['compliance'][-1] - history['compliance'][-2]) / history['compliance'][-2]
			if (abs(dJ) < rel_conv_tol and abs(cons) < rel_conv_tol):
				break


	return np.asarray(u), history


def topopt_pareto(fe_solver: sfea.StructFEA,
							desiredVolFrac: float = 0.5,
							rel_err: float = 0.025,
							vol_decr_max: float = 0.05,
							vol_decr_min: float = 0.001,
							min_local_iters: int = 2,
							max_local_iters: int = 10,
							rhoVoid: float = 0,
							removeHangingElems: bool = False,
							keepFixedElems: bool = False,
							imposeXSymmetry: bool = False,
							imposeYSymmetry: bool = False,
							imposeZSymmetry: bool = False,
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

	totalIter = 1

	# Initialize design field
	rho = np.ones((fe_solver.mesh.num_elems))
	volfrac = 1.0
	
	history = {'compliance': [], 'volume': []}
	print("Computing filters...")
	# Create filter
	H, Hs = createSmoothingFilter(fe_solver.mesh)
	if imposeXSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
	if imposeYSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
	if imposeZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)


	if imposeZAxisAngularSymmetry >	0:
		HAZ = createAngularSymmetryFilter(fe_solver.mesh, imposeZAxisAngularSymmetry)

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

	if (keepFixedElems):
		elemsWithFixedDOF = find_elements_with_fixedDOF(fe_solver)
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
	if (keepFixedElems):
		T[elemsWithFixedDOF] = np.max(T)
	T = (H * T) / Hs

	if imposeXSymmetry:
		T = (HX * T)	
	if imposeYSymmetry:
		T = (HY * T)
	if imposeZSymmetry:
		T = (HZ * T)	
	print(f"J={history['compliance'][-1]:.3g}, vf={history['volume'][-1]:.3f},  #FEA={totalIter:2d}")
	vol_decr = vol_decr_max
	wtDamping = 0.5 # 0 means full wt to current T values, else previous T values are damped in

	terminatePareto = False
	while volfrac > desiredVolFrac:
		# Move to next volume fraction
		volfrac = max(desiredVolFrac, volfrac - vol_decr)
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
			if (localIter >= max_local_iters) or abs(JTemp) > 10 * history['compliance'][-1]:  # Divergence check	
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
			if imposeXSymmetry:
				T = (HX * T)	
			if imposeYSymmetry:
				T = (HY * T)
			if imposeZSymmetry:
				T = (HZ * T)	
			if imposeZAxisAngularSymmetry >	0:
				T = (HAZ * T)
			
			if (elemsWithForces.size > 0):
				T[elemsWithForces] = np.max(T)

			if (keepFixedElems):
				T[elemsWithFixedDOF] = np.max(T)
			localIter += 1
			totalIter += 1

		if terminatePareto:
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
			scale = history['compliance'][-1] / history['compliance'][0]
			#vol_decr = max(vol_decr_min,vol_decr/scale**2) # Adjust volume decrease factor for steep increase in compliance
			print(f"J={history['compliance'][-1]:.3g}, vf={history['volume'][-1]:.3f},  #FEA={totalIter:2d}")
			fe_solver.mesh.setPseudoDensity(rho.flatten())
			
	return u, history

if __name__ == "__main__":    
	from examples_structural import *
	import struct_fea as fea
	import linear_solvers as lin_solv
	import time
	import matplotlib.pyplot as plt
	import deflation
	import plots	
	

	jax.config.update("jax_enable_x64", True)
	dsolver = deflation.DeflationSolver()

	# Select the problem and various options
	problem = StructuralExamples.EdgeCantilever
	optimizationMethod = Optimizers.OC # MMA, OC or PARETO
	nDOFDesired = 50000
	desiredVolFraction = 0.1
	material_model = MaterialModel.SIMP# Relevant only for MMA, OC. Use SIMPPLUS for problems with body forces
	solver = lin_solv.Solvers.PARDISO # Typically PARDISO, but DPCG for DOF > 200,000
	removeHangingElems = False # for Pareto, during optimization, remove hanging elements
	keepFixedElems = False # for MMA, OC, Pareto, keep elements with fixed DOF
	
	# Create the problem
	mesh, mat_prop, bc,elem_body_force,symmetry = getStructuralProblem(problem,nDOFDesired = nDOFDesired)

	if (optimizationMethod == Optimizers.MMA) or (optimizationMethod == Optimizers.OC):
		if elem_body_force is not None and (np.linalg.norm(elem_body_force) > 0) and (material_model == MaterialModel.SIMP):
			print("********For body forces, use material model SIMPPLUS********")
			input("Press Enter to continue...")
		if elem_body_force is not None and (np.linalg.norm(elem_body_force) == 0) and (material_model == MaterialModel.SIMPPLUS):
			print("********For no body forces, use material model SIMP ********")
			input("Press Enter to continue...")

	if (optimizationMethod == Optimizers.PARETO):
		if elem_body_force and (np.linalg.norm(elem_body_force) > 0):
			removeHangingElems = True

	imposeXSymmetry = symmetry[0]
	imposeYSymmetry = symmetry[1]
	imposeZSymmetry = symmetry[2]
	imposeZAxisAngularSymmetry = symmetry[3]
	
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
	if optimizationMethod == Optimizers.MMA:
		print("OptimizationMethod: MMA")
		u, history = topopt_mma(fe_solver = fe_solver,
									volfrac = desiredVolFraction,
									material_model = material_model,
									keepFixedElems = keepFixedElems,)
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

	elif optimizationMethod == Optimizers.OC:
		print("OptimizationMethod: OC")
		u, history = topopt_optimality_criteria(fe_solver = fe_solver,
												volfrac = desiredVolFraction,
												material_model = material_model,
												keepFixedElems = keepFixedElems,
												verbose = True)
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
	
	elif optimizationMethod == Optimizers.PARETO:
		print("OptimizationMethod: Pareto")
		u, history = topopt_pareto(fe_solver = fe_solver,
										desiredVolFrac =  desiredVolFraction,imposeXSymmetry=imposeXSymmetry,
										imposeYSymmetry=imposeYSymmetry,imposeZSymmetry=imposeZSymmetry,
										removeHangingElems=removeHangingElems,
										keepFixedElems=keepFixedElems,
										debug = False)
		
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

	print(f"Time taken: {timeTaken:.0f} s")
	
	plots.plotMesh(fe_solver.mesh, bc = None, u=None, title = title)
	plots.plotIsocontour(fe_solver.mesh, title = title,Binarization = True)
	
	

