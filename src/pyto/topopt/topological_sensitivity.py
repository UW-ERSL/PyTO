"""Topological sensitivity formulas -- not test-oracle-only code.

Unlike the functions in pyto.autodiff.reference_adjoint, these are not a
NumPy analytic twin superseded by a torch-native one -- topological
sensitivity (the change in an objective from nucleating an infinitesimal
hole at a point, an asymptotic small-hole expansion) is a genuinely
different mathematical object from d(objective)/d(density), not just a
different implementation of the same gradient. It isn't something
torch.autograd.grad computes, so there's no torch-native replacement for
it to be superseded by.

(These lived inside pyto.autodiff.reference_adjoint through Phase 4 of
the restructuring plan, incorrectly implying they were test-oracle-only
like their neighbors there -- moved here in Phase 5 once that turned out
to be wrong.)

CAUTION -- known duplication (flagged, not fixed; see
pyto.topopt.drivers.pareto's module docstring for the full explanation):
pyto.topopt.drivers.pareto does NOT actually call these -- it carries its
own second, independent copy of the same four formulas, with a different
physics-dispatch convention. This module's compute_objective_topological_sensitivity_compliance
takes no feaMode and reads mesh.edofMat (needs the manual edofMat patch,
inferring structural-vs-thermal from KE.shape[0]); pareto.py's local copy
takes an explicit feaMode and reads mesh.edofMatStructural/edofMatThermal
directly (no patch needed). Currently nothing in pyto.topopt imports from
this module at all; it exists as the formulas' canonical home and as a
placeholder for the day the two copies get consolidated.
"""
import numpy as np

import pyto.solve.numpy_backend as linear_solvers
from pyto.autodiff.material_model import (
    get_structural_material_model_scaling,
    get_thermal_material_model_scaling,
)


def compute_objective_topological_sensitivity_compliance(to_params, sol: np.ndarray, x: np.ndarray,	fe_solver, KE,
				material_model = None):
	from pyto.topopt.common import TO_QOI

	# Compute the compliance independent of objective
	dofMat = fe_solver.mesh.edofMat
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
