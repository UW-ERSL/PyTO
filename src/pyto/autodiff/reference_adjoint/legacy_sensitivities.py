"""Legacy, hand-derived NumPy discrete-adjoint sensitivity formulas.

TEST-ORACLE ONLY. Nothing in pyto.topopt (the optimizer drivers) imports
from this module -- it exists so tests/ can verify the torch-native QOI
functions in pyto.autodiff.qoi agree with an independently-derived
analytic formula. See plan Phase 4: before this split, these functions
and their torch-native twins lived in the same file
(topopt_obj_cons_sensitivities.py) and were called inconsistently by
different drivers, which is what caused most of the bugs found in the
initial analysis of this codebase.

Formulas are unchanged from before the split (only import locations
changed) -- they are the ground truth these files exist to preserve.
"""
import numpy as np

from pyto.autodiff.material_model import (
    _PNORM_EXPONENT,
    get_structural_material_model_scaling,
    get_thermal_material_model_scaling,
    get_structural_material_model_sensitivity,
    get_thermal_material_model_sensitivity,
)
import linear_solvers


def compute_volume_constraint_and_gradient(x: np.ndarray,
											 volfracUpper: float,
											 )-> np.ndarray:
	"""Compute the volume constraint.

	Args:
		density: Array of (num_elems,) containing the element densities.
		volfrac: The target volume fraction.

	Returns: The volume constraint. The constraint is satisfied when the
		returned value is zero. The constraint is inactive when the returned
		value is negative.
	"""

	volConstraint = ((np.mean(x)/volfracUpper) - 1.0)
	volConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
	return volConstraint, volConstraint_gradient


def compute_compliance(sol: np.ndarray, x: np.ndarray,
				fe_solver, KE,
				material_model = None) -> np.ndarray:
	"""Compute the  compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
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

	return compliance


def compute_compliance_and_gradient(sol: np.ndarray, x: np.ndarray,
				fe_solver, KE,
				material_model = None) -> np.ndarray:
	"""Compute the  compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
		compliance_grad = -get_structural_material_model_sensitivity(x,material_model) * ce

	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)
		compliance_grad = -get_thermal_material_model_sensitivity(x,material_model) * ce
	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")

	compliance = np.sum(materialScaling * ce)
	return compliance, compliance_grad


def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x, fe_solver, KE, material_model):
    """
    Compute von Mises stress and sensitivity with respect to x for p-norm stress.

	# "An efficient 146-line 3D sensitivity analysis code of
 	# stress based topology optimization written in MATLAB"
 	# Optimization and Engineering (2022) 23:1733–1757
    """
    mesh = fe_solver.mesh
    nelems = mesh.num_elems

    qStress = 0.5  # STRESS relaxation factor
    pSIMP = 3    # SIMP penalization
    p = _PNORM_EXPONENT  # p-norm exponent

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

    # B matrix setup
    gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])

    B = np.zeros((6, 24))
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

    idx = np.arange(8)
    B[:, (3 * idx)[:, None] + np.arange(3)] = Bi.transpose(0, 2, 1)

    F = D @ B  # shape (6, 24)

    vm_elems = fe_solver.vonMisesStress
    vm_pnorm = fe_solver.pNormStress

    # Compute dpn_dvms = (sum(vm^p))^(1/p - 1)
    dpn_dvms = (np.sum(vm_elems ** p)) ** (1/p - 1)

    # Pre-compute DvmDs for all elements
    DvmDs_all = np.zeros((nelems, 6))
    for e in range(nelems):
        stress_elem = fe_solver.stressComponents[e]
        sigma11, sigma22, sigma33, sigma12, sigma13, sigma23 = stress_elem

        # DvmDs - derivative of von Mises w.r.t. stress components
        DvmDs_all[e, 0] = 1/(2*vm_elems[e]) * (2*sigma11 - sigma22 - sigma33)
        DvmDs_all[e, 1] = 1/(2*vm_elems[e]) * (2*sigma22 - sigma11 - sigma33)
        DvmDs_all[e, 2] = 1/(2*vm_elems[e]) * (2*sigma33 - sigma11 - sigma22)
        DvmDs_all[e, 3] = 3/vm_elems[e] * sigma12
        DvmDs_all[e, 4] = 3/vm_elems[e] * sigma13
        DvmDs_all[e, 5] = 3/vm_elems[e] * sigma23

    # Compute T1 (direct sensitivity)
    beta = np.zeros(nelems)
    x = np.maximum(x, 1e-12) # avoid division by zero
    for e in range(nelems):
        edof = mesh.edofMat[e]
        u_e = sol[edof]
        beta[e] = qStress * (x[e]**(qStress-1)) * (vm_elems[e]**(p-1)) * DvmDs_all[e] @ D @ B @ u_e

    T1 = dpn_dvms * beta

    # Compute adjoint right-hand side using pre-computed DvmDs
    g = np.zeros(fe_solver.bc.num_dofs)
    for e in range(nelems):
        edof = mesh.edofMat[e]
        g_e = (x[e]**qStress) * dpn_dvms * B.T @ D.T @ DvmDs_all[e] * (vm_elems[e]**(p-1))
        g[edof] += g_e

    # Solve adjoint equation
    adjointSol = linear_solvers.solve(fe_solver.stiff_mtrx,
                                       g,
                                       fe_solver.solver,
                                       fe_solver.bc,
                                       dsolver=fe_solver.dsolver,
                                       **fe_solver.kwargs)

    # Compute T2 (indirect sensitivity via adjoint)
    dofMat = fe_solver.mesh.edofMat
    nRows = KE.shape[0]
    ce = (np.dot(adjointSol[dofMat].reshape(nelems, nRows), KE) *
          sol[dofMat].reshape(nelems, nRows)).sum(1)

    T2 = -pSIMP * (x**(pSIMP-1)) * ce  # Note the negative sign from MATLAB

    vm_pnorm_sensitivity = T1 + T2
    max_vm = np.max(vm_elems)

    return vm_pnorm, vm_pnorm_sensitivity, max_vm


def compute_solution_dotproduct_and_gradient(sol: np.ndarray, x,fe_solver,KE,material_model,g: np.ndarray,
				) -> np.ndarray:
	"""Compute the objective g'* sol, and its gradient.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	obj = np.dot(sol, g)


	adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)

	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		compliance_grad = get_structural_material_model_sensitivity(x,material_model) * ce

	elif (nRows == 8): # thermal hex
		compliance_grad = get_thermal_material_model_sensitivity(x,material_model) * ce
	return obj, compliance_grad


# NOTE: the topological-sensitivity functions that used to live here
# (compute_objective_topological_sensitivity_compliance,
# computeStructuralTopologicalSensitivity, compute_pnorm_stress_and_TS,
# computeThermalTopologicalSensitivity) moved to
# pyto.topopt.topological_sensitivity in Phase 5 -- they are the Pareto
# driver's actual production sensitivity computation, not a NumPy twin of
# a torch-native replacement, so grouping them with the test-oracle-only
# functions above was misleading. See that module's docstring for why.
