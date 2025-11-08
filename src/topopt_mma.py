from topopt_common import *
from topopt_obj_cons_sensitivities import *
from topopt_material_model import *
import time
import matplotlib.pyplot as plt
from mmaWrapper import runMMA


def topopt_mma(
    fe_solver,  # hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
    to_params,
    maxMMAIterations: int = 100,
    timeLimitSecs: float = 36000,  # 10 hour
    move_limit: float = 0.2,
    kkt_tol: float = 1.0e-6,
    objective_tol: float = 1.0e-4,
    constraint_tol: float = 1.0e-4,
    print_progress: bool = True,
    plot_progress: bool = False,
    binarize_topology: bool = True,
    progress_callback=None,
    plotter=None,
) -> tuple[np.ndarray, dict]:
    """MMA based topology optimization.

    Returns
    -------
    (sol, history, success, errorMsg, nFEAs)
    """
    def log_message(msg):  # GUI/console logger
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)

    nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
    material_model = MaterialModel.SIMP

    tStart = time.time()
    num_elems = fe_solver.mesh.num_elems

    # History buffers
    history = {"objective": [], "volfrac": []}
    for idx, _constraint in enumerate(to_params.Constraints):
        history[f"constraint_{idx+1}"] = []

    if print_progress:
        log_message("Computing Filters ...")

    # Element-wise density filter matrices
    H, Hs = createFilters(fe_solver, to_params)

    # Elements attached to any externally forced nodes (for retaining)
    elemsWithForces = find_elements_with_forces(
        fe_solver.mesh, fe_solver.bc.force, nDOFPerNode
    )

    # Element stiffness for each physics
    if isinstance(fe_solver.mat_prop, list):  # multiple materials (assume same props)
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_structural(
                    mp.youngs_modulus, mp.poissons_ratio, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
            KE = KE_list[0]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_thermal(
                    mp.thermal_conductivity, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
            KE = KE_list[0]
        log_message("Assuming all elements have the same material properties")
    else:  # single material
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
                fe_solver.mat_prop.youngs_modulus,
                fe_solver.mat_prop.poissons_ratio,
                fe_solver.mesh.elem_size,
            )
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal(
                fe_solver.mat_prop.thermal_conductivity, fe_solver.mesh.elem_size
            )

    # Optional body force mapping (structural)
    if fe_solver.elem_body_force is not None:
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = (
            fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        )
        nodal_body_force[1::3] = (
            fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        )
        nodal_body_force[2::3] = (
            fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
        )
    else:
        nodal_body_force = None

    success = True
    errorMsg = "No errors."
    nFEAs = 0
    obj0 = None
    mmaIterations = 0

    # ----------------- Core chains (x -> obj) and (x -> cons) -----------------

    def objective_function(
        x_raw: np.ndarray,
    ):
        """
        'x_raw -> x_filtered -> FE solve -> objective, grad' chain.

        Returns
        -------
        obj_raw : float
        grad_obj_filt : (n,) ndarray  (gradient w.r.t. FILTERED density)
        sol : ndarray
        x_filtered : (n,) ndarray     (filtered density used in analysis/solve)
        """
        # Density filtering for ANALYSIS
        x_filtered = (H @ x_raw / Hs) if to_params.APPLY_FILTER_TO_DENSITY else x_raw

        # Set filtered density on mesh for FE + plotting
        fe_solver.mesh.setPseudoDensity(x_filtered)

        # Optional plotting (now uses filtered density already set on mesh)
        if plot_progress:
            if progress_callback is not None:
                progress_callback()
            fe_solver.plot_pseudo_density(
                plotter=plotter,
                auto_close=False,
                title=f"Iter {len(history['objective']) + 1} - Density",
            )

        # FE solve & postprocess
        sol = fe_solver.solve(x_filtered, material_model)
        fe_solver.postprocess()

        # Base objective and gradient (w.r.t. FILTERED density)
        obj_raw, grad_obj_filt = compute_objective_and_gradient(
            to_params, sol, x_filtered, fe_solver, KE, material_model
        )
        return obj_raw, grad_obj_filt, sol, x_filtered

    def constraint_function(
        x_raw: np.ndarray,
    ):
        """
        'x_raw -> x_filtered -> constraints, grad' chain.

        If `sol` is provided (from the same x_filtered), we reuse it.

        Returns
        -------
        c : (m,) ndarray
        dcdx_filt : (m, n) ndarray  (grad w.r.t. FILTERED density)
        x_filtered : (n,) ndarray   (filtered density used for constraints)
        """
        x_filtered = (H @ x_raw / Hs) if to_params.APPLY_FILTER_TO_DENSITY else x_raw
        # Set filtered density on mesh only if we are going to solve here
        fe_solver.mesh.setPseudoDensity(x_filtered)
        sol = fe_solver.solve(x_filtered, material_model)
        fe_solver.postprocess()

        c, dcdx_filt = compute_constraint_and_gradient(
            to_params, sol, x_filtered, fe_solver, KE, material_model
        )
        return c, dcdx_filt, x_filtered

    # ------------------------ MMA objective callback -------------------------

    def optimizationFunction(x):
        nonlocal nFEAs, obj0, mmaIterations

        x = np.asarray(x).flatten()

        # Objective chain (filters are inside)
        obj_raw, grad_obj_filt, sol, x_filtered = objective_function(x)

        # Normalization
        if obj0 is None:
            obj0 = obj_raw
        obj = obj_raw / obj0
        grad_obj = grad_obj_filt / obj0  # still w.r.t. FILTERED density

        # Additional body-force term (structural)
        if nodal_body_force is not None:
            ce_body_force = (
                sol[fe_solver.mesh.edofMat].reshape(num_elems, 24)
                * nodal_body_force[fe_solver.mesh.edofMat].reshape(num_elems, 24)
            ).sum(1)
            grad_obj += 2.0 * ce_body_force * get_material_model_rho_sensitivity(
                x_filtered, material_model
            )

        # Sensitivity filtering/mapping (FILTERED -> RAW design variables)
        if to_params.APPLY_FILTER_TO_SENSITIVITY and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            # Weighted filter
            grad_obj = (H @ (x_filtered * grad_obj)) / Hs / x_filtered
        elif to_params.APPLY_FILTER_TO_SENSITIVITY and (
            to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION
        ):
            # Regular filter
            grad_obj = (H @ grad_obj) / Hs

        # Retain elements with forces / keep-list
        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)
        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)

        # Constraints chain (reuse sol and x_filtered)
        c, dcdx_filt, _ = constraint_function(x)

        # Sensitivity filtering for constraints (FILTERED -> RAW)
        if to_params.APPLY_FILTER_TO_SENSITIVITY:
            for m in range(len(to_params.Constraints)):
                if to_params.Constraints[m][0] is TO_QOI.COMPLIANCE:
                    dcdx_filt[m] = (H @ (x_filtered * dcdx_filt[m])) / Hs / x_filtered
                elif to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION:
                    dcdx_filt[m] = (H @ dcdx_filt[m]) / Hs
        dcdx = dcdx_filt

        # Book-keeping
        history["objective"].append(obj * obj0)        # store de-normalized (as before)
        history["volfrac"].append(np.mean(x_filtered)) # track filtered volume fraction
        for idx, val in enumerate(c.flatten()):
            history[f"constraint_{idx+1}"].append(val)

        # Initial-iteration warning on constraints
        if (mmaIterations == 0) and print_progress:
            if np.any(c > 0):
                print("Warning: Constraint(s) violated at start of optimization!")
                print("GCMMA may not converge for this problem.")
                print("Consider changing constraints if convergence issues occur.")

        # Shape for MMA
        grad_obj = grad_obj.reshape(-1, 1)

        # Pretty printing
        if print_progress:
            print(50 * "-")
            print(f"Iteration: {mmaIterations}")
            objective_name = getattr(
                to_params.Objective[0], "name", str(to_params.Objective[0])
            )
            print(f"Min. Objective ({objective_name}): {obj * obj0:.3g}")
            constraint_names = [
                getattr(cn[0], "name", str(cn[0])) for cn in to_params.Constraints
            ]
            for idx, val in enumerate(c.flatten()):
                inequality = ">=" if constraint_names[idx] == "STRESS_SAFETY_FACTOR" else "<="
                rhs = to_params.Constraints[idx][2]
                print(
                    f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*rhs:.3g} {inequality} {rhs:.3g}?"
                )

        mmaIterations += 1
        nFEAs += 1

        # Heuristic: enforce constraints more strongly for MMA if requested
        if to_params.Enforce_Constraints_MMA:
            for m in range(len(to_params.Constraints)):
                scaling = max(0.1, min(10, 10 ** c[m]))  # heuristic
                dcdx[m, :] /= scaling

        return obj, grad_obj, c, dcdx

    # ------------------------------ MMA driver -------------------------------

    initialDensity = 0.5
    x0 = initialDensity * np.ones(num_elems, dtype=float).reshape(-1, 1)
    lowerBound = np.zeros(num_elems, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_elems, dtype=float).reshape(-1, 1)
    nVariables = num_elems
    nConstraints = len(to_params.Constraints)

    (
        xOptimal,
        f0val,
        df0dx,
        gval,
        dgdx,
        nFEAs,
    ) = runMMA(
        nVariables,
        nConstraints,
        optimizationFunction,
        x0,
        lowerBound,
        upperBound,
        maxIterations=maxMMAIterations,
        timeLimitSecs=timeLimitSecs,
        move_limit=move_limit,
        fTolerance=objective_tol,
        gTolerance=constraint_tol,
        kktTol=kkt_tol,
        verbose=False,
        progress_callback=progress_callback,
    )

    # ---------------------------- Finalization -------------------------------

    x = np.asarray(xOptimal).flatten()

    # Evaluate final objective/constraints via the same chains (ensures consistency)
    obj_raw_final, _gobj_filt, sol, x_filtered_final = objective_function(x)
    c_final, _dg_filt, _ = constraint_function(x)

    # Track in history (store de-normalized to match earlier semantics)
    history["objective"].append(obj_raw_final)
    history["volfrac"].append(np.mean(x_filtered_final))
    for idx, val in enumerate(c_final.flatten()):
        history[f"constraint_{idx+1}"].append(val)

    # Grey fraction before binarization (based on RAW x for reporting)
    grey_elements = np.sum((x > 0.1) & (x < 0.9))
    fraction_grey = grey_elements / num_elems

    # Optional binarization for display/export
    if binarize_topology:
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1 - np.mean(x)) * len(x))]
        x = np.where(x < threshold, 0.0, 1.0)

    # For topological operations (hanging elements), operate on RAW (binarized) x
    fe_solver.mesh.setPseudoDensity(x)

    if to_params.Eliminate_Hanging_Elements:
        # Ensure binarized
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1 - np.mean(x)) * len(x))]
        x = np.where(x < threshold, 0.0, 1.0)
        fe_solver.mesh.setPseudoDensity(x)

        meshComponents = fe_solver.mesh.find_connected_components()
        if len(meshComponents) > 1:
            if print_progress:
                log_message(50 * "-")
                log_message("Removing hanging elements.")
            largest_component = max(meshComponents, key=len)
            x[:] = 0.0
            x[list(largest_component)] = 1.0
            fe_solver.mesh.setPseudoDensity(x.flatten())

    # Final FE evaluation for reporting (use analysis chain with filtering)
    obj_raw_final, _gobj_filt, sol, x_filtered_final = objective_function(x)
    c_final, _dg_filt, _ = constraint_function(x, sol, x_filtered_cached=x_filtered_final)

    # Log final line
    grey_elements = np.sum((x > 0.1) & (x < 0.9))
    fraction_grey = grey_elements / num_elems
    print("-" * 50)
    log_message(
        f"Final objective: {obj_raw_final:.4g}, vf: {np.mean(x_filtered_final):.3f}, grey: {fraction_grey:.3f}"
    )
    log_message(f"Total Time: {time.time() - tStart:.2f} s")
    log_message(f"Error: {errorMsg}")

    return np.asarray(sol), history, success, errorMsg, nFEAs


# ------------------------------ Script runner -------------------------------

if __name__ == "__main__":
    from topopt_structural_benchmarks import *
    from topopt_thermal_benchmarks import *

    print("-" * 50)

    to_problem = StructuralTOExamples.CantileverTipLoad  # Choose the TO problem

    if to_problem in StructuralTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(
            to_problem
        )
    elif to_problem in ThermalTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(
            to_problem
        )

    print(f"Running {to_problem.name}...")
    print("-" * 50)

    solver = lin_solv.Solvers.SPSOLVE  # default, see below
    dsolver = deflation.DeflationSolver(use_gpu=False)

    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF:
        # Typically PARDISO, but DPCG for large DOF problems
        solver = lin_solv.Solvers.DPCG
        nGroups = min(
            dsolver.maxGroups,
            max(
                dsolver.minGroups,
                round(3 * mesh.num_nodes / dsolver.dofPerGroup),
            ),
        )
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]

    if to_problem in StructuralTOExamples:
        fe_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force,
        )
    elif to_problem in ThermalTOExamples:
        fe_solver = hex_thermal_fea.HexThermalFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force,
        )

    print("Solver: ", fe_solver.solver.name)
    print("nDOF: ", 3 * fe_solver.mesh.num_nodes)
    print("nNodes: ", fe_solver.mesh.num_nodes)
    print("nElem: ", fe_solver.mesh.num_elems)

    title = f"nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}"
    # fe_solver.plot_mesh(title = title, save_path = None)

    startTime = time.time()
    print("OptimizationMethod: MMA")

    u, history, success, errorMsg, nFEAs = topopt_mma(
        fe_solver=fe_solver,
        to_params=to_params,
        maxMMAIterations=to_params.MaxIterations,
    )
    timeTaken = time.time() - startTime

    title = (
        f"MMA: vol: {history['volfrac'][-1]:0.2f}, "
        f"J: {history['objective'][-1]:.3g}, "
        f"nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    )
    fe_solver.plot_mesh(title=title, plot_bc=False, save_path=None)
    fe_solver.postprocess()
    fe_solver.plot_vonMisesStress()

    plt.figure()
    plt.plot(history["objective"], label="Objective")
    plt.xlabel("Iterations")
    plt.ylabel(f"Objective ({to_params.Objective[0].name})")
    plt.grid(True)
    plt.show()

    for idx, constraint_name in enumerate(
        [getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]
    ):
        plt.figure()
        val = np.array(history[f"constraint_{idx+1}"])
        constraint_val = (val + 1) * to_params.Constraints[idx][2]
        plt.plot(constraint_val, label=f"Constraint {idx+1} ({constraint_name})")
        plt.xlabel("Iterations")
        plt.ylabel(f"Constraint {idx+1} ({constraint_name})")
        plt.axhline(y=to_params.Constraints[idx][2], color="r", linestyle="--")
        plt.grid(True)
        plt.show()
