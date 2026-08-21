from pyto.topopt.common import *
from pyto.autodiff.qoi import *
from pyto.autodiff.material_model import *
from pyto.autodiff.sparse_solve import SparseLinearSolve, Solvers
import time
import matplotlib.pyplot as plt
from pyto.topopt.drivers.mmaWrapper import runMMA
from pyto.topopt.drivers._shared import setup_driver_state, torch_sparse_filter
from pyto.examples_benchmarks.topopt_structural_benchmarks import *
from pyto.examples_benchmarks.topopt_thermal_benchmarks import *


def run_topopt_mma(to_problem):
    """Run topology optimization using the MMA method.

    Mirrors run_topopt_ocm/run_topopt_levelset/run_pareto_topopt's pattern
    (see topopt/drivers/oc.py, levelset.py, pareto.py) -- builds the mesh
    and FE solver for the given example, then calls topopt_mma() and plots
    the result. This wrapper didn't previously exist (only topopt_mma()
    itself did), which meant `from pyto.topopt.drivers.mma import
    run_topopt_mma` -- e.g. main.ipynb's import cell -- raised an
    ImportError, unlike the equivalent OC/Levelset/Pareto imports.

    Args:
        to_problem: A StructuralTOExamples or ThermalTOExamples member
            identifying which example problem to set up.

    Returns: None
    """
    if to_problem in StructuralTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(to_problem)
    elif to_problem in ThermalTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(to_problem)

    print(f"Running {to_problem.name}...")
    print("-" * 50)
    solver = Solvers.PARDISO  # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
    dsolver = deflation.DeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF:
        solver = Solvers.DPCG
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)

    if to_problem in StructuralTOExamples:
        fe_solver = hex_structural_fea.HexStructuralFEA(mesh=mesh, mat_prop=mat_prop, bc=bc,
                    solver=solver, dsolver=dsolver, rtol=1e-8, elem_body_force=elem_body_force)
    elif to_problem in ThermalTOExamples:
        fe_solver = hex_thermal_fea.HexThermalFEA(mesh=mesh, mat_prop=mat_prop, bc=bc,
                    solver=solver, dsolver=dsolver, rtol=1e-8, elem_body_force=elem_body_force)

    print('Solver: ', fe_solver.solver.name)
    print("nDOF: ", 3 * fe_solver.mesh.num_nodes)
    print("nElem: ", fe_solver.mesh.num_elems)

    startTime = time.time()
    print("OptimizationMethod: MMA")
    sol, history, success, errorMsg, nFEAs = topopt_mma(fe_solver=fe_solver, to_params=to_params,
                    plot_progress=True, print_progress=True, maxMMAIterations=to_params.MaxIterations)
    timeTaken = time.time() - startTime
    print(f"Time taken: {timeTaken:.0f} s")
    if not success:
        print(f"Error: {errorMsg}")

    title = f"MMA: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    fe_solver.plot_mesh(title=title, plot_bc=False, save_path=None)

    plt.figure()
    plt.plot(history["objective"], label="Objective")
    plt.xlabel("Iterations")
    plt.ylabel(f"Objective ({to_params.Objective[0].name})")
    plt.grid(True)
    plt.show()


def topopt_mma(fe_solver, #hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
                           to_params,
                            maxMMAIterations: int = 150, 
                            timeLimitSecs: float = 36000, #10 hour
                            move_limit: float = 0.2,
                            kkt_tol: float = 1.e-6,
                            objective_tol: float = 1.e-4,
                            constraint_tol: float = 1.e-4,
                            print_progress: bool = True,
                            plot_progress: bool = False,
                            binarize_topology: bool = True,   
                            progress_callback=None, 
                            plotter=None  
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

    state = setup_driver_state(fe_solver, to_params, log_message if print_progress else None)
    H, Hs = state["H"], state["Hs"]
    elemsWithForces = state["elemsWithForces"]
    KE = torch.tensor(state["KE"])
    nodal_body_force = state["nodal_body_force"]

    # Torch-sparse copies of the (constant) filter, used inside the autograd
    # chain below -- H is a scipy sparse matrix, which can't be matmul'd
    # against a grad-tracked torch tensor directly.
    H_torch, Hs_torch = torch_sparse_filter(H, Hs)

    success = True
    errorMsg = "No errors."
    nFEAs = 0
    obj0 = None
    mmaIterations = 0

    # ----------------- Core chains (x -> obj) and (x -> cons) -----------------

    def obj_cons_function(
        x_raw: torch.Tensor,
    ):
        """
        'x_raw -> x_filtered -> FE solve -> objective, constraints, grads' chain.

        Returns
        -------
        obj_raw       : float
        grad_obj_filt : (n,) ndarray       (gradient w.r.t. FILTERED density)
        c             : (m,1) ndarray      (constraints)
        dcdx_filt     : (m,n) ndarray      (grad w.r.t. FILTERED density)
        sol           : ndarray            (FE solution)
        x_filtered    : (n,) ndarray       (filtered density used in analysis/solve)
        """

        x_raw = x_raw.view(-1)  # ensure 1D

        def chain(x_raw: torch.Tensor):
            # Density filtering for ANALYSIS
            x_filtered = (
                torch.sparse.mm(H_torch, x_raw.unsqueeze(1)).squeeze(1) / Hs_torch
            ) if to_params.APPLY_FILTER_TO_DENSITY else x_raw

            # Set filtered density on mesh for FE + plotting
            with torch.no_grad():
                fe_solver.mesh.setPseudoDensity(x_filtered.detach())

            # FE solve
            sol = fe_solver.solve(x_filtered, material_model)

            # Objective (Torch scalar)
            obj_raw = compute_objective_and_gradient(
                to_params, sol, x_filtered, fe_solver, KE, material_model
            )

            # Constraints (Torch vector)
            c_t, _ = compute_constraint_and_gradient(
                to_params, sol, x_filtered, fe_solver, KE, material_model
            )
            c_t = c_t.view(-1)  # (m,)

            return obj_raw, c_t, sol, x_filtered

        # Forward pass
        obj_raw_t, c_t, sol_t, x_filtered_t = chain(x_raw)

        # Gradient of objective wrt x_raw (like jacrev on a scalar)
        (grad_obj_t,) = torch.autograd.grad(
            outputs=obj_raw_t,
            inputs=x_raw,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )  # (n,)

        # Jacobian of constraints wrt x_raw, row-by-row (like jacrev on a vector)
        grads_c = []
        m = c_t.numel()
        for i in range(m):
            (grad_ci,) = torch.autograd.grad(
                outputs=c_t[i],
                inputs=x_raw,
                create_graph=False,
                retain_graph=(i < m - 1),
                allow_unused=False,
            )
            grads_c.append(grad_ci)  # each (n,)
        dcdx_t = torch.stack(grads_c, dim=0)
        # Convert to numpy with MMA-friendly shapes
        obj_raw = obj_raw_t.detach().cpu().numpy()
        grad_obj_filt = grad_obj_t.detach().cpu().numpy().reshape(-1)       # (n,)
        c_np = c_t.detach().cpu().numpy().reshape(-1, 1)                    # (m,1)
        dcdx_filt = dcdx_t.detach().cpu().numpy()                           # (m,n)
        sol_np = sol_t.detach().cpu().numpy()
        x_filtered_np = x_filtered_t.detach().cpu().numpy()

        return obj_raw, grad_obj_filt, c_np, dcdx_filt, sol_np, x_filtered_np


    # ------------------------ MMA objective callback -------------------------

    def optimizationFunction(x):
        nonlocal nFEAs, obj0, mmaIterations

        x = np.asarray(x).flatten()
        x_t = torch.tensor(x, requires_grad=True)

        # Objective chain (filters are inside)
        obj_raw, grad_obj_filt, c, dcdx_filt, sol, x_filtered = obj_cons_function(x_t)

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
        if to_params.APPLY_FILTER_TO_DENSITY and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            # Weighted filter
            grad_obj = (H @ (x_filtered * grad_obj)) / Hs / x_filtered
        elif to_params.APPLY_FILTER_TO_DENSITY and (
            to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION
        ):
            # Regular filter
            grad_obj = (H @ grad_obj) / Hs

        # Retain elements with forces / keep-list
        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)
        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)


        # Sensitivity filtering for constraints (FILTERED -> RAW)
        if to_params.APPLY_FILTER_TO_DENSITY:
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

        # Extract names for printing
        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        # Print objective and constraints for this iteration
        if (print_progress):
            print(50* '-')
            print(f"Iteration: {mmaIterations}")
            objective_name = getattr(
                to_params.Objective[0], "name", str(to_params.Objective[0])
            )
            print(f"Min. Objective ({objective_name}): {obj * obj0:.3g}")
            constraint_names = [
                getattr(cn[0], "name", str(cn[0])) for cn in to_params.Constraints
            ]
            for idx, val in enumerate(c.flatten()):
                # Every branch in compute_constraint_and_gradient computes
                # c <= 0 (including STRESS_FAILURE_FACTOR, fixed to follow
                # the same value/limit - 1.0 convention as every other
                # constraint), so the displayed inequality is always "<=".
                inequality = "<="
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


    objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
    constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]
    
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
    x_t = torch.tensor(x, requires_grad=True)

    # Evaluate final objective/constraints via the same chains (ensures consistency)
    obj_raw_final, _gobj_filt, c_final, _dg_filt, sol, x_filtered_final = obj_cons_function(x_t)

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
    fe_solver.mesh.setPseudoDensity(torch.tensor(x))

    if to_params.Eliminate_Hanging_Elements:
        # Ensure binarized
        x_sorted = np.sort(x)
        threshold = x_sorted[int((1 - np.mean(x)) * len(x))]
        x = np.where(x < threshold, 0.0, 1.0)
        fe_solver.mesh.setPseudoDensity(torch.tensor(x))

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
    obj_raw_final, _gobj_filt, c_final, _dg_filt, sol, x_filtered_final = obj_cons_function(x_t)

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
    from pyto.examples_benchmarks.topopt_structural_benchmarks import *
    from pyto.examples_benchmarks.topopt_thermal_benchmarks import *

    print("-" * 50)

    to_problem = StructuralTOExamples.LBracketTopLoadStressObjective # Choose the TO problem
    
    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

    print(f"Running {to_problem.name}...")
    print("-" * 50)

    solver = Solvers.SPSOLVE  # default, see below
    dsolver = deflation.DeflationSolver(use_gpu=False)

    # if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF:
    #     # Typically PARDISO, but DPCG for large DOF problems
    #     solver = lin_solv.Solvers.DPCG
    #     nGroups = min(
    #         dsolver.maxGroups,
    #         max(
    #             dsolver.minGroups,
    #             round(3 * mesh.num_nodes / dsolver.dofPerGroup),
    #         ),
    #     )
    #     dsolver.create_deflation_groups(mesh, nGroups)
    #     dsolver.create_deflation_matrix(mesh)
    #     dsolver.W = dsolver.W[bc.free_dofs, :]

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
    
    title = f'nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
    #fe_solver.plot_mesh(title = title, save_path = None)
    
    plot_progress = True
    print_progress = True
    startTime = time.time()
    print("OptimizationMethod: MMA")
    
    u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,
                                to_params = to_params,
                                plot_progress= plot_progress,
                                print_progress= print_progress,
                                maxMMAIterations= to_params.MaxIterations,)
    timeTaken = time.time() - startTime
    fe_solver.postprocess()
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
