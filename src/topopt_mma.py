# topopt_mma_main.py
from topopt_common import *
from topopt_obj_cons_sensitivities import *
from topopt_material_model import *
import time
import matplotlib.pyplot as plt
from mmaWrapper import runMMA
from topopt_thermostructural_sensitivity import ThermoElasticSensitivity
from topopt_structural_benchmarks import *
from topopt_thermal_benchmarks import *
from topopt_thermostructural_benchmarks import *


def run_topopt_mma(to_problem):

    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(to_problem)
        feaMode = FEA_MODE.STRUCTURAL
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(to_problem)
        feaMode = FEA_MODE.THERMAL
    elif (to_problem in ThermoStructuralTOExamples):
        mesh, mat_prop, structural_bc, thermal_bc, elem_body_force, to_params = getThermoStructuralTOProblem(to_problem)
        feaMode = FEA_MODE.THERMO_STRUCTURAL
    else:
        raise ValueError("Unknown TO problem type.")

    print(f"Running {to_problem.name}...")
    print("-" * 50)

    solver = lin_solv.Solvers.PARDISO
    dsolver = deflation.DeflationSolver(use_gpu=False)
    if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):
        solver = lin_solv.Solvers.DPCG
        nGroups = dsolver.getRecommendedNumberOfGroups(nDOF=3 * mesh.num_nodes)
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)

    print("Solver:", solver.name)

    fe_structural_solver = None
    fe_thermal_solver = None

    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_structural_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force
        )
        nNodes = fe_structural_solver.mesh.num_nodes
        nElems = fe_structural_solver.mesh.num_elems
        nDOF = 3 * nNodes

    elif (feaMode == FEA_MODE.THERMAL):
        fe_thermal_solver = hex_thermal_fea.HexThermalFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force
        )
        nNodes = fe_thermal_solver.mesh.num_nodes
        nElems = fe_thermal_solver.mesh.num_elems
        nDOF = nNodes
        fe_thermal_solver.plot_mesh(title="Thermal Load", plot_bc=True, save_path=None)

    elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
        fe_structural_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=structural_bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force
        )
        fe_thermal_solver = hex_thermal_fea.HexThermalFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=thermal_bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8
        )
        nNodes = fe_structural_solver.mesh.num_nodes
        nElems = fe_structural_solver.mesh.num_elems
        nDOF = 3 * nNodes

    print("nNodes:", nNodes)
    print("nElem:", nElems)
    print("nDOF:", nDOF)

    plot_progress = True
    print_progress = True
    startTime = time.time()
    print("OptimizationMethod: MMA")

    u, history, success, errorMsg, nFEAs = topopt_mma(
        feaMode,
        fe_structural_solver,
        fe_thermal_solver,
        to_params=to_params,
        plot_progress=plot_progress,
        print_progress=print_progress,
        maxiteration=to_params.MaxIterations,
    )

    timeTaken = time.time() - startTime
    title = f"MMA: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"

    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_structural_solver.postprocess()
        fe_structural_solver.plot_vonMisesStress()
    elif (feaMode == FEA_MODE.THERMAL):
        fe_thermal_solver.postprocess()
        fe_thermal_solver.plot_temperature()

    plt.close("all")
    plt.figure()
    plt.plot(history["objective"], label="Objective")
    plt.xlabel("Iterations")
    plt.ylabel(f"Objective ({to_params.Objective[0].name})")
    plt.grid(True)
    plt.show()

    for idx, constraint_name in enumerate([getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]):
        plt.figure()
        val = np.array(history[f"constraint_{idx+1}"])
        constraint_val = (val + 1) * to_params.Constraints[idx][2]
        plt.plot(constraint_val, label=f"Constraint {idx+1} ({constraint_name})")
        plt.xlabel("Iterations")
        plt.ylabel(f"Constraint {idx+1} ({constraint_name})")
        plt.axhline(y=to_params.Constraints[idx][2], color="r", linestyle="--")
        plt.grid(True)
        plt.show()

    plt.close("all")


def topopt_mma(
    feaMode: FEA_MODE,
    fe_structural_solver,
    fe_thermal_solver,
    to_params,
    maxiteration: int = 300,
    timeLimitSecs: float = 36000,
    move_limit: float = 0.05,
    kkt_tol: float = 1.0e-6,
    objective_tol: float = 1.0e-3,
    constraint_tol: float = 1.0e-3,
    print_progress: bool = True,
    plot_progress: bool = False,
    use_continuation: bool = False,
    binarize_topology: bool = True,
    progress_callback=None,
    plotter=None,
) -> tuple[np.ndarray, dict]:


    def log_message(msg):
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)

    # Pick primary solver
    if (feaMode == FEA_MODE.STRUCTURAL):
        fe_solver = fe_structural_solver
        mesh = fe_structural_solver.mesh
        mat_prop = fe_structural_solver.mat_prop
        nDOFPerNode = 3
    elif (feaMode == FEA_MODE.THERMAL):
        fe_solver = fe_thermal_solver
        mesh = fe_thermal_solver.mesh
        mat_prop = fe_thermal_solver.mat_prop
        nDOFPerNode = 1
    elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
        fe_solver = fe_structural_solver
        if (fe_structural_solver.mesh.num_elems != fe_thermal_solver.mesh.num_elems):
            raise ValueError("Structural and thermal meshes must have the same number of elements.")
        mesh = fe_structural_solver.mesh
        mat_prop = fe_structural_solver.mat_prop
        nDOFPerNode = 3
        if (not use_continuation):
            print("Continuation is recommended for thermo-structural problems.")
    else:
        raise ValueError("Invalid FEA mode.")

    material_model = to_params.materialModel

    tStart = time.time()
    num_elems = mesh.num_elems



    history = {"objective": []}
    for idx, _constraint in enumerate(to_params.Constraints):
        history[f"constraint_{idx+1}"] = []
    history["volfrac"] = []

    if print_progress:
        log_message("Computing Filters ...")
    H, Hs = createFilters(fe_solver, to_params)

    if print_progress:
        log_message("Finding elements with forces ...")
    elemsWithForces = find_elements_with_forces(mesh, fe_solver.bc.force, nDOFPerNode)

    fe_solver.set_material(mat_prop)
    KE = fe_solver.elem_stiff[0]

    # Body force term (optional)
    if print_progress:
        log_message("Finding body forces ...")
    if (fe_solver.elem_body_force is not None):
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None

    success = True
    errorMsg = "No errors."
    nFEAs = 0
    obj0 = None
    iteration = 0

    if use_continuation:
        initialize_SIMP_STRUCTURAL_PENALTY(1)
        initialize_SIMP_THERMAL_PENALTY(1)
    else:
        initialize_SIMP_STRUCTURAL_PENALTY(3)

    objective_name = getattr(to_params.Objective[0], "name", str(to_params.Objective[0]))
    constraint_names = [getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]

    def optimizationFunction(x):
        nonlocal nFEAs, obj0, iteration

        tIterationStart = time.time()

        # MMA variable
        x = np.asarray(x).flatten()

        # Physical density
        if to_params.APPLY_FILTER_TO_DENSITY:
            rho = (H @ x) / Hs
        else:
            rho = x

        # Grey stats should be measured on PHYSICAL density
        grey_elements = np.sum((rho > 0.1) & (rho < 0.9))
        fraction_grey = grey_elements / num_elems
        if print_progress:
            log_message(f"Percentage grey elements (rho): {fraction_grey*100:.2f}%")

        

        # Set density for plotting/postprocess
        fe_solver.mesh.setPseudoDensity(rho)

        if progress_callback is not None:
            progress_callback()
        if plot_progress:
            fe_solver.plot_pseudo_density_realtime(
                title=f"Iter {iteration}",
                iteration=iteration,
                external_plotter=plotter
            )

        # Solve + sensitivities (rho)
        if (feaMode == FEA_MODE.STRUCTURAL) or (feaMode == FEA_MODE.THERMAL):
            sol = fe_solver.solve(rho, material_model)
            fe_solver.postprocess()
            obj, grad_obj = compute_objective_and_gradient(feaMode, to_params, sol, rho, fe_solver, KE)
            c, dcdx = compute_constraint_and_gradient(feaMode, to_params, sol, rho, fe_solver, KE)

        elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
            temperature = fe_thermal_solver.solve(rho, material_model)
            thermo_elastic_force = fe_thermal_solver.get_thermoelastic_force(rho, material_model)
            fe_structural_solver.set_thermal_forces(thermo_elastic_force)
            displacement = fe_structural_solver.solve(rho, material_model)
            sol = displacement.copy()
            fe_structural_solver.postprocess()
            obj, grad_obj = compute_thermoelastic_compliance_and_gradient(
                rho, temperature, displacement, to_params, fe_thermal_solver, fe_structural_solver
            )
            c, dcdx = compute_constraint_and_gradient(feaMode, to_params, displacement, rho, fe_solver, KE)

        if obj0 is None:
            obj0 = obj

        # Normalize
        if any(c0 > 0.5 for c0 in c.flatten()):
            grad_obj *= 0.0

        obj = obj / obj0
        grad_obj = grad_obj / obj0

        # Changed
        if (nodal_body_force is not None):
            ce_body_force = (
                sol[mesh.edofMatStructural].reshape(num_elems, 24)
                * nodal_body_force[mesh.edofMatStructural].reshape(num_elems, 24)
            ).sum(1)
            grad_obj += 2.0 * ce_body_force * get_material_model_rho_sensitivity(rho, material_model)

   

        if not to_params.APPLY_FILTER_TO_DENSITY and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            eps = 1e-12
            grad_obj = (H @ (rho * grad_obj)) / Hs / (rho + eps)
        elif not to_params.APPLY_FILTER_TO_DENSITY and (to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION):
            grad_obj = (H @ grad_obj) / Hs

        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = np.min(grad_obj)

        if (to_params.ElemsToKeep is not None):
            grad_obj[to_params.ElemsToKeep] = np.min(grad_obj)

        if not to_params.APPLY_FILTER_TO_DENSITY:
            for m in range(len(to_params.Constraints)):
                if (to_params.Constraints[m][0] is TO_QOI.COMPLIANCE):
                    eps = 1e-12
                    dcdx[m, :] = (H @ (rho * dcdx[m, :])) / Hs / (rho + eps)
                elif (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
                    dcdx[m, :] = (H @ dcdx[m, :]) / Hs

        #CHAIN RULE BACK TO MMA VARIABLES d
        # rho = (H @ d) / Hs  => drho/dd = diag(1/Hs) @ H
        # df/dd = H^T @ (df/drho / Hs)
        if to_params.APPLY_FILTER_TO_DENSITY:
            grad_obj = (H.T @ (grad_obj / Hs))
            for m in range(dcdx.shape[0]):
                dcdx[m, :] = (H.T @ (dcdx[m, :] / Hs))

        # Logging/history uses physical density rho
        history["objective"].append(obj * obj0)
        history["volfrac"].append(np.mean(rho))
        for idx, val in enumerate(c.flatten()):
            history[f"constraint_{idx+1}"].append(val)

        # MMA expects grad as column vector
        grad_obj = np.asarray(grad_obj).reshape(-1, 1)

        if print_progress:
            msg_lines = ["-" * 50]
            msg_lines.append(f"Iteration: {iteration}")
            msg_lines.append(f"Min. Objective ({objective_name}): {obj*obj0:.3g}")
            inequality = "<="
            for idx, val in enumerate(c.flatten()):
                msg_lines.append(
                    f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*to_params.Constraints[idx][2]:.3g} {inequality} {to_params.Constraints[idx][2]:.3g}?"
                )
            msg_lines.append(f"Time for iteration {iteration}: {time.time() - tIterationStart:.2f} s")
            log_message("\n".join(msg_lines))

        iteration += 1
        nFEAs += 1

        if use_continuation and (iteration % 10 == 0):
            increment_SIMP_THERMAL_PENALTY(0.25)
            increment_SIMP_STRUCTURAL_PENALTY(0.25)

        return obj, grad_obj, c, dcdx

    # Initial density from volume fraction constraint if present
    initialDensity = 0.5
    for constraint in to_params.Constraints:
        if constraint[0] == TO_QOI.VOLUME_FRACTION:
            initialDensity = constraint[2]
            break

    x0 = initialDensity * np.ones(num_elems, dtype=float).reshape(-1, 1)
    lowerBound = 1e-6 * np.ones(num_elems, dtype=float).reshape(-1, 1)
    upperBound = (1 - 1e-6) * np.ones(num_elems, dtype=float).reshape(-1, 1)
    nVariables = num_elems
    nConstraints = len(to_params.Constraints)

    if print_progress:
        log_message("Calling MMA ...")

    xOptimal, f0val, df0dx, gval, dgdx, nFEAs = runMMA(
        nVariables,
        nConstraints,
        optimizationFunction,
        x0,
        lowerBound,
        upperBound,
        maxIterations=maxiteration,
        timeLimitSecs=timeLimitSecs,
        move_limit=move_limit,
        fTolerance=objective_tol,
        gTolerance=constraint_tol,
        kktTol=kkt_tol,
        verbose=False,
        progress_callback=progress_callback
    )

    # NOTE!! : xOptimal is MMA variable d_opt, not rho
    d_opt = np.asarray(xOptimal).flatten()
    if to_params.APPLY_FILTER_TO_DENSITY:
        rho = (H @ d_opt) / Hs
    else:
        rho = d_opt

    # Post-processing: binarization/hanging elements operate on PHYSICAL density
    if to_params.Eliminate_Hanging_Elements:
        x_sorted = np.sort(rho)
        threshold = x_sorted[int((1 - np.mean(rho)) * len(rho))]
        rho = np.where(rho < threshold, 0.0, 1.0)
        fe_solver.mesh.setPseudoDensity(rho)
        meshComponents = fe_solver.mesh.find_connected_components()
        if len(meshComponents) > 1:
            if print_progress:
                log_message("-" * 50)
                log_message("Removing hanging elements.")
            largest_component = max(meshComponents, key=len)
            rho[:] = 0.0
            rho[list(largest_component)] = 1.0
            fe_solver.mesh.setPseudoDensity(rho.flatten())

    elif binarize_topology:
        x_sorted = np.sort(rho)
        threshold = x_sorted[int((1 - np.mean(rho)) * len(rho))]
        rho = np.where(rho < threshold, 0.0, 1.0)

    # Final solve on rho
    if feaMode == FEA_MODE.THERMO_STRUCTURAL:
        temperature = fe_thermal_solver.solve(rho, material_model)
        thermo_elastic_force = fe_thermal_solver.get_thermoelastic_force(rho, material_model)
        fe_structural_solver.set_thermal_forces(thermo_elastic_force)
        displacement = fe_structural_solver.solve(rho, material_model)
        sol = displacement.copy()
        fe_structural_solver.postprocess()
        obj, grad_obj = compute_thermoelastic_compliance_and_gradient(
            rho, temperature, displacement, to_params, fe_thermal_solver, fe_structural_solver
        )
        c, dcdx = compute_constraint_and_gradient(feaMode, to_params, sol, rho, fe_solver, KE)
    else:
        fe_solver.mesh.setPseudoDensity(rho)
        sol = fe_solver.solve(rho, material_model)
        obj, grad_obj = compute_objective_and_gradient(feaMode, to_params, sol, rho, fe_solver, KE)
        c, dcdx = compute_constraint_and_gradient(feaMode, to_params, sol, rho, fe_solver, KE)

    grey_elements = np.sum((rho > 0.1) & (rho < 0.9))
    fraction_grey = grey_elements / num_elems

    history["objective"].append(obj)
    history["volfrac"].append(np.mean(rho))
    for idx, val in enumerate(c.flatten()):
        history[f"constraint_{idx+1}"].append(val)

    print("-" * 50)
    log_message(f"Final objective: {obj:.4g}, vf: {np.mean(rho):.3f}, grey: {100*fraction_grey:.2f}%")
    log_message(f"Total Time: {time.time() - tStart:.2f} s")
    log_message(f"Error: {errorMsg}")

    return np.asarray(sol), history, success, errorMsg, nFEAs


if __name__ == "__main__":
    from topopt_structural_benchmarks import *
    from topopt_thermal_benchmarks import *
    from topopt_thermostructural_benchmarks import *

    print("-" * 50)

    to_problem = StructuralTOExamples.Inverter
    # to_problem = ThermalTOExamples.FourCornersThermal
    # to_problem = ThermoStructuralTOExamples.MBBBeam

    run_topopt_mma(to_problem)
