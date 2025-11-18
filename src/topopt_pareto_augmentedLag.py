from topopt_common import *
import time
import numpy as np
from topopt_obj_cons_sensitivities import *

class AugmentedLagrangianHandler:
    """
    Implements augmented Lagrangian method for handling multiple constraints
    in thermo-elastic topology optimization (Section 3.3 of the paper).
    
    Note: Volume constraints are NOT handled here - they are managed by Pareto tracing.
    This handler is only for other constraints like compliance, stress, displacement, etc.
    """
    
    def __init__(self, constraint_names: list, zeta: float = 0.25, eta: float = 10.0):
        """
        Args:
            constraint_names: Names of constraints (for tracking/debugging)
            zeta: Parameter for penalty update (0 < zeta < 1)
            eta: Parameter for penalty increase (eta > 0)
        """
        self.constraint_names = constraint_names
        self.num_constraints = len(constraint_names)
        self.zeta = zeta
        self.eta = eta
        self.iteration = 0
        
        # Initialize Lagrangian multipliers and penalty parameters (Eq. 3.44)
        self.mu = np.ones(self.num_constraints) * 100.0  # Initial multipliers
        self.gamma = np.ones(self.num_constraints) * 10.0  # Initial penalties
        
        # Store previous constraint values for penalty update
        self.g_prev = np.zeros(self.num_constraints)
        self.g_current = np.zeros(self.num_constraints)
        
    def compute_auxiliary_lagrangian(self, g_i: float, mu_i: float, 
                                     gamma_i: float) -> float:
        """
        Compute auxiliary Lagrangian L̄ᵢ for constraint i (Equation 3.43)
        
        Args:
            g_i: Constraint value (should be <= 0 when satisfied)
            mu_i: Lagrangian multiplier
            gamma_i: Penalty parameter
            
        Returns:
            Auxiliary Lagrangian value
        """
        if mu_i + gamma_i * g_i > 0:
            return mu_i * g_i + 0.5 * gamma_i * g_i**2
        else:
            return -0.5 * mu_i**2 / gamma_i
    
    def compute_augmented_sensitivity(self, T_obj: np.ndarray, 
                                      T_constraints: list) -> np.ndarray:
        """
        Combine objective and constraint sensitivities using augmented Lagrangian (Eq. 3.45-3.46)
        
        Args:
            T_obj: Objective sensitivity (gradient)
            T_constraints: List of constraint sensitivities (same order as constraint_names)
            
        Returns:
            Combined augmented Lagrangian sensitivity
        """
        T_augmented = T_obj.copy()
        
        for i in range(self.num_constraints):
            g_i = self.g_current[i]
            T_i = T_constraints[i]
            
            # Compute gradient of auxiliary Lagrangian (Eq. 3.46)
            if self.mu[i] + self.gamma[i] * g_i > 0:
                # Constraint is active or violated
                T_augmented += (self.mu[i] + self.gamma[i] * g_i) * T_i
            # else: constraint is inactive, add nothing
                
        return T_augmented
    
    def update_lagrangian_multipliers(self, g_current: np.ndarray) -> None:
        """
        Update Lagrangian multipliers (Equation 3.47)
        
        Args:
            g_current: Current constraint values at local minimum x̂ᵏ
        """
        for i in range(self.num_constraints):
            self.mu[i] = max(self.mu[i] + self.gamma[i] * g_current[i], 0.0)
    
    def update_penalty_parameters(self, g_current: np.ndarray) -> None:
        """
        Update penalty parameters (Equation 3.48)
        
        Args:
            g_current: Current constraint values
        """
        k = self.iteration + 1
        
        for i in range(self.num_constraints):
            # Check if constraint is improving
            improving = (min(g_current[i], 0) >= self.zeta * min(self.g_prev[i], 0))
            
            if not improving:
                # Increase penalty parameter
                self.gamma[i] = max(self.eta * self.gamma[i], k**2)
            # else: keep penalty parameter the same
        
        # Store current for next iteration
        self.g_prev = g_current.copy()
    
    def check_constraints_satisfied(self, g_current: np.ndarray, 
                                    tolerance: float = 1e-6) -> bool:
        """
        Check if all constraints are satisfied
        
        Args:
            g_current: Current constraint values
            tolerance: Tolerance for constraint satisfaction
            
        Returns:
            True if all constraints satisfied
        """
        return np.all(g_current <= tolerance)
    
    def step(self, g_current: np.ndarray) -> dict:
        """
        Perform one augmented Lagrangian update step
        
        Args:
            g_current: Current constraint values (excluding volume constraint)
            
        Returns:
            Dictionary with status information
        """
        self.g_current = g_current.copy()
        
        # Check constraint satisfaction
        satisfied = self.check_constraints_satisfied(g_current)
        
        # Update multipliers
        self.update_lagrangian_multipliers(g_current)
        
        # Update penalties (only after first iteration)
        if self.iteration > 0:
            self.update_penalty_parameters(g_current)
        else:
            self.g_prev = g_current.copy()
        
        self.iteration += 1
        
        return {
            'constraints_satisfied': satisfied,
            'constraint_values': g_current.copy(),
            'constraint_names': self.constraint_names,
            'multipliers': self.mu.copy(),
            'penalties': self.gamma.copy(),
            'iteration': self.iteration
        }
    
    def get_status_string(self) -> str:
        """Get a formatted string with current status"""
        status = []
        for i, name in enumerate(self.constraint_names):
            status.append(f"{name}: g={self.g_current[i]:.3e}, μ={self.mu[i]:.2f}, γ={self.gamma[i]:.2f}")
        return " | ".join(status)


def topopt_pareto(fe_solver,
                  to_params,
                  rel_err: float = 0.025,
                  vol_decr_max: float = 0.05,
                  vol_decr_min: float = 0.0025,
                  min_local_iters: int = 2,
                  max_local_iters: int = 5,
                  xVoid: float = 0,
                  print_progress: bool = True,
                  plot_progress: bool = False,
                  debug: bool = False,
                  use_augmented_lagrangian: bool = True,  # Enable/disable AL for non-volume constraints
                  progress_callback=None, 
                  plotter=None  
                  )-> tuple[np.ndarray, dict]:
    """Pareto method for Topology Optimization with Augmented Lagrangian constraint handling.

    Volume constraints are handled by Pareto tracing. All other constraints (compliance, stress,
    displacement, etc.) can be handled using the Augmented Lagrangian method.

    Args:
        fe_solver: The structural FEA solver object.
        to_params: Topology optimization parameters
        rel_err: The relative error tolerance.
        vol_decr_max: The maximum volume decrease in each iteration.
        vol_decr_min: Minimum volume decrement
        min_local_iters: The minimum number of local iterations.
        max_local_iters: The maximum number of local iterations.
        xVoid: Void element density value
        print_progress: Print progress messages
        plot_progress: Plot progress
        debug: Enable debug output
        use_augmented_lagrangian: Use augmented Lagrangian for non-volume constraints
        progress_callback: Callback function for progress updates
        plotter: Plotter object for visualization

    Returns: A tuple containing the displacement field of the optimized structure
        and a dictionary containing the optimization history.
    """
    def log_message(msg):
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)  
    
    nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
    tStart = time.time()
    material_model = MaterialModel.SIMP 
    
    removeHangingElems = to_params.RemoveHangingElems
    if fe_solver.elem_body_force is not None and (np.linalg.norm(fe_solver.elem_body_force) > 0) and not removeHangingElems:
        removeHangingElems = True
    
    totalIter = 1
    
    # Initialize design field
    x = np.ones((fe_solver.mesh.num_elems))
    volfrac = 1.0
    
    history = {
        'objective': [],
        'compliance': [], 
        'volfrac': [],
        'constraints': [],  # Track non-volume constraint values
        'constraint_names': [],  # NEW: Track constraint names
        'multipliers': [],  # Track Lagrangian multipliers
        'penalties': []     # Track penalty parameters
    }
    
    if (print_progress):
        log_message("Computing Filters ...")
    [H, Hs] = createFilters(fe_solver, to_params)
    
    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force, nDOFPerNode)
    
    # Handle body forces
    if (fe_solver.elem_body_force is not None):
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None
    
    # Get element stiffness matrix
    if isinstance(fe_solver.mat_prop, list):
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(
                mp.youngs_modulus, mp.poissons_ratio, fe_solver.mesh.elem_size)
                for mp in fe_solver.mat_prop]
            KE = KE_list[0]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [hex_element_stiffness.hex8_stiffness_matrix_thermal(
                mp.thermal_conductivity, fe_solver.mesh.elem_size)
                for mp in fe_solver.mat_prop]
            KE = KE_list[0]	
        log_message("Assuming all elements have the same material properties")
    else:
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_structural(
                fe_solver.mat_prop.youngs_modulus,
                fe_solver.mat_prop.poissons_ratio,
                fe_solver.mesh.elem_size)
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE = hex_element_stiffness.hex8_stiffness_matrix_thermal(
                fe_solver.mat_prop.thermal_conductivity, 
                fe_solver.mesh.elem_size)
    
    # Initial FEA solve
    fe_solver.mesh.setPseudoDensity(x.flatten())
    sol = fe_solver.solve(x)
    fe_solver.postprocess()
    nFEAs = 1
    
    # Compute initial objective and sensitivity
    obj, grad_obj = compute_objective_and_gradient(to_params, sol, x, fe_solver, KE, material_model)
  
    # Helps to keep track of compliance changes
    compliance0, _ = compute_compliance_and_gradient( sol, x, fe_solver, KE, material_model)
    obj0 = obj
    T = -grad_obj / obj0
    
    history['objective'].append(obj)
    history['compliance'].append(compliance0)
    history['volfrac'].append(volfrac)
    
    # Add body force contribution to topological sensitivity
    if (nodal_body_force is not None):
        T_body = np.zeros(fe_solver.mesh.num_elems)
        for elem in range(fe_solver.mesh.num_elems):
            edof = fe_solver.mesh.edofMat[elem]
            T_body[elem] = (x[elem] * sol[edof] * nodal_body_force[edof]).sum()
        T += 2 * T_body
    
    if (elemsWithForces.size > 0):
        T[elemsWithForces] = np.max(T)
    if (to_params.ElemsToKeep is not None):
        T[to_params.ElemsToKeep] = np.max(T)
    T = (H * T) / Hs
    
    # Compute constraints and their sensitivities
    c, dcdx = compute_constraint_and_gradient(to_params, sol, x, fe_solver, KE, material_model)
    
    # Apply filters to constraint sensitivities
    for m in range(len(to_params.Constraints)):
        if (to_params.Constraints[m][0] is TO_QOI.COMPLIANCE):
            dcdx[m] = ((H * (x * dcdx[m])) / Hs / x)
        elif (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
            dcdx[m] = ((H * dcdx[m]) / Hs)
    
    T_dcdx = -dcdx  # Constraint sensitivities (negative for minimization)
    
    # ========================================================================
    # INITIALIZE AUGMENTED LAGRANGIAN (ONLY FOR NON-VOLUME CONSTRAINTS)
    # ========================================================================
    al_handler = None
    constraint_info = []  # List of tuples: (constraint_index, constraint_name, constraint_value)
    volume_constraint_idx = None
    
    if use_augmented_lagrangian:
        # Identify non-volume constraints
        # Assuming to_params.Constraints is a list where each element is:
        # (constraint_type, sense, value) or similar structure
        for m in range(len(to_params.Constraints)):
            constraint_tuple = to_params.Constraints[m]
            constraint_type = constraint_tuple[0]  # First element is the type
            constraint_value = constraint_tuple[2] if len(constraint_tuple) > 2 else constraint_tuple[1]  # Value
            
            if constraint_type == TO_QOI.VOLUME_FRACTION:
                # Store volume constraint info but DON'T add to AL
                volume_constraint_idx = m
                volFractionConstraint = constraint_value
                if print_progress:
                    log_message(f"Volume constraint (handled by Pareto): target = {constraint_value:.3f}")
            else:
                # Add to augmented Lagrangian
                # For non-volume constraints, we assume they are of the form: value <= constraint_value
                # i.e., g = value - constraint_value <= 0
                constraint_name = f"{constraint_type.name}"
                constraint_info.append((m, constraint_name, constraint_value))
        
        # Create AL handler only if there are non-volume constraints
        if len(constraint_info) > 0:
            constraint_names = [info[1] for info in constraint_info]
            al_handler = AugmentedLagrangianHandler(constraint_names)
            
            # Store constraint names in history
            history['constraint_names'] = constraint_names
            
            # Compute initial constraint values (normalized: g <= 0 when satisfied)
            # Assuming constraints are of the form: c[m] <= constraint_value
            # So g = c[m] - constraint_value <= 0
            g_current = np.zeros(len(constraint_info))
            for i, (m, name, value) in enumerate(constraint_info):
                g_current[i] = c[m] - value  # g <= 0 when satisfied
            
            al_status = al_handler.step(g_current)
            history['constraints'].append(g_current.copy())
            history['multipliers'].append(al_status['multipliers'])
            history['penalties'].append(al_status['penalties'])
            
            # Combine sensitivities using augmented Lagrangian
            T_constraints_for_AL = [T_dcdx[info[0]] for info in constraint_info]
            T = al_handler.compute_augmented_sensitivity(T, T_constraints_for_AL)
            
            if print_progress:
                log_message(f"Augmented Lagrangian initialized: {len(constraint_info)} non-volume constraints")
                for i, (m, name, value) in enumerate(constraint_info):
                    satisfied_str = "✓" if g_current[i] <= 0 else "✗"
                    log_message(f"  {satisfied_str} {name}: current={c[m]:.3g}, target≤{value:.3g}, g={g_current[i]:.3e}")
        else:
            use_augmented_lagrangian = False
            if print_progress:
                log_message("No non-volume constraints found - Augmented Lagrangian disabled")
    else:
        # Find volume constraint manually if AL is disabled
        for m in range(len(to_params.Constraints)):
            if to_params.Constraints[m][0] == TO_QOI.VOLUME_FRACTION:
                volume_constraint_idx = m
                volFractionConstraint = to_params.Constraints[m][2] if len(to_params.Constraints[m]) > 2 else to_params.Constraints[m][1]
                break
    
    # Check that we found a volume constraint
    if volume_constraint_idx is None:
        raise ValueError("No volume constraint found! Pareto method requires a volume constraint.")
    
    if (print_progress):
        log_message(f"vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g}, #FEA={nFEAs:2d}")
    
    vol_decr = vol_decr_max
    success = True
    terminatePareto = False
    errorMsg = "No errors."
    
    # Damping and smoothing parameters
    wtDamping = 0.5
    nSmoothSteps = 2

    
    # ========================================================================
    # MAIN PARETO LOOP
    # ========================================================================
    while volfrac > volFractionConstraint:
        if (plot_progress):
            if progress_callback is not None:
                progress_callback()
            fe_solver.plot_mesh(plotter=plotter, plot_bc=False, auto_close=False, 
                              title=f'Volfrac: {volfrac:0.3f}')
        
        # Move to next volume fraction
        volfrac = max(volFractionConstraint, volfrac - vol_decr)
        if (debug):
            log_message("-" * 50)
            log_message(f"Attempting v={volfrac:.3f}")
        
        # Initialize local iteration variables
        localIter = 0
        JTemp = history['compliance'][-1]
        JPrev = JTemp
        JPrevPrev = JTemp
        TPrev = T.copy()
        xPrev = x.copy()
        innerLoopSuccess = True
        
        # Inner fixed-point iteration loop
        while True:
            if (debug):
                log_message(f"Local Iteration: {localIter}/{max_local_iters}, JTemp: {JTemp:.3g}, JPrev: {JPrev:.3g}")
            
            # Check convergence
            if localIter >= min_local_iters:
                if abs(JPrev - JTemp)/abs(JTemp) < rel_err or abs(min(JPrev, JPrevPrev) - JTemp)/abs(JTemp) < rel_err:
                    vol_frac_success = volfrac
                    innerLoopSuccess = True
                    break
            
            if (localIter >= max_local_iters) or abs(JTemp) > 10 * history['objective'][-1]:
                innerLoopSuccess = False
                x = xPrev.copy()
                T = TPrev.copy()
                fe_solver.mesh.setPseudoDensity(x.flatten())
                JTemp = JPrev
                volfrac = volfrac + vol_decr
                vol_decr *= 0.75
                if (debug):
                    log_message("**Failed to converge, restoring previous design")
                    log_message(f"Previous successful vol_frac: {vol_frac_success:.5g}")
                    log_message(f"Reducing vol_decr to: {vol_decr:.5g}")
                if vol_decr < vol_decr_min:
                    terminatePareto = True
                break
            
            # Find cutoff value and update design
            value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))]
            x = np.ones((fe_solver.mesh.num_elems))
            x[T < value] = xVoid
            
            fe_solver.mesh.setPseudoDensity(x.flatten())
            if (removeHangingElems):
                meshComponents = fe_solver.mesh.find_connected_components()
                if (len(meshComponents) > 1):
                    largest_component = max(meshComponents, key=len)
                    x[:] = xVoid
                    x[list(largest_component)] = 1.0
                    fe_solver.mesh.setPseudoDensity(x.flatten())
            
            JPrevPrev = JPrev
            JPrev = JTemp
            
            # Solve FEA
            sol = fe_solver.solve(x, material_model)
            fe_solver.postprocess()
            nFEAs += 1
            
            # Compute objective and sensitivity
            obj, grad_obj = compute_objective_and_gradient(to_params, sol, x, fe_solver, KE, material_model)
            
            # Keep track of compliance changes
            compliance, _ = compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)
            
            TTemp = -grad_obj/obj0
            JTemp = compliance
            
            if (to_params.Objective[0] == TO_QOI.COMPLIANCE):
                T = TTemp.copy()
            else:
                T = np.where(x == 0, TPrev.copy(), TTemp.copy())
            
            # Add body force contribution
            if (nodal_body_force is not None):
                T_body = np.zeros(fe_solver.mesh.num_elems)
                for elem in range(fe_solver.mesh.num_elems):
                    edof = fe_solver.mesh.edofMat[elem]
                    T_body[elem] = (x[elem] * sol[edof] * nodal_body_force[edof]).sum()
                T += 2 * T_body
            
            # Apply smoothing
            for _ in range(nSmoothSteps):
                T = (H * T) / Hs
            
            T /= np.max(np.abs(T))
            T = ((1 - wtDamping) * T + wtDamping * TPrev)
            
            if (elemsWithForces.size > 0):
                T[elemsWithForces] = np.max(T)
            if (to_params.ElemsToKeep is not None):
                T[to_params.ElemsToKeep] = np.max(T)
            
            # Compute constraints and their sensitivities
            c, dcdx = compute_constraint_and_gradient(to_params, sol, x, fe_solver, KE, material_model)
            
            # Apply filters
            for m in range(len(to_params.Constraints)):
                if (to_params.Constraints[m][0] is TO_QOI.COMPLIANCE):
                    dcdx[m] = ((H * (x * dcdx[m])) / Hs / x)
                elif (to_params.Constraints[m][0] is not TO_QOI.VOLUME_FRACTION):
                    dcdx[m] = ((H * dcdx[m]) / Hs)
            
            T_dcdx = -dcdx
            
            # ================================================================
            # UPDATE AUGMENTED LAGRANGIAN (ONLY FOR NON-VOLUME CONSTRAINTS)
            # ================================================================
            if use_augmented_lagrangian and al_handler is not None:
                # Compute current constraint values (excluding volume)
                g_current = np.zeros(len(constraint_info))
                for i, (m, name, value) in enumerate(constraint_info):
                    g_current[i] = c[m] - value  # g <= 0 when satisfied
                
                # Update augmented Lagrangian parameters
                al_status = al_handler.step(g_current)
                
                # Combine sensitivities
                T_constraints_for_AL = [T_dcdx[info[0]] for info in constraint_info]
                T = al_handler.compute_augmented_sensitivity(T, T_constraints_for_AL)
                
                if debug:
                    log_message(f"  AL Status: {al_handler.get_status_string()}")
                    log_message(f"  AL Satisfied: {al_status['constraints_satisfied']}")
            
            localIter += 1
            totalIter += 1
        
        # Check termination
        if terminatePareto:
            if (volfrac > 1.1 * volFractionConstraint):
                success = False
                errorMsg = f"vf {volFractionConstraint:0.3f} not reached"
                log_message("-" * 50)
                log_message("Pareto: Failed to reach volume fraction.")
                log_message("1. Check for incorrect symmetry constraints")
                log_message("2. Increase mesh size")
            break
        
        if innerLoopSuccess:
            # Remove disconnected components
            meshComponents = fe_solver.mesh.find_connected_components()
            if (len(meshComponents) > 1):
                largest_component = max(meshComponents, key=len)
                x[:] = xVoid
                x[list(largest_component)] = 1.0
                fe_solver.mesh.setPseudoDensity(x.flatten())
                volfrac = np.mean(x)
            
            # Update history
            history['objective'].append(obj)
            history['compliance'].append(compliance)
            history['volfrac'].append(volfrac)
            
            # Store constraint information (only non-volume constraints)
            if use_augmented_lagrangian and al_handler is not None:
                history['constraints'].append(g_current.copy())
                history['multipliers'].append(al_status['multipliers'])
                history['penalties'].append(al_status['penalties'])
            
            scale = (compliance / compliance0)**2
            vol_decr = max(vol_decr_min, min(vol_decr, vol_decr_max / scale))
            
            if (print_progress):
                msg = f"vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g}, #FEA={nFEAs:2d}"
                if use_augmented_lagrangian and al_handler is not None:
                    msg += f", AL_satisfied={al_status['constraints_satisfied']}"
                    # Show constraint violations
                    for i, (m, name, value) in enumerate(constraint_info):
                        if g_current[i] > 0:  # Constraint violated
                            msg += f", {name}_viol={g_current[i]:.2e}"
                log_message(msg)
            
            fe_solver.mesh.setPseudoDensity(x.flatten())
    
    totalTime = time.time() - tStart
    
    log_message(f"Final: vf={history['volfrac'][-1]:.3f}, obj={history['objective'][-1]:.3g}, #FEA={nFEAs:2d}")
    if use_augmented_lagrangian and al_handler is not None:
        log_message(f"Final constraint status:")
        for i, (m, name, value) in enumerate(constraint_info):
            satisfied = "✓" if g_current[i] <= 0 else "✗"
            log_message(f"  {satisfied} {name}: {c[m]:.3g} (target ≤ {value:.3g})")
    log_message(f"Total Time: {totalTime:.2f} s")
    log_message(f"Error: {errorMsg}")
    
    return sol, history, success, errorMsg, nFEAs


if __name__ == "__main__":    
    from topopt_structural_benchmarks import *
    from topopt_thermal_benchmarks import *
    
    print("-" * 50)
    to_problem = StructuralTOExamples.LBracketTopLoadStressObjective
    
    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(to_problem)
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(to_problem)
    
    print(f"Running {to_problem.name}...") 
    print("-" * 50)
    
    plot_progress = True
    print_progress = True
    debug = False
    use_augmented_lagrangian = True  # Enable AL for non-volume constraints
    
    solver = lin_solv.Solvers.PARDISO
    dsolver = deflation.DeflationSolver()
    if (to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF):
        solver = lin_solv.Solvers.DPCG
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, 
                     round(3 * mesh.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
        dsolver.W = dsolver.W[bc.free_dofs, :]
    
    if (to_problem in StructuralTOExamples):
        fe_solver = hex_structural_fea.HexStructuralFEA(mesh=mesh,
                    mat_prop=mat_prop,
                    bc=bc,
                    solver=solver,
                    dsolver=dsolver,
                    elem_body_force=elem_body_force)
    elif (to_problem in ThermalTOExamples):
        fe_solver = hex_thermal_fea.HexThermalFEA(mesh=mesh,
                    mat_prop=mat_prop,
                    bc=bc,
                    solver=solver,
                    dsolver=dsolver,
                    elem_body_force=elem_body_force)
    
    print('Solver: ', fe_solver.solver.name)
    print("nNodes: ", fe_solver.mesh.num_nodes)
    print("nElem: ", fe_solver.mesh.num_elems)	
    
    startTime = time.time()
    
    print("OptimizationMethod: Pareto with Augmented Lagrangian (non-volume constraints only)")
    sol, history, success, errorMsg, nFEAs = topopt_pareto(
        fe_solver=fe_solver,
        to_params=to_params,
        plot_progress=plot_progress,
        debug=debug,
        use_augmented_lagrangian=use_augmented_lagrangian
    )
    
    timeTaken = time.time() - startTime
    print(f"Time taken: {timeTaken:.0f} s")
    if not success:
        print(f"Error: {errorMsg}")
    
    title = f"Pareto+AL: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {nFEAs:3d}, time: {timeTaken:.0f} s"
    fe_solver.plot_mesh(title=title, save_path=None)
    
    # Plot convergence
    n_plots = 3 if (len(history.get('constraints', [])) > 0) else 1
    plt.figure(figsize=(8, 6))
    
    plt.subplot(1, n_plots, 1)
    plt.plot(history['volfrac'], history['objective'], marker='o')
    plt.xlabel('Volume Fraction')
    plt.ylabel('Objective')
    plt.title('Pareto Curve: Volume vs Objective')
    plt.grid(True)
    
    if 'constraints' in history and len(history['constraints']) > 0:
        # Get constraint names from history
        constraint_names = history.get('constraint_names', [f'Constraint {i+1}' for i in range(len(history['constraints'][0]))])
        
        plt.subplot(1, n_plots, 2)
        constraints_array = np.array(history['constraints'])
        for i in range(constraints_array.shape[1]):
            constraint_name = constraint_names[i] if i < len(constraint_names) else f'g_{i+1}'
            plt.plot(constraints_array[:, i], marker='o', label=constraint_name)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Satisfied (g≤0)')
        plt.xlabel('Pareto Step')
        plt.ylabel('Constraint Value g')
        plt.title('Non-Volume Constraint Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, n_plots, 3)
        multipliers_array = np.array(history['multipliers'])
        penalties_array = np.array(history['penalties'])
        ax1 = plt.gca()
        for i in range(multipliers_array.shape[1]):
            ax1.plot(multipliers_array[:, i], marker='s', label=f'μ_{i+1}')
        ax1.set_xlabel('Pareto Step')
        ax1.set_ylabel('Lagrangian Multiplier μ', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        for i in range(penalties_array.shape[1]):
            ax2.plot(penalties_array[:, i], marker='^', linestyle='--', 
                    color=f'C{i+multipliers_array.shape[1]}', label=f'γ_{i+1}')
        ax2.set_ylabel('Penalty Parameter γ', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
        
        plt.title('AL Parameters Evolution')
    
    plt.tight_layout()
    plt.show()