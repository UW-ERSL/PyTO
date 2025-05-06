import jax
from topopt_common import *
import hex_structural_fea as sfea
import jax.numpy as jnp
from jax import grad
import numpy as np
import matplotlib.pyplot as plt
import time

def normalize_field(T): # this normalizes the T field. I think it is better than just deiving by the max value, because it is more robust to outliers.
    #T_centered = T - np.mean(T)
    #scale = np.percentile(np.abs(T_centered), 95) + 1e-6  # prevent division by very small numbers
    Tmin = np.min(T)
    Tmax = np.max(T)
    TScaled = (T - Tmin) / (Tmax - Tmin)  # Normalize to [0, 1]
    return TScaled

def compute_stress_jax(u: jnp.ndarray, mesh, mat_prop):
    gradN = (1 / 8) * jnp.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])

    edof = mesh.edofMat

    u_x = u[edof[:, 0::3]]
    u_y = u[edof[:, 1::3]]
    u_z = u[edof[:, 2::3]]

    uGrad = gradN @ u_x.T
    vGrad = gradN @ u_y.T
    wGrad = gradN @ u_z.T

    strain = jnp.stack([
        uGrad[0], vGrad[1], wGrad[2],
        uGrad[1] + vGrad[0],
        uGrad[2] + wGrad[0],
        vGrad[2] + wGrad[1]
    ], axis=0).T

    if isinstance(mat_prop, list):
        D_list = []
        for mp in mat_prop:
            E = mp.youngs_modulus
            nu = mp.poissons_ratio
            coeff = E / ((1 + nu) * (1 - 2 * nu))
            D = coeff * jnp.array([
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
            ])
            D_list.append(D)
        D_stack = jnp.stack(D_list)
        elem_ids = mesh.elemComponentId
        D_elems = D_stack[elem_ids]
        stress = jnp.einsum('eij,ej->ei', D_elems, strain)
    else:
        E = mat_prop.youngs_modulus
        nu = mat_prop.poissons_ratio
        coeff = E / ((1 + nu) * (1 - 2 * nu))
        D = coeff * jnp.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
        ])
        stress = strain @ D.T
    return stress

def stress_pnorm_function_jax(mesh, mat_prop, x, p=8):
    def S(u_flat):
        sigma = x*compute_stress_jax(u_flat, mesh, mat_prop)
        von_mises = jnp.sqrt(
            0.5 * ((sigma[:, 0] - sigma[:, 1])**2 +
                   (sigma[:, 1] - sigma[:, 2])**2 +
                   (sigma[:, 2] - sigma[:, 0])**2)
            + 3 * (sigma[:, 3]**2 + sigma[:, 4]**2 + sigma[:, 5]**2)
        )
        return (jnp.sum(von_mises ** p)) ** (1. / p)
    return S

def voigt_to_tensors(stress_voigt: np.ndarray, strain_voigt: np.ndarray):
    """
    Convert both stress and strain from Voigt (n_elem, 6) to (n_elem, 3, 3) tensor form.

    Parameters:
    -----------
    stress_voigt : np.ndarray
        Stress in Voigt notation, shape (n_elem, 6)

    strain_voigt : np.ndarray
        Strain in Voigt notation, shape (n_elem, 6)

    Returns:
    --------
    stress_tensor : np.ndarray
        Stress tensor, shape (n_elem, 3, 3)

    strain_tensor : np.ndarray
        Strain tensor, shape (n_elem, 3, 3), using engineering convention (shear halved)
    """
    stress_tensor = np.array([
        [stress_voigt[:, 0], stress_voigt[:, 3], stress_voigt[:, 4]],
        [stress_voigt[:, 3], stress_voigt[:, 1], stress_voigt[:, 5]],
        [stress_voigt[:, 4], stress_voigt[:, 5], stress_voigt[:, 2]]
    ]).transpose(2, 0, 1)

    strain_tensor = np.array([
        [strain_voigt[:, 0], strain_voigt[:, 3] / 2, strain_voigt[:, 4] / 2],
        [strain_voigt[:, 3] / 2, strain_voigt[:, 1], strain_voigt[:, 5] / 2],
        [strain_voigt[:, 4] / 2, strain_voigt[:, 5] / 2, strain_voigt[:, 2]]
    ]).transpose(2, 0, 1)

    return stress_tensor, strain_tensor


def compute_stress_topological_sensitivity(fe_solver: sfea.HexStructuralFEA,
                                           rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the stress-based topological sensitivity using the adjoint method.

    Steps:
    - Solve and postprocess primal to get σ(u)
    - Solve adjoint to get λ, then postprocess to get ε(λ)
    - Compute T_s using inner product and trace terms
    - Restore primal state

    Parameters:
    -----------
    fe_solver : HexStructuralFEA
        The FEA solver with all mesh, material, boundary info.
    rhs : np.ndarray
        RHS vector for the adjoint system.
    x : np.ndarray
        Pseudo-density vector used for stiffness scaling.

    Returns:
    --------
    T_s : np.ndarray
        Topological sensitivity for stress constraint.
    """

    # Save original force and primal solution
    original_force = fe_solver.bc.force.copy()
    u_primal = fe_solver.solve(x)
    fe_solver.sol = u_primal
    fe_solver.postprocess()
    stress_voigt = fe_solver.stressComponents.copy()  # σ(u)

    # Solve adjoint system
    try:
        fe_solver.bc.force = rhs.copy()
        lambda_vec = fe_solver.solve(x)
    finally:
        fe_solver.bc.force = original_force  # Restore force BC

    # Get strain field from λ
    fe_solver.sol = lambda_vec
    fe_solver.postprocess()
    strain_voigt = fe_solver.strainComponents.copy()  # ε(λ)

    # Restore primal u and postprocess for future use
    fe_solver.sol = u_primal
    fe_solver.postprocess()

    # Convert to full tensors
    stress_tensor, strain_tensor = voigt_to_tensors(stress_voigt, strain_voigt)

    # Compute final T_s
    nu = fe_solver.mat_prop.poissons_ratio
    tr_sigma = np.trace(stress_tensor, axis1=1, axis2=2)
    tr_eps = np.trace(strain_tensor, axis1=1, axis2=2)
    double_contract = np.sum(stress_tensor * strain_tensor, axis=(1, 2))

    T_s = (4 / (1 + nu)) * double_contract - ((1 - 3 * nu) / (1 - nu ** 2)) * tr_sigma * tr_eps
    #T_s = x*T_s # Scale by pseudo-density
    return T_s


def compute_compliance3D_topological_sensitivity(mat_prop, strain_voigt, stress_voigt): # I saw in paper that 3D top sensitivity is different from 2D, so I made a new function for it. but in current run I didn't use it.
    """
    Compute the topological sensitivity for compliance in 3D.
    
    Parameters:
    -----------
    mat_prop : MaterialProperty object (must have .youngs_modulus and .poissons_ratio)
    strain_voigt : (n_elem, 6) array of strain tensors in Voigt notation
    stress_voigt : (n_elem, 6) array of stress tensors in Voigt notation
    x : (n_elem,) array of pseudo-densities

    Returns:
    --------
    T_J : (n_elem,) array of topological sensitivity values
    """
    E = mat_prop.youngs_modulus
    nu = mat_prop.poissons_ratio

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Compute trace(σ) and trace(ε)
    trace_stress = stress_voigt[:, 0] + stress_voigt[:, 1] + stress_voigt[:, 2]
    trace_strain = strain_voigt[:, 0] + strain_voigt[:, 1] + strain_voigt[:, 2]

    # Compute σ : ε = sum of component-wise products
    double_contract = np.sum(stress_voigt * strain_voigt, axis=1)

    # Final topological sensitivity formula
    T_J = -20 * mu * double_contract - (3 * lam - 2 * mu) * trace_stress * trace_strain

    return T_J

def update_multipliers(mu, gamma, g_val, eta=10.0, zeta=0.25): # ignore this
    mu_new = max(mu - g_val, 0.0)
    gamma_new = zeta * gamma if g_val <= 0 else max(eta * gamma, zeta * gamma)
    return mu_new, gamma_new

def plot_Tfield(fe_solver, T_sensitivity, threeD = False, title='Topological Compliance Sensitivity', cmap='coolwarm'):
    """
    Plots the T_comp (topological sensitivity for compliance) over the mesh.

    Args:
        fe_solver: The structural FEA solver object (must have mesh information).
        T_sensitivity: Array of topological sensitivity values per element (num_elems,).
        title: Title of the plot.
        cmap: Matplotlib colormap.
    """
    # Extract element centroids
    centroids = fe_solver.mesh.elem_centers  # (num_elems, 3)

    # Basic 3D scatter plot
    if threeD:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=T_sensitivity, cmap=cmap)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()

    # Optional 2D projection for clearer inspection
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc2 = ax2.scatter(centroids[:, 0], centroids[:, 1], c=T_sensitivity, cmap=cmap)
    fig2.colorbar(sc2, ax=ax2)
    ax2.set_title(title + ' (2D projection)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    return

def topopt_pareto_volume_stress(fe_solver: sfea.HexStructuralFEA, to_params, allowable_stress=None, p_norm=4,
                                 rel_err=0.025, vol_decr_max=0.025, vol_decr_min=0.001,
                                 min_local_iters=2, max_local_iters=8, xVoid=0,
                                 debug=False, weight_stress = 0.9, wtDamping=0):

    tStart = time.time()
    x = np.ones(fe_solver.mesh.num_elems)
    volfrac = 1.0
    to_params.RelativeFilterRadius = 2.5
    [H, Hs] = createFilters(fe_solver, to_params)
    
    
    history = {'volume': [], 'compliance': [], 'max_stress': []}
    vol_decr = vol_decr_max
    success = True
    terminatePareto = False
    nFEAs = 0

    # removeHangingElems = to_params.RemoveHangingElems
    removeHangingElems = True
    if fe_solver.elem_body_force is not None and np.linalg.norm(fe_solver.elem_body_force) > 0 and not removeHangingElems:
        removeHangingElems = True

    elemsWithForces = find_elements_with_forces(fe_solver.mesh, fe_solver.bc.force)

    if fe_solver.elem_body_force is not None:
        elem_force = fe_solver.elem_body_force.copy()
        nNodes = fe_solver.mesh.num_nodes
        nodal_body_force = np.zeros((nNodes * 3,))
        nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
        nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
        nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    else:
        nodal_body_force = None

    u = np.asarray(fe_solver.solve(x))
    nFEAs += 1
    fe_solver.postprocess()
    compliance_val = float(fe_solver.total_force.T @ u)
    max_stress = np.max(fe_solver.vonMisesStress)



     # Step 1: Compliance topological sensitivity
    T_comp = computeTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio, fe_solver.strainComponents, fe_solver.stressComponents,x)
    # T_comp = compute_compliance3D_topological_sensitivity(fe_solver.mat_prop, fe_solver.strainComponents, fe_solver.stressComponents)

    #***************if you want to use JAX for top sensitivity, uncomment the first two lines and comment the third line ***************

    rhs = -grad(stress_pnorm_function_jax(fe_solver.mesh, fe_solver.mat_prop, x, p=p_norm))(jnp.array(u))
    T_stress = compute_stress_topological_sensitivity(fe_solver, rhs, x)

    # T_stress = compute_stress_top_sensitivity_explicit(fe_solver,p_norm)

    #***********************************************************************************************************************************
   
    # calculate the total topological sensitivity, using the normalized T_comp and T_stress fields
    T_comp_normalized = normalize_field(T_comp)
    T_stress_normalized = normalize_field(T_stress)


    T = (1- weight_stress)* T_comp_normalized + (weight_stress) * T_stress_normalized 
    
    if nodal_body_force is not None:
        T_body = np.zeros(fe_solver.mesh.num_elems)
        for elem in range(fe_solver.mesh.num_elems):
            edof = fe_solver.mesh.edofMat[elem]
            T_body[elem] = (x[elem] * u[edof] * nodal_body_force[edof]).sum()
        T += 2 * T_body

    T = (H * T) / Hs


    history['compliance'].append(compliance_val)
    history['volume'].append(volfrac)
    history['max_stress'].append(max_stress)
    J_max = compliance_val

    plot_progress = True
    while volfrac > to_params.DesiredVolFraction:
        if (plot_progress):
            fe_solver.plot_mesh(plot_bc = False,auto_close = False, title = f'Volfrac: {volfrac:0.3f}')
        volfrac = max(to_params.DesiredVolFraction, volfrac - vol_decr)
        if debug:
            print("-" * 50)
            print(f"Attempting v={volfrac:.3f}, Weight stress: {weight_stress:.3f}, vol_decr: {vol_decr:.3f}")

        localIter = 0
        JTemp = history['compliance'][-1]
        JPrev = JTemp
        JPrevPrev = JTemp
        xPrev = x.copy()
        TPrev = T.copy()
        innerLoopSuccess = True

        while True:
           
            if (elemsWithForces.size > 0):
                T[elemsWithForces] = np.max(T)

            if (to_params.ElemsToKeep is not None):
                T[to_params.ElemsToKeep] = np.max(T)

            value = np.sort(T.flatten())[int(fe_solver.mesh.num_elems * (1 - volfrac))] #sort in acscending order and get the value at the index corresponding to the volume fraction
            x = np.ones((fe_solver.mesh.num_elems))
            x[T < value] = xVoid
            fe_solver.mesh.setPseudoDensity(x)

            if removeHangingElems:
                meshComponents = fe_solver.mesh.find_connected_components()
                if len(meshComponents) > 1:
                    largest_component = max(meshComponents, key=len)
                    x[:] = xVoid
                    x[list(largest_component)] = 1.0
                    fe_solver.mesh.setPseudoDensity(x)

            JPrevPrev = JPrev  # Store previous to previous value
            JPrev = JTemp  # Store previous value
            u = np.asarray(fe_solver.solve(x)) 
            nFEAs += 1
            JTemp = float(fe_solver.total_force.T @ u) # update new complaince value
            fe_solver.postprocess()
            max_stress = np.max(fe_solver.vonMisesStress)
            print(f"Max stress: {max_stress:.3g}")

            if debug:
                print(f"Local Iteration: {localIter}/{max_local_iters}, JTemp: {JTemp:.3g}, JPrev: {JPrev:.3g}")
                fe_solver.plot_mesh(plot_bc = False,auto_close = False, title = f'Volfrac: {volfrac:0.3f}')


            # --- 2. Normal convergence check after minimum local iterations ---
            if localIter >= min_local_iters:
                relative_change = abs(JPrev - JTemp) / (abs(JTemp) + 1e-8)
                relative_change_min = abs(min(JPrev, JPrevPrev) - JTemp) / (abs(JTemp) + 1e-8)
               
                if (relative_change < rel_err) or (relative_change_min < rel_err):
                    print(f"**Converged: relative compliance change {relative_change:.2e} or min change {relative_change_min:.2e}")
                    innerLoopSuccess = True
                    break

        
            # --- 3. Divergence or too many iterations ---
            if (localIter >= max_local_iters) or (abs(JTemp) > 10 * history['compliance'][0]):
                print("**Failed to converge within max iterations, restoring previous design.")
                x = xPrev.copy()
                T = TPrev.copy()
                fe_solver.mesh.setPseudoDensity(x)
                volfrac += vol_decr
                vol_decr *= 0.9
                #weight_stress *= 0.95  # Also slightly reduce weight if stagnation
                innerLoopSuccess = False
                if vol_decr < vol_decr_min:
                    terminatePareto = True
                break


            T_comp = computeTopologicalSensitivity(fe_solver.mat_prop.poissons_ratio, fe_solver.strainComponents, fe_solver.stressComponents,x)
            # T_comp = compute_compliance3D_topological_sensitivity(fe_solver.mat_prop, fe_solver.strainComponents, fe_solver.stressComponents)


            #***************if you want to use JAX for top sensitivity, uncomment this part***************

            rhs = -grad(stress_pnorm_function_jax(fe_solver.mesh, fe_solver.mat_prop, p=p_norm))(jnp.array(u))
            T_stress = compute_stress_topological_sensitivity(fe_solver, rhs, x)

            # *******************************************************************************************************************

            # Calculate the total topological sensitivity
            T_comp_normalized = normalize_field(T_comp)
            T_stress_normalized = normalize_field(T_stress)
            T = (1- weight_stress)* T_comp_normalized + (weight_stress) * T_stress_normalized

            if nodal_body_force is not None:
                T_body = np.zeros(fe_solver.mesh.num_elems)
                for elem in range(fe_solver.mesh.num_elems):
                    edof = fe_solver.mesh.edofMat[elem]
                    T_body[elem] = (x[elem] * u[edof] * nodal_body_force[edof]).sum()
                T += 2 * T_body

            T = (H * T) / Hs
            T = (1-wtDamping)*T + wtDamping*TPrev



            localIter += 1

        if terminatePareto:
            if volfrac > 1.1 * to_params.DesiredVolFraction:
                success = False
                errorMsg = f"vf {to_params.DesiredVolFraction:0.3f} not reached"
                print("-" * 50)
                print("Pareto: Failed to reach volume fraction.")
                break

        if innerLoopSuccess:
            history['compliance'].append(JTemp)
            history['volume'].append(volfrac)
            history['max_stress'].append(max_stress)
            scale = history['compliance'][-1] / history['compliance'][0]
            vol_decr = max(vol_decr_min, min(vol_decr, vol_decr_max / scale))
            print(f"vf={history['volume'][-1]:.3f}, J={history['compliance'][-1]:.3g}, max_stress={history['max_stress'][-1]:.3g}, #FEA={nFEAs:2d}")
            fe_solver.mesh.setPseudoDensity(x)

    timeTaken = time.time() - tStart
    print(f"Final vf: {volfrac:.3f},  J: {compliance_val:.4g}, max_stress: {max_stress:.3g}, time: {timeTaken:.2f} s")
    return u, history, success, '' if success else errorMsg, nFEAs


if __name__ == "__main__":
    # Example usage of the topopt_pareto_volume_stress function
    # Topopt demo

    jax.config.update("jax_enable_x64", True)
    from topopt_benchmarks import *
    import time

    print("-" * 50)
    to_problem = StructuralTOExamples.LBracketMidLoad # Choose the TO problem
    print(f"Running {to_problem.name}...") 
    print("-" * 50)
    solver = lin_solv.Solvers.PARDISO # # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
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

    fe_solver = sfea.HexStructuralFEA(mesh = mesh,
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
    #fe_solver.plot_mesh(title = title, save_path = None)

    startTime = time.time()

    u, history, success, errorMsg, nFEAs = topopt_pareto_volume_stress(fe_solver=fe_solver,
                                                                        to_params=to_params,
                                                                        allowable_stress=3e8,       # 300 MPa
                                                                        p_norm=4,
                                                                        debug=True,
                                                                    )
	
    timeTaken = time.time() - startTime
    print(f"Time taken: {timeTaken:.0f} s")
    if not success:
        print(f"Error: {errorMsg}")

    title = f"Pareto: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
    fe_solver.plot_mesh(title = title, save_path = None)

        
    # plot other quantities over the optimized mesh
    fe_solver.plot_deformation()
    fe_solver.plot_vonMisesStress()

    # Plot volume vs compliance history
    plt.figure()
    plt.plot(history['volume'], history['max_stress'], marker='o')
    plt.xlabel('Volume Fraction')
    plt.ylabel('Mximum Stress')
    plt.title('Pareto: Volume vs Maximum Stress History')
    plt.grid(True)
    plt.show(block=False)
	