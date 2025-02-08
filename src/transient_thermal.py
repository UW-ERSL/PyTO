import numpy as np
import mesher
import plots
import linear_solvers as lin_sol
import mat_lib
import bound_cond
import element_stiffness as es
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs

class TransientThermalFEA:
    def __init__(self,
                 mesh,
                 mat_prop: mat_lib.ThermalMaterial,
                 bc: bound_cond.BC,
                 solver: lin_sol.Solvers,
                 T0=0.0, # initial temperature
                 deltaTime = 0.01,
                 **kwargs):

        self.mesh = mesh
        self.initial_temp = T0*np.ones_like(mesh.node_indices[:, 0])
        self.node_idx = jnp.stack((
                        np.kron(mesh.edofMat, np.ones((8, 1))).flatten(),
                        np.kron(mesh.edofMat, np.ones((1, 8))).flatten())
                        ).T.astype(int)
        self.bc = bc
        self.solver = solver
        self.deltaTime = deltaTime
        #self.elem_effective_stiff = elem_stiff + elem_specific_heat*(1.0/self.deltaTime)
        #self.staticThermalFEA = ThermalFEA(mesh, mat_prop, bc, solver, **kwargs)
        # Overide the element stiffness matrix  
        #self.staticThermalFEA.set_element_stiffness( self.elem_effective_stiff)


        elem_stiff = jnp.asarray(es.hex8_stiffness_matrix_thermal(mat_prop, mesh.elem_size))
        elem_stiffness_stacked = jnp.einsum('ij, e -> eij',
                                 elem_stiff,
								np.ones((mesh.num_elems,)) ).flatten(order = 'C')

        self.K_mtrx = jax_sprs.BCOO((elem_stiffness_stacked, self.node_idx),
                                shape=(bc.num_dofs, bc.num_dofs))
        
        elem_specific_heat = jnp.asarray( es.hex8_specific_heat_matrix(mat_prop, mesh.elem_size))
        elem_specific_heat_stacked = jnp.einsum('ij, e -> eij',
                                 elem_specific_heat,
								np.ones((mesh.num_elems,)) ).flatten(order = 'C')

        self.C_mtrx = jax_sprs.BCOO((elem_specific_heat_stacked, self.node_idx),
                                shape=(bc.num_dofs, bc.num_dofs))
    
        self.num_dofs = bc.num_dofs
        # Check CFL condition
        mesh_size = mesh.elem_size[0]  # assuming uniform mesh
       
        diffusivity = mat_prop.thermal_conductivity / (mat_prop.mass_density* mat_prop.specific_heat)
        cfl = diffusivity*deltaTime / (mesh_size**2 )
        
        if cfl > 0.5:
            print(" thermal_conductivity", mat_prop.thermal_conductivity)
            print(" mass_density", mat_prop.mass_density)
            print(" specific_heat", mat_prop.specific_heat)
            print(" diffusivity", diffusivity)
            print(" deltaTime", deltaTime)
            print(" mesh_size", mesh_size)
            print(f"Warning: CFL condition not met. CFL = {cfl:.3f}")
            print("Time step should be reduced for stability")
            input("Press Enter to continue...")
            
    def solve_newmark(self, time_steps: int, heat_flux_func) -> np.ndarray:
        """
        Solves the transient thermal problem using the Newmark method.
        Parameters:
        -----------
        time_steps : int
            The number of time steps for the simulation.
        heat_flux_func : callable
            A function that takes the current time index, delta time, and mesh as input and returns the heat flux applied.
        Returns:
        --------
        np.ndarray
            A 2D array where each column represents the temperature distribution at a given time step.
        Notes:
        ------
        - The method initializes the temperature distribution array `u` with zeros and sets the initial temperature.
        - The stiffness matrix `K` and damping matrix `C` are used to form the matrix `A`.
        - For each time step, the heat flux is calculated and the temperature distribution is updated.
        """

        self.u = np.zeros((self.num_dofs, time_steps))
        self.u[:, 0] = self.initial_temp
        dt = self.deltaTime
        K = self.K_mtrx
        C = self.C_mtrx
        A = K + C/dt
        for timeIndex in range(1, time_steps):
            print(f"Time step {timeIndex} / {time_steps-1}")
            heatFluxApplied = heat_flux_func(timeIndex, self.deltaTime,self.mesh)
            b = self.C_mtrx  @ self.u[:, timeIndex-1]/self.deltaTime +  heatFluxApplied
            self.u[:, timeIndex] = lin_sol.solve(A, b, self.solver, self.bc) # 
            uMin = np.min(self.u[:, timeIndex])
            uMax = np.max(self.u[:, timeIndex])
            #print(f"Max u: {uMax:.3e}, Min u: {uMin:.3e}")
            #plots.plotMesh(self.mesh, None, self.u[:,timeIndex],title=f'Dof = {self.num_dofs}, max u: {uMax:.3e}',show_edges=False,)
        return self.u

    def solve_newmark_generalized(self, time_steps: int, heat_flux_func, beta=0.5, gamma=0.5) -> np.ndarray:
        """Solve using Newmark beta method for thermal problems.
        
        Args:
            time_steps: Number of time steps
            heat_flux_func: Function that returns heat flux at given time
            beta: Newmark beta parameter (0.25 for constant average acceleration)
            gamma: Newmark gamma parameter (0.5 for constant average acceleration)
        
        Returns:
            Array of temperatures over time
        """
        u = np.zeros((self.num_dofs, time_steps))
        v = np.zeros((self.num_dofs, time_steps)) # velocity (time derivative)
        u[:, 0] = self.initial_temp
        
        dt = self.deltaTime
        K = self.K_mtrx
        C = self.C_mtrx
        A = K + (gamma/(beta*dt))*C
        for i in range(1, time_steps):
            print(f"Time step {i} / {time_steps-1}")
            
            # Predict
            u_pred = u[:, i-1] + dt*v[:, i-1] + (0.5 - beta)*dt**2 * (K @ u[:, i-1] + C @ v[:, i-1])
            v_pred = v[:, i-1] + (1 - gamma)*dt * (K @ u[:, i-1] + C @ v[:, i-1])

            # Force term
            f = heat_flux_func(i, dt,self.mesh.elem_size[0])
            b = f + C @ ((1.0/(beta*dt))*u_pred + (1.0/(2*beta))*v_pred)
            
            # Solve
            u[:, i] = lin_sol.solve(A, b, self.solver, self.bc)
            v[:, i] = (1.0/(beta*dt))*(u[:, i] - u[:, i-1]) - (1.0/(2*beta))*v_pred
            uMin = np.min(u[:, i])
            uMax = np.max(u[:, i])
            #print(f"Max T: {uMax:.3e}, Min T: {uMin:.3e}")
            #plots.plotMesh(self.mesh, None, u[:,i], title=f' max u: {uMax:.3e}', show_edges=False)
            
        return u
    
if __name__ == "__main__":
    import thermal_fea as thermal_fea
    import linear_solvers as lin_solv
    import time
    import jax # import jax to enable 64 bit precision
    import time	

    jax.config.update("jax_enable_x64", True)
    
    # See Paper: "Utility of superposition-based finite element ..."  by Moran, at. al., Additive Manuf, 2018
    xStart = 0.0025 # This is the start of the laser
    deltaX = 0.00238 # This is the length of the laser travel
    V = 0.8 # This is the velocity of the laser
    laserSpotSize = 100e-6 # This is the size of the laser spot

    # Simulation parameters
    totalTime = 0.0125   # This is the total time for the simulation
    deltaTime = 48e-6 # This is the time step for the simulation
    nDOFDesired = 30000 # This is the number of degrees of freedom desired for the FE Mesh
    
    nTimeSteps = int(totalTime/deltaTime)+1
    mesh, mat_prop, bc = thermal_fea.createMoranBenchMark(nDOFDesired=nDOFDesired)
    transient_solver = TransientThermalFEA(mesh = mesh,
                              mat_prop = mat_prop,
                              bc = bc,
                              deltaTime = deltaTime,
                              solver = lin_solv.Solvers.PARDISO)
    
    elemSize = mesh.elem_size[0]

    def transientHeatFluxMoranBenchMark(timeStep,dt,mesh):
        def transientHeatFluxMoranBenchMark(timeStep, dt, mesh):
            """
            Calculate the transient heat flux for the Moran benchmark problem.
            See Paper: "Utility of superposition-based finite element ..."  by Moran, at. al., Additive Manuf, 2018

            This function computes the heat flux distribution over a mesh at a given 
            time step for a transient thermal analysis. The heat flux is modeled as 
            a Gaussian distribution centered around a moving laser spot.
            Parameters:
            -----------
            timeStep : int
                The current time step in the simulation.
            dt : float
                The time increment for each time step.
            mesh : Mesh
                The mesh object containing the nodes and their coordinates.
            Returns:
            --------
            q : numpy.ndarray
                An array of heat flux values at each node in the mesh.
            Notes:
            ------
            - The laser spot moves along the x-axis with a velocity `V`.
            - The heat flux is distributed as a Gaussian function around the laser spot.
            - In the paper, a point source is implemented, but here, we use a Gaussian distribution.
            - If no nodes are found within the laser spot radius, a warning is printed.
            """
        q = np.zeros(mesh.num_nodes)
        x = xStart + timeStep*dt*V
        if (x > xStart + deltaX): # laser has stopped 
            return q
        laser_loc = np.array([x,0.002275,0.00]) # travel along x-axis, y is offset from center line
        
        laser_nodes = mesh.get_nodes_within_radius(laser_loc, 6*laserSpotSize)
        if (laser_nodes.size == 0):
            print(f"Warning: No nodes found at location {laser_loc}")
        totalHeat = 1000 # watt
        distances = np.linalg.norm(mesh.node_xyz[laser_nodes] - laser_loc, axis=1)
        sigma = laserSpotSize
        q[laser_nodes] =  np.exp(-0.5 * (distances / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        q[laser_nodes] *= totalHeat / np.sum(q[laser_nodes])
        return q 
    

    start_time = time.time()
    u = transient_solver.solve_newmark(nTimeSteps, transientHeatFluxMoranBenchMark)
    end_time = time.time()
    print(f"elemSize: {elemSize}")
    print(f"timeStep: {deltaTime}")
    print(f"Time taken for simulation: {end_time - start_time:.2f} seconds")
    
    
    import matplotlib.pyplot as plt
    plt.figure()
    for eta in [0, 0.5, 1.0]:
        xLocationOfInterest = xStart + eta * deltaX
        locationOfInterest = np.array([xLocationOfInterest, 0.002275, -0.0003])  # 300 microns below the center line
        nodes = mesh.get_nodes_from_locations(locationOfInterest)
        temperature_history = u[nodes, :]
        if eta == 0:
            plt.plot(np.arange(nTimeSteps) * deltaTime, temperature_history.T, 'r')
        elif eta == 0.5:
            plt.plot(np.arange(nTimeSteps) * deltaTime, temperature_history.T, 'b')
        elif eta == 1.0:
            plt.plot(np.arange(nTimeSteps) * deltaTime, temperature_history.T, 'g')

        max_temp = np.max(temperature_history)
        max_time = np.argmax(temperature_history) * deltaTime
        #plt.plot(max_time, max_temp, 'ro')  # Mark the maximum temperature with a red dot
        #plt.annotate(f'Max: {max_temp:.2f}C', xy=(max_time, max_temp), xytext=(max_time * 1.1, max_temp * 0.8),
                     #arrowprops=dict(facecolor='black', shrink=0.25))

    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature History')
    plt.legend(['left', 'center', 'right'])
    plt.grid(True)
    plt.show()
   

   
   
    
    

