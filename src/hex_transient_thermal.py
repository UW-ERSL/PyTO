import numpy as np
import linear_solvers
import mat_lib
import bound_cond
import hex_element_stiffness as es
import matplotlib.pyplot as plt
import scipy.sparse as sp

class HexTransientThermalFEA:
    def __init__(self,
                 mesh,
                 mat_prop: mat_lib.ThermalMaterial,
                 bc: bound_cond.BC,
                 solver: linear_solvers.Solvers,
                 T0=0.0, # initial temperature
                 deltaTime = 0.01,
                 **kwargs):

        self.mesh = mesh
        self.initial_temp = T0*np.ones_like(mesh.node_indices[:, 0])
     
        self.edofMat = np.array(self.mesh.elemArray[:, :8], dtype=int)

        self.node_idx = np.stack((
                        np.kron(self.edofMat, np.ones((8, 1))).flatten(),
                        np.kron(self.edofMat, np.ones((1, 8))).flatten())
                        ).T.astype(int)
        self.bc = bc
        self.solver = solver
        self.deltaTime = deltaTime

        elem_stiff = np.asarray(es.hex8_stiffness_matrix_thermal(mat_prop, mesh.elem_size))

        x = np.ones((self.mesh.num_elems,))
        elem_stiffness_stacked = np.einsum('ij, e -> eij',
                                 elem_stiff, x).flatten(order = 'C')

    
        self.K_mtrx = sp.coo_matrix((elem_stiffness_stacked, (self.node_idx[:, 0], self.node_idx[:, 1])),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
        
        elem_specific_heat = np.asarray( es.hex8_specific_heat_matrix(mat_prop, mesh.elem_size))
        elem_specific_heat_stacked = np.einsum('ij, e -> eij',
                                 elem_specific_heat,
								np.ones((mesh.num_elems,)) ).flatten(order = 'C')

        self.C_mtrx = sp.coo_matrix((elem_specific_heat_stacked,  (self.node_idx[:, 0], self.node_idx[:, 1])),
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
            
    def solve_newmark(self, time_steps: int, heat_flux_func, callback=None) -> np.ndarray:
        """
        Solves the transient thermal problem using the Newmark method.
        Parameters:
        -----------
        time_steps : int
            The number of time steps for the simulation.
        heat_flux_func : callable
            A function that takes the current time index, delta time, and mesh as input and returns the heat flux applied.
        callback : callable, optional
            A function that is called at each time step with the current time index and temperature distribution.
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
        for timeIndex in range(time_steps):
            print(f"Time step {timeIndex}/{time_steps}")
            heatFluxApplied = heat_flux_func(timeIndex, self.deltaTime, self.mesh)
            if timeIndex == 0:
                b = heatFluxApplied
            else:
                b = self.C_mtrx @ self.u[:, timeIndex-1]/self.deltaTime + heatFluxApplied
            self.u[:, timeIndex] = linear_solvers.solve(A, b, self.solver, self.bc)
            
            if callback:
                callback(timeIndex, self.u[:, timeIndex])
                
        return self.u


    
if __name__ == "__main__":
    from hex_thermal_examples import HexThermalExamples, getThermalProblem
    import linear_solvers as lin_solv
    import time

    from hex_transient_thermal_examples import HexTransientThermalExamples,getHexTransientThermalProblem 


    nDOFDesired = 50000
    problem = HexTransientThermalExamples.ThickPlate
    hexmesh, mat_prop, bc, initialTemperature, totalTime,timeStep,transientHeatFunction,ptsOfInterest = getHexTransientThermalProblem(problem, nDOFDesired=nDOFDesired)
    nTimeSteps = int(totalTime/timeStep)+1
   
    transient_solver = HexTransientThermalFEA(hexmesh, mat_prop, bc,linear_solvers.Solvers.PARDISO,T0 = initialTemperature,deltaTime=timeStep)
    start_time = time.time()
    u = transient_solver.solve_newmark(nTimeSteps, transientHeatFunction)
    end_time = time.time()
 
    print(f"Time taken for simulation: {end_time - start_time:.2f} seconds")
    nodes = hexmesh.get_nodes_from_locations(ptsOfInterest)
    temperature_history = u[nodes, :]
    plt.figure()
    plt.plot(np.arange(nTimeSteps) * timeStep, temperature_history.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature History')
    plt.legend([f'Node {i+1}' for i in range(len(nodes))])
    plt.grid(True)
    plt.show()
   

   
   
    
    
