import numpy as np
import linear_solvers as lin_sol
import mat_lib
import bound_cond
from tet_thermal_fea import tet4_stiffness_matrix_thermal, tet4_specific_heat_matrix
from tet_thermal_examples import createThickPlateThermalProblemTet
import time
import scipy.sparse as sp

class TetTransientThermalFEA:
    def __init__(self,
                 mesh,
                 mat_prop: mat_lib.Material,
                 bc: bound_cond.BC,
                 solver = lin_sol.Solvers.PARDISO,
                 T0=0.0, # initial temperature
                 deltaTime = 0.01,
                 **kwargs):

        self.mesh = mesh
        self.mat_prop = mat_prop
        self.initial_temp = T0*np.ones_like(mesh.node_xyz[:, 0])
        self.edofMat = np.array(self.mesh.elems[:, :4], dtype=int)
        self.node_idx = np.stack((
              np.kron(self.edofMat, np.ones((4, 1))).flatten(),
              np.kron(self.edofMat, np.ones((1, 4))).flatten())
              ).T.astype(int)
        self.bc = bc
        self.solver = solver
        self.deltaTime = deltaTime
   
        self.num_dofs = bc.num_dofs 
        # Check CFL condition
        mesh_size = mesh.elem_size  # assuming uniform mesh
        diffusivity = mat_prop.thermal_conductivity / (mat_prop.mass_density* mat_prop.specific_heat)
        cfl = diffusivity*deltaTime / (mesh_size**2 )
        print(f"cfl: {cfl:.3e}")

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


    def assemble_global_stiffness_matrices(self):
        data = []
        start_time = time.time()
        K = self.mat_prop.thermal_conductivity
        for i in range(self.mesh.num_elems):
            elem_nodes = self.mesh.node_xyz[self.mesh.elems[i]]
            elem_stiff = tet4_stiffness_matrix_thermal(K, elem_nodes)
            data.append(elem_stiff.flatten())

        elem_stiffness_stacked = np.concatenate(data)
        self.K_mtrx = sp.coo_matrix((elem_stiffness_stacked,  (self.node_idx[:, 0], self.node_idx[:, 1])),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
        datasp = []
        for i in range(self.mesh.num_elems):
            elem_specific_heat =  tet4_specific_heat_matrix(self.mat_prop.specific_heat,
                                                                        self.mat_prop.mass_density, 
                                                                        self.mesh.node_xyz[self.edofMat[i, :]])
            datasp.append( elem_specific_heat.flatten())

        elem_specific_heat_stacked = np.concatenate(datasp)
        self.C_mtrx = sp.coo_matrix((elem_specific_heat_stacked,  (self.node_idx[:, 0], self.node_idx[:, 1])),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

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
        self.assemble_global_stiffness_matrices()
        K = self.K_mtrx
        C = self.C_mtrx
        A = K + C/dt
        for timeIndex in range(1,time_steps):
            if (callback is None):
                print(f"Time step {timeIndex} / {time_steps-1}")
            heatFluxApplied = heat_flux_func(timeIndex, self.deltaTime, self.mesh)
            b = self.C_mtrx @ self.u[:, timeIndex-1]/self.deltaTime + heatFluxApplied
            self.u[:, timeIndex] = lin_sol.solve(A, b, self.solver, self.bc)
            if callback:
                callback(timeIndex, self.u[:, timeIndex])
                
        return self.u

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from tet_transient_thermal_examples import TetTransientThermalExamples, getTetTransientThermalProblem

    nDOFDesired = 50000
    problem = TetTransientThermalExamples.ThickPlate
    tetmesh, mat_prop, bc, initialTemperature, totalTime,timeStep,transientHeatFunction,ptsOfInterest = getTetTransientThermalProblem(problem, nDOFDesired=nDOFDesired)
    nTimeSteps = int(totalTime/timeStep)+1
   
    transient_solver = TetTransientThermalFEA(tetmesh, mat_prop, bc, T0 = initialTemperature,deltaTime=timeStep)
    start_time = time.time()
    u = transient_solver.solve_newmark(nTimeSteps, transientHeatFunction)
    end_time = time.time()
 
    print(f"Time taken for simulation: {end_time - start_time:.2f} seconds")
    nodes = tetmesh.get_nodes_from_locations(ptsOfInterest)
    temperature_history = u[nodes, :]
    plt.figure()
    plt.plot(np.arange(nTimeSteps) * timeStep, temperature_history.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature History')
    plt.legend([f'Node {i+1}' for i in range(len(nodes))])
    plt.grid(True)
    plt.show()
   

   
   
    
    

