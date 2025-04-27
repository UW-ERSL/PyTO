"""Thermal Finite Element Analysis."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import hex_element_stiffness as elem_stiff
import mat_lib
import bound_cond
import os
import pyvista as pv


script_dir = os.path.dirname(os.path.abspath(__file__))

class ThermalFEA:
  """Linear Thermal Finite Element Analysis using Hex8 elements."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.ThermalMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_thermal(mat_prop, mesh.elem_size))

    self.node_idx = jnp.stack((
                      np.kron(self.mesh.edofMat, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 8))).flatten())
                      ).T.astype(int)


  def solve(self, x: jnp.ndarray = None,) -> jnp.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = jnp.ones((self.mesh.num_elems,))
    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff, x).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

    u =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    self.sol = u.copy()
    return u
  
  def plot_temperature(self):
    # Return if no solution exists yet
    if not hasattr(self, 'sol'):
      return None

    # Create vertices array
    vertices = self.mesh.node_xyz
  
    sol = self.sol.copy()
 


    # Match plotMeshOld exactly
    faceIndex = np.array([[0,4,7,3],
                          [0,1,5,4],
                          [0,3,2,1],
                          [1,2,6,5],
                          [2,3,7,6],
                          [4,5,6,7]], dtype=np.uint32)
    nFacesPerHex = 6
    faces = []
    face_densities = []
    
    for e in range(self.mesh.num_elems):
      if self.mesh.elemPseudoDensity[e] < 0.5:
        continue
      elif (self.mesh.elemPseudoDensity[e] > 0.5 and 
            np.all(self.mesh.elemNeighborsArray[e] > 0) and 
            np.all(self.mesh.elemPseudoDensity[[int(elem) for elem in 
                                      self.mesh.elemNeighborsArray[e]]] > 0.5)):
        continue

      # Add all faces for this element
      for j in range(nFacesPerHex):
        faces.append(self.mesh.elemArray[e,faceIndex[j,:]])
        face_densities.append(self.mesh.elemPseudoDensity[e])

    # Convert to numpy arrays
    faces = np.array(faces)
    face_densities = np.array(face_densities)
    
    if len(faces) == 0:
      print("No faces to plot after filtering")
      return None

    # Create cells array for PyVista
    n_faces = len(faces)
    cells = np.hstack((
                      np.full((n_faces, 1), 4),  # 4 vertices per face
                      faces
                      ))

    pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD

    # Add scalar values
    pv_mesh.point_data['values'] = sol
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    # Create plotter
    save_path = None
    if save_path is  None:
      plotter = pv.Plotter(window_size=(500, 400))
    else:
      plotter = pv.Plotter(off_screen=True)
    
    plotter.add_title(f'Max Temp: {np.max(sol):.4g}', font_size=8)
    # Add mesh to plotter
    nDOF = 3*self.mesh.num_nodes
    plotter.add_mesh(
                    pv_mesh,
                    scalars='values',
                    show_edges=True,
                    cmap='jet',
                    edge_color='black',
                    line_width=1,
                    scalar_bar_args={
                            'title': '',
                            'vertical': True,
                            'position_x': 0.8,
                            'position_y': 0.3,
                            'width': 0.06
                            }
                  )

    # Add coordinate axes widget
    plotter.add_axes(
                    xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                    line_width=2,
                    labels_off=False,  # Show axis labels
                    color='black'
                    )

    # Set camera position for left-bottom-forward view
    view_distance = 2.5 * self.mesh.bbox.diag_length
    offset = 0.2 * view_distance  # Offset for object position
    plotter.camera_position = [
                    (view_distance*0.5, -view_distance*0.3, view_distance),
                    (offset, offset, 0),   # Focus point - right and bottom
                    (0, 0.8, 0.4)]         # Up vector - Y axis up

    # Reset camera and zoom out slightly
    plotter.camera.zoom(0.8)
    
    # Enable anti-aliasing for better quality
    plotter.enable_anti_aliasing()
    
    # Save image if path is provided
    if save_path:
      #plotter.show(screenshot = save_path)
      plotter.screenshot(save_path)
      plotter.close()
    else:
      plotter.show() 
    
    return 

def runDOFTest():
   
    problem = ThermalExamples.ThickPlate
    # Create arrays to store results
    dof_sizes = [100, 200, 400, 800, 1600, 5000,10000]  # Different DOF sizes to test
    dof_sizes = [2000]
    umax_values = []
    timing = []
    solver = lin_solv.Solvers.PARDISO
    
    for nDOFDesired in dof_sizes:
      mesh, mat_prop, bc = getThermalProblem(problem, nDOFDesired=nDOFDesired)
      
      fe_solver = fea.ThermalFEA(mesh=mesh,
                    mat_prop=mat_prop,
                    bc=bc,
                    solver=solver)
      
      startTime = time.time()
      u = np.asarray(fe_solver.solve())
      uMax = np.max(np.abs(u))
      umax_values.append(uMax)
      timing.append(time.time() - startTime )
      print(f'DOF: {fe_solver.mesh.num_nodes}, Max u: {uMax:.3e}')
    
    # Plot DOF vs uMax
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(dof_sizes, umax_values, 'bo-', markerfacecolor='none')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Maximum Temperature')
    plt.grid(True)
    plt.title('Convergence Study: Hexmesh')
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.show()

    plt.figure()
    plt.plot(dof_sizes, timing, 'bo-', markerfacecolor='none')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Timing (secs)')
    plt.grid(True)
    plt.title('Timing Study: Hexmesh')
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.show()

    nDOF = fe_solver.mesh.num_nodes
   
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('-----------------------------')
if __name__ == "__main__":
    import plots
    import hex_thermal_fea as fea
    import linear_solvers as lin_solv
    import jax # import jax to enable 64 bit precision
    import time	
    from hex_thermal_examples import *
    jax.config.update("jax_enable_x64", True)

    problem = ThermalExamples.LBracket
    nDOFDesired = 10000
    umax_values = []
    timing = []
    solver = lin_solv.Solvers.PARDISO
    
    mesh, mat_prop, bc = getThermalProblem(problem, nDOFDesired=nDOFDesired)
    
    fe_solver = fea.ThermalFEA(mesh=mesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)
    
    startTime = time.time()
    u = np.asarray(fe_solver.solve())
    uMax = np.max(np.abs(u))
    umax_values.append(uMax)
    timing.append(time.time() - startTime )
    print(f'DOF: {fe_solver.mesh.num_nodes}, Max u: {uMax:.3e}')
    
    nDOF = fe_solver.mesh.num_nodes
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max u: ', uMax)
    print('-----------------------------')
	
    plots.plotMesh(fe_solver.mesh, None, u,title=f'Dof = {nDOF}, Tmax: {uMax:.3g}',show_edges=True,)
