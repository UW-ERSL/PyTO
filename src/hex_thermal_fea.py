"""Thermal Finite Element Analysis."""

import time
import numpy as np
import linear_solvers as lin_sol
import hex_element_stiffness as elem_stiff
import mat_lib
import bound_cond
import os
import pyvista as pv
import scipy.sparse as sp
import deflation
from topopt_material_model import *
from hex_thermal_examples import HexThermalExamples
script_dir = os.path.dirname(os.path.abspath(__file__))

class HexThermalFEA:
  """Linear Thermal Finite Element Analysis using Hex8 elements."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.Material,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
               dsolver: deflation.DeflationSolver = None,
                elem_body_force: np.ndarray = None,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    # Handle single material or list of materials
    if isinstance(mat_prop, list):
      # Create element stiffness matrix for each material
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_thermal(mp.thermal_conductivity, mesh.elem_size)
                         for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)  # shape: (num_materials, 8, 8)
    else:
      self.elem_stiff = np.expand_dims(
        elem_stiff.hex8_stiffness_matrix_thermal(mat_prop.thermal_conductivity, mesh.elem_size), axis=0)
      

    self.node_idx = np.stack((
                      np.kron(self.mesh.edofMat, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 8))).flatten())
                      ).T.astype(int)
    self.elem_body_force = elem_body_force

        #default camera position
    view_distance = 2.5 * self.mesh.bbox.diag_length
    offset = 0.2 * view_distance  # Offset for object position
    self.camera_position =  [
                    (view_distance*0.5, -view_distance*0.3, view_distance),
                    (offset, offset, 0),   # Focus point - right and bottom
                    (0, 0.8, 0.4)]         # Up vector - Y axis up
    self.create_pyvista_plotter()

#################################################################
  def create_pyvista_plotter(self):
    self.pyVistaPlotter = pv.Plotter(window_size=(500, 400))
    self.pyVistaPlotter.camera_position =self.camera_position
    # Enable anti-aliasing for better quality
    self.pyVistaPlotter.enable_anti_aliasing()
##################################################################
  def set_thermal_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
    """
    Set or update the thermal material(s) and recompute element stiffness matrices.
    Args:
      mat_prop: Single Material or list of Materials.
    """
    self.mat_prop = mat_prop
    if isinstance(mat_prop, list):
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_thermal(mp.thermal_conductivity, self.mesh.elem_size)
                         for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
        elem_stiff.hex8_stiffness_matrix_thermal(mat_prop.thermal_conductivity, self.mesh.elem_size), axis=0)   
#################################################################
  def solve(self, x: np.ndarray = None,material_model: MaterialModel = None, elem_mat_id: np.ndarray = None) -> np.ndarray:
    """Solve the thermal finite element problem.

    Args:
       x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = np.ones((self.mesh.num_elems,))

    elem_material_scaling = get_thermal_material_model_scaling(x, material_model)

     # Multi-material support
    if self.elem_stiff.shape[0] == 1 or elem_mat_id is None:
        # Single material case
        elem_stiff_mtrx = np.einsum('ij, e -> eij',
                                    self.elem_stiff[0], elem_material_scaling).flatten(order='C')
    else:
        # Multi-material case: select correct stiffness for each element
        # elem_mat_id: array of shape (num_elems,) with values in [0, num_materials-1]
        elem_stiff_mtrx = np.zeros((self.mesh.num_elems, 8, 8))
        for e in range(self.mesh.num_elems):
          mat_idx = elem_mat_id[e]
          elem_stiff_mtrx[e] = self.elem_stiff[mat_idx] * elem_material_scaling[e]
        elem_stiff_mtrx = elem_stiff_mtrx.flatten(order='C')


    stiff_mtrx = sp.coo_matrix((elem_stiff_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

    sol =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    self.sol = sol.copy()
    return sol
  
  #################################################################
  def postprocess(self):
      """Computes the stresses at the center of each element.

      Args:
          u: Displacement field (num_dofs,).

      Returns:
          Array of (num_elems, 6) of the stresses at the center of each element.
          The order of the stress components is:
          sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy
      """
   
      gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
      ])
      
      # Get element degrees of freedom
      edof = self.mesh.edofMat
      
      # Compute displacement gradients
      self.strain = gradN @ self.sol[edof].T


  #################################################################
  def plot_temperature(self,auto_close = True,plotter=None, save_path=None):
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
    externalPlotter = False # assume that a plotter is provided
    if plotter is None: 
      externalPlotter = True # create a new plotter
      # Create plotter
      save_path = None
      if save_path is  None:
        plotter = pv.Plotter(window_size=(500, 400))
      else:
        plotter = pv.Plotter(off_screen=True)
      
      plotter.add_title(f'Max Temp: {np.max(sol):.4g}', font_size=8)
          # Add coordinate axes widget
      plotter.add_axes(
                      xlabel='X',
                      ylabel='Y',
                      zlabel='Z',
                      line_width=2,
                      labels_off=False,  # Show axis labels
                      color='black'
                      )
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
      if (externalPlotter):
        plotter.show(interactive_update=not auto_close, auto_close=auto_close)
      else:
        plotter.show()

    return

#################################################################
  def plot_mesh(self, title = None,plot_bc = True,auto_close = True, save_path=None):
    
    self.pyVistaPlotter.clear()
    if (title is None):
      title = f'DOF: {3*self.mesh.num_nodes}'

    vertices = self.mesh.node_xyz
    # We only plot the boundary faces to save on memory
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
    cells = np.hstack(( np.full((n_faces, 1), 4),  # 4 vertices per face
                      faces))

    pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    # Create plotter

    if save_path is  None:
      plotter = self.pyVistaPlotter 
      if plotter.iren is None:
        self.create_pyvista_plotter()
        plotter = self.pyVistaPlotter 
        plotter.show(interactive_update=True, auto_close=False)
    else:
      plotter = pv.Plotter(off_screen=True) # for saving images
      plotter.camera_position =self.camera_position
      plotter.enable_anti_aliasing()
    
    plotter.add_title(title, font_size=8)
  
    plotter.add_mesh(
                    pv_mesh,
                    color='lightgreen',
                    show_edges=True,
                    edge_color='black',
                    line_width=1
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
    
    if (plot_bc):
      # Add dots and force arrows for labeled nodes
      point_size = 10  # Size of dots in pixels

      # Add black dots for label 1 (fixed nodes)
      label1_nodes = np.where(self.mesh.node_indices[:, 3] == 1)[0]
      if len(label1_nodes) > 0 and self.bc is not None:
        points1 = vertices[label1_nodes]
        dots1 = pv.PolyData(points1)
        plotter.add_points(dots1,
                          color='black',
                          point_size=point_size,
                          render_points_as_spheres=True)

      # Add force arrows for label 2 (without red dots)
      label2_nodes = np.where(self.mesh.node_indices[:, 3] == 2)[0]
      if len(label2_nodes) > 0  and self.bc is not None: #structural
        # Add force arrows
        arrow_scale = 0.1 * self.mesh.bbox.diag_length
        for node in label2_nodes:
          # Get force components for this node
          fx = self.bc.force[3*node]
          fy = self.bc.force[3*node + 1]
          fz = self.bc.force[3*node + 2]
          force_vec = np.array([fx, fy, fz])
          
          # Only add arrow if force is non-zero
          if np.linalg.norm(force_vec) > 0:
            # Normalize and scale force vector
            force_vec = force_vec / np.linalg.norm(force_vec) * arrow_scale
            
            # Create arrow
            start_point = vertices[node]
      
            # Add arrow to plot
            arrow = pv.Arrow(start = start_point,
                            direction = force_vec,
                            scale = arrow_scale)
            plotter.add_mesh(arrow, color='red')
    
    # Save image if path is provided
    if save_path:
      plotter.screenshot(save_path)
      plotter.close()
    else:
      plotter.show(interactive_update=not auto_close, auto_close=auto_close) 
    self.camera_position = plotter.camera_position # For all future displays
    return

#################################################################
  def plot_elem_field(self,
            elem_field,
            mask_low_pseudodensity = True,
            title = '',
            save_path=None,
            colormap = 'jet',
            auto_close = True,
            fontsize=10,
            cross_section=None):  # New parameter for cross-section
    """Plot element field on the mesh.
    
    Args:
        elem_field: Field values for each element
        mask_low_pseudodensity: Whether to filter out elements with low density
        title: Plot title
        save_path: Path to save the image
        colormap: Color map for the field visualization
        auto_close: Whether to auto-close the plotter
        fontsize: Font size for the title
        cross_section: Dict with keys 'axis' ('x', 'y', or 'z') and 'position' (float)
                      to create a cross-section view, or None for full view
    """
    # Filter elements based on pseudo_density
    if (mask_low_pseudodensity):
      mask = self.mesh.elemPseudoDensity > 0.5
      mask = self.mesh.elemPseudoDensity > 0.5
      filtered_elems = self.mesh.elemArray[mask]
      filtered_field = elem_field[mask]

    else:
      filtered_elems = self.mesh.elemArray
      filtered_field = elem_field

    if len(filtered_elems) == 0:
        print("No elements to plot after filtering")
        return None

    # Create vertices array
    vertices = self.mesh.node_xyz

    # Create cells array for PyVista
    cells = np.hstack((
              np.full((len(filtered_elems), 1), 8),  # 8 vertices per hexahedron
              filtered_elems
            ))

    # Create PyVista mesh
    pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

    # Add field data to cell data
    pv_mesh.cell_data['field'] = filtered_field

    # Create plotter
    if save_path is None:
      plotter = self.pyVistaPlotter 
      if plotter.iren is None:
        self.create_pyvista_plotter()
        plotter = self.pyVistaPlotter

    else:
      plotter = pv.Plotter(off_screen=True)

    # If cross-section is specified, create a clipped mesh
    if cross_section is not None:
        axis = cross_section.get('axis', 'x').lower()
        rel_position = cross_section.get('position', 0.0)
        # Convert relative position to actual position based on bounding box
        bbox_min = pv_mesh.bounds[::2]  # [xmin, ymin, zmin]
        bbox_max = pv_mesh.bounds[1::2]  # [xmax, ymax, zmax]

  
        # Create a clipping plane based on the axis
        if axis == 'x':
            normal = (1, 0, 0)
            position = bbox_min[0] + (bbox_max[0] - bbox_min[0]) * (rel_position + 1) / 2
            origin = (position, 0, 0)
        elif axis == 'y':
            normal = (0, 1, 0)
            position = bbox_min[1] + (bbox_max[1] - bbox_min[1]) * (rel_position + 1) / 2
            origin = (0, position, 0)
        elif axis == 'z':
            normal = (0, 0, 1)
            position = bbox_min[2] + (bbox_max[2] - bbox_min[2]) * (rel_position + 1) / 2
            origin = (0, 0, position)
        else:
            print(f"Invalid axis '{axis}'. Using 'x' instead.")
            normal = (1, 0, 0)
            position = 0
            origin = (position, 0, 0)
        
        # Apply clipping
        pv_mesh = pv_mesh.clip(normal=normal, origin=origin)
        
        # Update title to indicate cross-section
        if title:
            title += f" (Cross-section {axis}={position})"
        else:
            title = f"Cross-section {axis}={position}"

    # Add mesh to plotter
    plotter.add_mesh(
            pv_mesh,
            scalars='field',
            cmap=colormap,
            show_edges=True,
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

    # Add title
    plotter.add_title(title, font_size=8)

    # Add coordinate axes widget
    plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False,  # Show axis labels
            color='black'
            )
    # Save image if path is provided
    if save_path:
      plotter.screenshot(save_path)
      plotter.close()
    else:
      plotter.show(interactive_update=not auto_close, auto_close=auto_close)
    self.camera_position = plotter.camera_position # For all future displays
    return 

#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            fontsize=10):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, fontsize=fontsize)
#################################################################   
def runDOFTest():
    from hex_thermal_examples import getThermalProblem, ThermalExamples
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

#################################################################
if __name__ == "__main__":
    import hex_thermal_fea as fea
    import linear_solvers as lin_solv
    import time	
    from hex_thermal_examples import *

    problem = HexThermalExamples.BridgeThermal
    nDOFDesired = 25000

   
    solver = lin_solv.Solvers.PARDISO
    mesh, mat_prop, bc, elem_body_force = getThermalProblem(problem, nDOFDesired=nDOFDesired)
    
    fe_solver = HexThermalFEA(mesh=mesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)
    
    startTime = time.time()
    T = fe_solver.solve()
    TMax = np.max(np.abs(T ))


    print(f'DOF: {fe_solver.mesh.num_nodes}, Max u: {TMax:.3e}')
    
    nDOF = fe_solver.mesh.num_nodes
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max Temp: ', TMax)
    print('-----------------------------')
	
    fe_solver.plot_temperature()
