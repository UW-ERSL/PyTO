"""Structural Finite Element Analysis."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import linear_solvers as lin_sol
import element_stiffness as elem_stiff
import mat_lib
import bound_cond

import struct_fea as fea
import linear_solvers as lin_solv
import mat_lib
import os
import deflation 
import pyvista as pv
script_dir = os.path.dirname(os.path.abspath(__file__))


class StructFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
         mesh,
         mat_prop: mat_lib.StructuralMaterial | list[mat_lib.StructuralMaterial],
         bc: bound_cond.BC,
         solver: lin_sol.Solvers,
         elem_body_force: jnp.ndarray = None,
         **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs

    # Handle single material or list of materials
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_structural(mp, mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = jnp.stack(elem_stiff_list)
    else:
      self.elem_stiff = jnp.expand_dims(
          elem_stiff.hex8_stiffness_matrix_structural(mat_prop, mesh.elem_size), axis=0)

   
    self.node_idx = jnp.stack((
            np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
            np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
            ).T.astype(int)
    self.elem_body_force = elem_body_force

#################################################################
  def solve(self,
            x: jnp.ndarray = None,
            elasticity_material_model: dict = None) -> jnp.ndarray:
    """Solve the structural finite element problem.

    Args:
      x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = jnp.ones((self.mesh.num_elems,))

    if elasticity_material_model is None:
      elem_material_scaling = x**3 # default to SIMP with penal = 3 if nothing is provided

    elif elasticity_material_model['name'] == 'SIMP':
      penal = elasticity_material_model['penal']
      elem_material_scaling = x**penal
    elif elasticity_material_model['name'] == 'SIMPPLUS':
      #  model from the paper here: https://doi.org/10.1002/nme.2499 
      alpha = elasticity_material_model['alpha']
      penal = elasticity_material_model['penal']
      elem_material_scaling = (alpha-1)/alpha * x ** penal + (1/alpha) * x
    elif elasticity_material_model['name'] == 'GRIP':# Generalized Rational Interpolation with Penalization
      penal = elasticity_material_model['penal']
      elem_material_scaling = x/((2-x)**penal)

  
    # Handle different shapes of elem_stiff
    if self.elem_stiff.shape[0] == 1:
      # Single material case (1,N,N)
      elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                    self.elem_stiff[0],
                    elem_material_scaling).flatten(order = 'C')
    else:
      # Multiple materials case (M,N,N)
      # Assuming elem_mat_id contains material ID (0 to M-1) for each element
      # Randomly assign material IDs (0 or 1) to each element
      
      elem_stiff_mtrx = jnp.einsum('mij, e, em -> eij',
                    self.elem_stiff,
                    elem_material_scaling,
                    jnp.eye(self.elem_stiff.shape[0])[self.mesh.elemComponentId]).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                   shape=(self.bc.num_dofs, self.bc.num_dofs))
    self.total_force = self.bc.force.copy()
    if self.elem_body_force is not None:
      elem_force = self.elem_body_force.copy()
      if elasticity_material_model is None:
        masspenal = 1
      else:
        masspenal = elasticity_material_model['masspenal']
      
      for i in range(3):
        elem_force[i::3]  *= (x**masspenal)
        
      node_forces = np.zeros((self.mesh.num_nodes * 3,))
      node_forces[0::3] = self.mesh.elem_to_node_field_mapping* elem_force[0::3] 
      node_forces[1::3] = self.mesh.elem_to_node_field_mapping* elem_force[1::3] 
      node_forces[2::3] = self.mesh.elem_to_node_field_mapping* elem_force[2::3] 
      self.total_force += node_forces
    sol =  lin_sol.solve(stiff_mtrx,
                      self.total_force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    self.sol = sol
    self.deformation = jnp.sqrt(sol[0::3]**2 + sol[1::3]**2 + sol[2::3]**2)
    self.max_deformation = jnp.max(self.deformation)
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
      uGrad = gradN @ self.sol[edof[:, ::3]].T
      vGrad = gradN @ self.sol[edof[:, 1::3]].T
      wGrad = gradN @ self.sol[edof[:, 2::3]].T
      
      # Compute Engineering strains
      strain = np.stack([
        uGrad[0], vGrad[1], wGrad[2],
        uGrad[1] + vGrad[0],
        uGrad[2] + wGrad[0],
        vGrad[2] + wGrad[1]
      ], axis=1)  # Shape: (num_elems, 6)

      # Constitutive matrix D for each material
      if isinstance(self.mat_prop, list):
        # Create D matrix for each material
        D_list = []
        for mp in self.mat_prop:
          E = mp.youngs_modulus
          nu = mp.poissons_ratio
          D = E / ((1 + nu) * (1 - 2*nu)) * jnp.array([
        [1-nu, nu, nu, 0, 0, 0],
        [nu, 1-nu, nu, 0, 0, 0],
        [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0],
        [0, 0, 0, 0, (1-2*nu)/2, 0],
        [0, 0, 0, 0, 0, (1-2*nu)/2]
          ])
          D_list.append(D)
        D_stack = jnp.stack(D_list)
        # Use elem_mat_id to select correct D matrix for each element
        element_stress = jnp.einsum('mij,ej,em->ei', D_stack, strain, 
                  jnp.eye(len(self.mat_prop))[self.mesh.elemComponentId])
      else:
        # Single material case
        E = self.mat_prop.youngs_modulus 
        nu = self.mat_prop.poissons_ratio
        D = E / ((1 + nu) * (1 - 2*nu)) * jnp.array([
          [1-nu, nu, nu, 0, 0, 0],
          [nu, 1-nu, nu, 0, 0, 0],
          [nu, nu, 1-nu, 0, 0, 0],
          [0, 0, 0, (1-2*nu)/2, 0, 0],
          [0, 0, 0, 0, (1-2*nu)/2, 0],
          [0, 0, 0, 0, 0, (1-2*nu)/2]
        ])
        element_stress = jnp.einsum('ij,ej->ei', D, strain)
      self.strainComponents = strain
      self.stressComponents = element_stress
      self.vonMisesStress = jnp.sqrt(0.5*((element_stress[:,0]-element_stress[:,1])**2 +
                (element_stress[:,1]-element_stress[:,2])**2 +
                (element_stress[:,2]-element_stress[:,0])**2) +
                3*(element_stress[:,3]**2 + element_stress[:,4]**2 +
                   element_stress[:,5]**2))
      return 
#################################################################
  def plot_mesh(self, title = None,plot_bc = True, save_path=None):
    
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
    cells = np.hstack((
                      np.full((n_faces, 1), 4),  # 4 vertices per face
                      faces
                      ))

    pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    # Create plotter
    save_path = None
    if save_path is  None:
      plotter = pv.Plotter(window_size=(500, 400))
    else:
      plotter = pv.Plotter(off_screen=True)
    
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
################################################################# 
  def plot_deformation(self):
    # Return if no solution exists yet
    if not hasattr(self, 'sol'):
      return None

    # Create vertices array
    vertices = self.mesh.node_xyz
  
    sol = self.sol.copy()
    sol = sol.reshape((-1, 3))
    delta = self.deformation
    deltaMax = self.max_deformation
    scale = float(0.1*self.mesh.bbox.diag_length/deltaMax)
    vertices += scale*sol


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
    pv_mesh.point_data['values'] = delta
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    # Create plotter
    save_path = None
    if save_path is  None:
      plotter = pv.Plotter(window_size=(500, 400))
    else:
      plotter = pv.Plotter(off_screen=True)
    
    plotter.add_title(f'Deformation scale: {scale:.2g}', font_size=8)
    # Add mesh to plotter
    nDOF = 3*self.mesh.num_nodes
    plotter.add_mesh(
                    pv_mesh,
                    scalars='values' if delta is not None else 'density',
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

#################################################################
  def plot_elem_field(self,
            elem_field,
            mask_low_pseudodensity = True,
            title = '',
            save_path=None,
            fontsize=10):
    """Plot element field on the mesh.
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
      plotter = pv.Plotter(window_size=(500, 400))
    else:
      plotter = pv.Plotter( off_screen=True)

    # Add mesh to plotter
    plotter.add_mesh(
            pv_mesh,
            scalars='field',
            cmap='jet',
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
      plotter.screenshot(save_path)
      plotter.close()
    else:
      plotter.show()

    return 

#################################################################
  def plot_vonMisesStress(self,
            save_path=None,
            fontsize=10):
    self.plot_elem_field(self.vonMisesStress, title = f'vonMises stress; max: {np.max(self.vonMisesStress):.2e} ',
                          save_path=save_path, fontsize=fontsize)

#################################################################    
  def plot_strain_component(self,strainComponent = 0,
            save_path=None,
            fontsize=10):
    self.plot_elem_field(self.strainComponents[:,strainComponent], title = f'Strain component: {strainComponent} ',
                          save_path=save_path, fontsize=fontsize)


#################################################################
  def plot_stress_component(self,stressComponent = 0,
            save_path=None,
            fontsize=10):

    self.plot_elem_field(self.stressComponents[:,stressComponent], title = f'Stress component: {stressComponent} ',
                          save_path=save_path, fontsize=fontsize)

    
#################################################################
if __name__ == "__main__":    
  jax.config.update("jax_enable_x64", True)
  from examples_structural import StructuralExamples,getStructuralProblem

  problem = StructuralExamples.TorsionBar
  nDOFDesired = 5000
  mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
  solver = lin_solv.Solvers.PARDISO # typically DPCG or PARDISO
  
  dsolver = deflation.DeflationSolver()
  startTime = time.time()
  if (solver == lin_solv.Solvers.DPCG):
    nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
    dsolver.create_deflation_groups(mesh, nGroups)
    dsolver.create_delfation_matrix(mesh)
    dsolver.W = dsolver.W[bc.free_dofs, :]
  
  fe_solver = fea.StructFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = solver,
        dsolver = dsolver,
        rtol = 1e-8,
        elem_body_force = elem_body_force)

  fe_solver.plot_mesh()
  u = np.asarray(fe_solver.solve())
  fe_solver.postprocess()
  fe_solver.plot_deformation()
  fe_solver.plot_vonMisesStress()
  fe_solver.plot_stress_component(0)
  
  