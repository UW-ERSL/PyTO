"""Structural Finite Element Analysis."""

from topopt_material_model import *
import time
import numpy as np
import os
import pyvista as pv
import mat_lib
import bound_cond
import linear_solvers
import hex_element_stiffness
import deflation
import scipy.sparse as sp


script_dir = os.path.dirname(os.path.abspath(__file__))

class HexStructuralFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
         mesh,
         mat_prop: mat_lib.Material | list[mat_lib.Material],
         bc: bound_cond.BC,
         solver: linear_solvers.Solvers,
         dsolver: deflation.DeflationSolver = None,
         elem_body_force: np.ndarray = None,
         **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.dsolver = dsolver

    # Handle single material or list of materials
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(mp.youngs_modulus, mp.poissons_ratio, mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
          hex_element_stiffness.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus,mat_prop.poissons_ratio, mesh.elem_size), axis=0)

   
    self.node_idx = np.stack((
            np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
            np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
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
    
#################################################################
  def set_structural_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
    
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(mp.youngs_modulus,mp.poissons_ratio, self.mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
          hex_element_stiffness.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus,mat_prop.poissons_ratio, self.mesh.elem_size), axis=0)

  #################################################################
  def solve(self,x: np.ndarray = None,
            material_model: MaterialModel = None) -> np.ndarray:
    """Solve the structural finite element problem.

    Args:
      x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = np.ones((self.mesh.num_elems,))


    elem_material_scaling = get_structural_material_model_scaling(x, material_model)
    # Handle different shapes of elem_stiff
    if self.elem_stiff.shape[0] == 1:
      # Single material case (1,N,N)
      elem_stiff_mtrx = np.einsum('ij, e -> eij',
                    self.elem_stiff[0],
                    elem_material_scaling).flatten(order = 'C')
    else:
      # Multiple materials case (M,N,N)
      elem_stiff_mtrx = np.einsum('mij, m -> mij',
                    self.elem_stiff,
                    elem_material_scaling).flatten(order = 'C')

    
    self.stiff_mtrx = sp.coo_matrix((elem_stiff_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
                   shape=(self.bc.num_dofs, self.bc.num_dofs))
    self.total_force = self.bc.force.copy()
    if self.elem_body_force is not None:
      elem_force = self.elem_body_force.copy()
      if material_model is None:
        masspenal = 1
      else:
        masspenal = 1
      
      for i in range(3):
        elem_force[i::3]  *= (x**masspenal)
        
      node_forces = np.zeros((self.mesh.num_nodes * 3,))
      node_forces[0::3] = self.mesh.elem_to_node_field_mapping* elem_force[0::3] 
      node_forces[1::3] = self.mesh.elem_to_node_field_mapping* elem_force[1::3] 
      node_forces[2::3] = self.mesh.elem_to_node_field_mapping* elem_force[2::3] 
      self.total_force += node_forces

    
    if hasattr(self.mesh, 'externalSprings') and self.mesh.externalSprings is not None:
      # Add spring stiffnesses to diagonal terms
      # external_springs = [
      # (1000.0, 42),  # Add spring with stiffness 1000 at DOF 42
      #  (2000.0, 156), # Add spring with stiffness 2000 at DOF 156
      #  (500.0, 789)   # Add spring with stiffness 500 at DOF 789
      #]
      # Convert to CSR format for modification
      self.stiff_mtrx  = self.stiff_mtrx.tocsr()
      for KSpring, dof in self.mesh.externalSprings:
          self.stiff_mtrx [dof,dof] += KSpring

    sol =  linear_solvers.solve(self.stiff_mtrx,
                      self.total_force,
                      self.solver,
                      self.bc,
                      dsolver = self.dsolver,
                      **self.kwargs)
    self.sol = sol
    self.deformation = np.sqrt(sol[0::3]**2 + sol[1::3]**2 + sol[2::3]**2)
    self.max_deformation = np.max(self.deformation)
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
          D = E / ((1 + nu) * (1 - 2*nu)) * np.array([
        [1-nu, nu, nu, 0, 0, 0],
        [nu, 1-nu, nu, 0, 0, 0],
        [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0],
        [0, 0, 0, 0, (1-2*nu)/2, 0],
        [0, 0, 0, 0, 0, (1-2*nu)/2]
          ])
          D_list.append(D)
        D_stack = np.stack(D_list)
        # Use elem_mat_id to select correct D matrix for each element
        element_stress = np.einsum('mij,ej,em->ei', D_stack, strain, 
                  np.eye(len(self.mat_prop))[self.mesh.elemComponentId])
      else:
        # Single material case
        E = self.mat_prop.youngs_modulus 
        nu = self.mat_prop.poissons_ratio
        D = E / ((1 + nu) * (1 - 2*nu)) * np.array([
          [1-nu, nu, nu, 0, 0, 0],
          [nu, 1-nu, nu, 0, 0, 0],
          [nu, nu, 1-nu, 0, 0, 0],
          [0, 0, 0, (1-2*nu)/2, 0, 0],
          [0, 0, 0, 0, (1-2*nu)/2, 0],
          [0, 0, 0, 0, 0, (1-2*nu)/2]
        ])
        element_stress = np.einsum('ij,ej->ei', D, strain)
      self.strainComponents = strain
      self.stressComponents = element_stress
      self.vonMisesStress = np.sqrt(0.5*((element_stress[:,0]-element_stress[:,1])**2 +
                (element_stress[:,1]-element_stress[:,2])**2 +
                (element_stress[:,2]-element_stress[:,0])**2) +
                3*(element_stress[:,3]**2 + element_stress[:,4]**2 +
                   element_stress[:,5]**2))
      self.elemStrainEnergy = 0.5 * np.sum(strain * element_stress, axis=1)  # Element-wise strain energy
      return 
#################################################################
  def plot_mesh(self, title = None,plot_bc = True,rel_arrow_scale = 0.5, 
                auto_close = True, save_path=None,offsetArrow = False,transparency = 1.0):
    
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
            line_width=1,
            opacity=transparency  # Add transparency (0.0 is fully transparent, 1.0 is fully opaque)
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
        
        force_norm_avg = np.linalg.norm(self.bc.force)
        for node in label2_nodes:
          # Get force components for this node
          fx = self.bc.force[3*node]
          fy = self.bc.force[3*node + 1]
          fz = self.bc.force[3*node + 2]
          force_vec = np.array([fx, fy, fz])
          
          force_nrm = np.linalg.norm(force_vec)
          # Only add arrow if force is non-zero
          if force_nrm > 0:
            arrow_scale = rel_arrow_scale * self.mesh.bbox.diag_length*force_nrm/force_norm_avg
            # Normalize and scale force vector
            force_vec_dir = force_vec / force_nrm
            
            # Create arrow
            start_point = vertices[node]
            if (offsetArrow):
              start_point = start_point -  force_vec_dir*arrow_scale  # Offset start point by arrow length so that it is visible
            # Add arrow to plot
            arrow = pv.Arrow(start = start_point,
                            direction = force_vec_dir,
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
  def plot_deformation(self,auto_close = True, save_path=None):
    """Plot the deformed mesh with the deformation scaled by a factor."""
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
      plotter = self.pyVistaPlotter 
      if plotter.iren is None:
        self.create_pyvista_plotter()
        plotter = self.pyVistaPlotter 
        plotter.show(interactive_update=True, auto_close=False)
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


    # Save image if path is provided
    if save_path:
      #plotter.show(screenshot = save_path)
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
  def plot_vonMisesStress(self,
            save_path=None,
            fontsize=8):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.vonMisesStress, title = f'vonMises stress; max: {np.max(self.vonMisesStress):.2e} ',
                          save_path=save_path, fontsize=fontsize)

#################################################################    
  def plot_strain_component(self,strainComponent = 0,
            save_path=None,
            fontsize=8):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.strainComponents[:,strainComponent], title = f'Strain component: {strainComponent} ',
                          save_path=save_path,fontsize=fontsize)
#################################################################
  def plot_stress_component(self,stressComponent = 0,
            save_path=None,
            fontsize=10):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.stressComponents[:,stressComponent], title = f'Stress component: {stressComponent} ',
                          save_path=save_path, fontsize=fontsize)
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
if __name__ == "__main__":    
  from hex_structural_examples import StructuralExamples,getStructuralProblem

  problem = StructuralExamples.KnuckleAssembly
  nDOFDesired = 100000
  mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
  solver = linear_solvers.Solvers.DPCG # typically DPCG or PARDISO
  
  dsolver = deflation.DeflationSolver()
  startTime = time.time()
  if (solver == linear_solvers.Solvers.DPCG):
    nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
    dsolver.create_deflation_groups(mesh, nGroups)
    #dsolver.plot_deflation_groups(mesh)
    dsolver.create_deflation_matrix(mesh)
    dsolver.W = dsolver.W[bc.free_dofs, :]
  
  fe_solver = HexStructuralFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = solver,
        dsolver = dsolver,
        rtol = 1e-8,
        elem_body_force = elem_body_force)

  fe_solver.plot_mesh(plot_bc = True,offsetArrow = True)
  startTime = time.time()

  fe_solver.solve()
  print(f"Time to solve: {time.time() - startTime:.2f} seconds")
  fe_solver.postprocess()
  print(f"Maximum deformation: {fe_solver.max_deformation:.4e}")
  print(f"Maximum von Mises stress: {np.max(fe_solver.vonMisesStress):.4e}")
  
  fe_solver.plot_deformation()
  fe_solver.plot_vonMisesStress()