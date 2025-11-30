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

# Format values based on magnitude
def format_value(val):
  abs_val = abs(val)
  if abs_val == 0:
    return '0.0'
  elif abs_val < 0.01 or abs_val >= 1000:
    return f'{val:.3e}'
  else:
    return f'{val:.3f}'
  
class HexStructuralFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
         mesh,
         mat_prop: mat_lib.Material | list[mat_lib.Material],
         bc: bound_cond.BC,
         solver: linear_solvers.Solvers,
         dsolver: deflation.DeflationSolver = None,
         elem_body_force: np.ndarray = None,
         thermo_elastic_force = None,
         **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.thermo_elastic_force = thermo_elastic_force
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
            np.kron(self.mesh.edofMatStructural, np.ones((24, 1))).flatten(),
            np.kron(self.mesh.edofMatStructural, np.ones((1, 24))).flatten())
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
  def set_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
    
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(mp.youngs_modulus,mp.poissons_ratio, self.mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
          hex_element_stiffness.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus,mat_prop.poissons_ratio, self.mesh.elem_size), axis=0)

#################################################################
  def set_thermal_forces(self, thermo_elastic_force: np.ndarray):
    self.thermo_elastic_force = thermo_elastic_force
  
  #################################################################
  def solve(self,x: np.ndarray = None,
            material_model: MaterialModel =  MaterialModel.SIMP ) -> np.ndarray:
    """Solve the structural finite element problem.

    Args:
      x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = np.ones((self.mesh.num_elems,))

    self.x = x # store for postprocessing

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

    self.elem_stiff_mtrx = elem_stiff_mtrx
    self.stiff_mtrx = sp.coo_matrix((elem_stiff_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
                   shape=(self.bc.num_dofs, self.bc.num_dofs))
    self.total_force = self.bc.force.copy()
    if self.elem_body_force is not None: # convert element body forces to nodal forces
      elem_force = self.elem_body_force.copy()
      for i in range(3):
        elem_force[i::3]  *= x
        
      node_forces = np.zeros((self.mesh.num_nodes * 3,))
      node_forces[0::3] = self.mesh.elem_to_node_field_mapping* elem_force[0::3] 
      node_forces[1::3] = self.mesh.elem_to_node_field_mapping* elem_force[1::3] 
      node_forces[2::3] = self.mesh.elem_to_node_field_mapping* elem_force[2::3] 
      self.total_force += node_forces

    
    if hasattr(self.mesh, 'externalSprings') and self.mesh.externalSprings is not None:
      # Add spring stiffnesses to diagonal terms
      # Convert to CSR format for modification
      self.stiff_mtrx  = self.stiff_mtrx.tocsr()
      for KSpring, dof in self.mesh.externalSprings:
          self.stiff_mtrx [dof,dof] += KSpring

     # purely elastic
    sol = linear_solvers.solve(self.stiff_mtrx,
                        self.total_force,
                        self.solver,
                        self.bc,
                        dsolver = self.dsolver,
                        **self.kwargs)
    self.sol = sol.copy()
    self.deformation = np.sqrt(sol[0::3]**2 + sol[1::3]**2 + sol[2::3]**2)
    self.max_deformation = np.max(self.deformation)
    self.solElastic = self.sol.copy()

    if self.thermo_elastic_force is not None: # must solve again since we need to keep track of elastic deformations only 
      self.solElastic = self.sol.copy()
      self.total_force += self.thermo_elastic_force
      sol = linear_solvers.solve(self.stiff_mtrx,
                        self.total_force,
                        self.solver,
                        self.bc,
                        dsolver = self.dsolver,
                        **self.kwargs)
      self.sol = sol.copy()
      self.deformation = np.sqrt(self.sol[0::3]**2 + self.sol[1::3]**2 + self.sol[2::3]**2)
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
      for i in range(3):
        gradN[i, :] = 2*gradN[i,:] / self.mesh.elem_size[i]
      # Get element degrees of freedom
      edof = self.mesh.edofMatStructural
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
      self.strainComponents = strain.copy() # store the total strain (elastic + thermal)
    
      if (self.thermo_elastic_force is not None): # We must use only the elastic strain for computing stresses
        # Compute elastic strains only
        uGradElastic = gradN @ self.solElastic[edof[:, ::3]].T
        vGradElastic = gradN @ self.solElastic[edof[:, 1::3]].T
        wGradElastic = gradN @ self.solElastic[edof[:, 2::3]].T
        
        # Compute Engineering strains
        strain = np.stack([
          uGradElastic[0], vGradElastic[1], wGradElastic[2],
          uGradElastic[1] + vGradElastic[0],
          uGradElastic[2] + wGradElastic[0],
          vGradElastic[2] + wGradElastic[1]
        ], axis=1)  # Shape: (num_elems, 6)

      # STRESS_RELAXATION method;
      r = SIMP_STRESS_RELAXATION  # SIMP like penalization for stress
      if isinstance(self.mat_prop, list):
        # Create D matrix for each material
        D_list = []
      
        for mp in self.mat_prop:
          E = mp.youngs_modulus
          nu = mp.poissons_ratio
          D = hex_element_stiffness.isotropic_constitutive_matrix ( E, nu)
          D_list.append(D)
        D_stack = np.stack(D_list)
        self.stressComponents =  np.einsum('eij, ej -> ei', D_stack, strain)
      else:
        E = self.mat_prop.youngs_modulus
        nu = self.mat_prop.poissons_ratio
        D = hex_element_stiffness.isotropic_constitutive_matrix ( E, nu)
        self.stressComponents = np.einsum('ij,ej->ei', D, strain)
        
      correction = (EVOID_RELATIVE + (1-EVOID_RELATIVE) * (self.x**r)).reshape((-1,1))
      self.stressComponents *= correction 
      eStress = self.stressComponents
      self.vonMisesStress = np.sqrt(0.5*((eStress[:,0]-eStress[:,1])**2 +
                (eStress[:,1]-eStress[:,2])**2 +
                (eStress[:,2]-eStress[:,0])**2) +
                3*(eStress[:,3]**2 + eStress[:,4]**2 +
                   eStress[:,5]**2))
      
      self.pNormStress = (np.sum(self.vonMisesStress**PNORM_EXPONENT))**(1/PNORM_EXPONENT)  
     

      self.elemStrainEnergy = 0.5 * np.sum(strain * eStress, axis=1)  # Element-wise strain energy
      #print(f"Maximum von Mises stress: {np.max(self.vonMisesStress):.4e}")
      return 
#################################################################
  def plot_mesh(self, title = None,plot_bc = True,rel_arrow_scale = 0.5, 
                auto_close = True, save_path=None,offsetArrow = False,transparency = 1.0, plotter=None):
    
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
    externalPlotter = False # assume that a plotter is provided
    if (plotter is None):
      externalPlotter = True
      # Create plotter
      if save_path is None:
        plotter = self.pyVistaPlotter
        if plotter.iren is None:
          self.create_pyvista_plotter()
          plotter = self.pyVistaPlotter 
          plotter.show(interactive_update=True, auto_close=False)
      else:
        plotter = pv.Plotter(off_screen=True) # for saving images
        plotter.camera_position =self.camera_position
        plotter.enable_anti_aliasing()
      # Add coordinate axes widget
      plotter.add_axes(
                    xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                    line_width=2,
                    labels_off=False,  # Show axis labels
                    color='black'
                    )
    
      plotter.add_title(title, font_size=8)
  
    plotter.add_mesh(
            pv_mesh,
            color='lightgreen',
            show_edges=True,
            edge_color='black',
            line_width=1,
            opacity=transparency  # Add transparency (0.0 is fully transparent, 1.0 is fully opaque)
            )

    if (plot_bc):
      # Add dots and force arrows for labeled nodes
      point_size = 10.0  # Size of dots in pixels

      # Add black dots for label 1 (fixed nodes)
      label1_nodes = np.where(self.mesh.node_indices[:, 3] == 1)[0]
  
      if len(label1_nodes) > 0 and self.bc is not None:
        points1 = vertices[label1_nodes]
        pts = pv.PointSet(points1)
        plotter.add_mesh(pts, color='black')

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
      if (externalPlotter):
        # Show plot interactively if not saving
        plotter.show(interactive_update=not auto_close, auto_close=auto_close)
      else:
        plotter.show()
    self.camera_position = plotter.camera_position # For all future displays
    return
################################################################# 
  def plot_deformation(self,show_geometry=False, auto_close = True, save_path=None, plotter=None):
    """Plot the deformed mesh with the deformation scaled by a factor."""
    # Return if no solution exists yet
    if not hasattr(self, 'sol'):
      return None

    # Create vertices array
    vertices = self.mesh.node_xyz.copy()
  
    sol = self.sol.copy()
    sol = sol.reshape((-1, 3))
    delta = self.deformation
    deltaMax = self.max_deformation
    if deltaMax < 1e-16:
      deltaMax = 1e-16
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

    externalPlotter = False # assume that a plotter is provided
    if plotter is None: 
      externalPlotter = True # create a new plotter
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
                    scalars='values' if delta is not None else 'density',
                    show_edges=True,
                    cmap='jet',
                    edge_color='black',
                    line_width=1,
                    scalar_bar_args={
                            'title': 'Deformation',
                            'vertical': True,
                            'position_x': 0.8,
                            'position_y': 0.3,
                            'width': 0.06
                            }
                  )
    if (show_geometry):
      vertices = self.mesh.stlGeom.mesh.vectors.reshape(-1, 3)
      faces = np.arange(len(vertices)).reshape(-1, 3)
      faces = np.column_stack((np.full(len(faces), 3), faces))
      geomMesh = pv.PolyData(vertices, faces)
      plotter.add_mesh(geomMesh, opacity=0.25, color='white', show_edges=True)

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
            cross_section=None,
            show_geometry=False,
            plotter = None,
            colors=None):  # New parameter for custom colors
    """
    Plot element field on the mesh, with optional discrete custom colors.

    Args:
        elem_field: Field values for each element (e.g., material indices)
        mask_low_pseudodensity: Whether to filter out elements with low density
        title: Plot title
        save_path: Path to save the image
        colormap: Color map for the field visualization (ignored if colors is provided)
        auto_close: Whether to auto-close the plotter
        fontsize: Font size for the title
        cross_section: Dict with keys 'axis' ('x', 'y', or 'z') and 'position' (float)
                      to create a cross-section view, or None for full view
        colors: List or array of hex codes or RGB tuples for each element (discrete coloring)
    """
    # Filter elements based on pseudo_density
    if mask_low_pseudodensity:
        mask = self.mesh.elemPseudoDensity > 0.5
        filtered_elems = self.mesh.elemArray[mask]
        filtered_field = np.array(elem_field)[mask]
        filtered_colors = None
        if colors is not None:
            filtered_colors = np.array(colors)[mask]
    else:
        filtered_elems = self.mesh.elemArray
        filtered_field = np.array(elem_field)
        filtered_colors = np.array(colors) if colors is not None else None

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

    externalPlotter = False
    if plotter is None:
        externalPlotter = True
        if save_path is None:
            plotter = self.pyVistaPlotter
            if plotter.iren is None:
                self.create_pyvista_plotter()
                plotter = self.pyVistaPlotter
        else:
            plotter = pv.Plotter(off_screen=True)
        plotter.disable_depth_peeling()
        plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False,
            color='black'
        )

    # If cross-section is specified, create a clipped mesh
    if cross_section is not None:
        axis = cross_section.get('axis', 'x').lower()
        rel_position = cross_section.get('position', 0.0)
        bbox_min = pv_mesh.bounds[::2]
        bbox_max = pv_mesh.bounds[1::2]
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
        pv_mesh = pv_mesh.clip(normal=normal, origin=origin)
        if title:
            title += f" (Cross-section {axis}={position})"
        else:
            title = f"Cross-section {axis}={position}"

    # Add mesh to plotter
    if filtered_colors is not None:
        # Use custom discrete colors for each cell
        plotter.add_mesh(
            pv_mesh,
            scalars=filtered_colors,
            rgb=True,
            show_edges=True,
            edge_color='black',
            line_width=1,
            opacity=1.0,
            scalar_bar_args={
                'title': title,
                'vertical': True,
                'position_x': 0.85,   # Move to the right (0.0 = left, 1.0 = right)
                'position_y': 0.05,   # Move to the bottom (0.0 = bottom, 1.0 = top)
                'width': 0.08,        # Make it narrower
                'height': 0.9,        # Make it taller
                'title_font_size': 18,
                'label_font_size': 18
            }
        )
    else:
        plotter.add_mesh(
            pv_mesh,
            scalars='field',
            cmap=colormap,
            show_edges=True,
            edge_color='black',
            line_width=1,
            scalar_bar_args={
          'title': title,
          'vertical': True,
          'position_x': 0.85,   # Move to the right (0.0 = left, 1.0 = right)
          'position_y': 0.05,   # Move to the bottom (0.0 = bottom, 1.0 = top)
          'width': 0.08,        # Make it narrower
          'height': 0.9,        # Make it taller
          'title_font_size': 18,
          'label_font_size': 18
            }
        )
    # Add annotations for max and min values
    field_values = pv_mesh.cell_data['field']
    if len(field_values) > 0:
      max_idx = np.argmax(field_values)
      min_idx = np.argmin(field_values)
      max_val = field_values[max_idx]
      min_val = field_values[min_idx]
      
      # Get cell centers for annotation positions
      cell_centers = pv_mesh.cell_centers().points
      max_pos = cell_centers[max_idx]
      min_pos = cell_centers[min_idx]
      
      # Add text annotations with larger font and better visibility
      plotter.add_point_labels(
      [max_pos],
      [f'Max: {format_value(max_val)}'],
      point_size=10,
      font_size=fontsize * 2,
      text_color='red',
      fill_shape=True,
      shape_color='white',
      shape_opacity=0.9,
      bold=True,
      always_visible=True
      )
      
      plotter.add_point_labels(
      [min_pos],
      [f'Min: {format_value(min_val)}'],
      point_size=10,
      font_size=fontsize * 2,
      text_color='blue',
      fill_shape=True,
      shape_color='white',
      shape_opacity=0.9,
      bold=True,
      always_visible=True
      )
    if (show_geometry):
      vertices = self.mesh.stlGeom.mesh.vectors.reshape(-1, 3)
      faces = np.arange(len(vertices)).reshape(-1, 3)
      faces = np.column_stack((np.full(len(faces), 3), faces))
      geomMesh = pv.PolyData(vertices, faces)
      plotter.add_mesh(geomMesh, opacity = 0.5, color='white', show_edges=True)
    # Save image if path is provided

    if save_path:
        # Get mesh bounds
        bounds = pv_mesh.bounds
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]

        # Set camera above the mesh, looking down the z-axis
        distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2
        camera_location = [center[0], center[1], center[2] + distance]
        focal_point = center
        view_up = [0, 1, 0]

        plotter.camera_position = [camera_location, focal_point, view_up]
        plotter.camera.zoom(1.3)
        plotter.screenshot(save_path)
        plotter.close()
    else:
        if externalPlotter:
            plotter.show(interactive_update=not auto_close, auto_close=auto_close)
        else:
            plotter.show()
    self.camera_position = plotter.camera_position
    return
#################################################################
  def plot_vonMisesStress(self,
            save_path=None,
            fontsize=8,plotter = None):
    
    self.plot_elem_field(self.vonMisesStress, title = f'vonMises stress ',
                          save_path=save_path, fontsize=fontsize,plotter = plotter)

#################################################################    
  def plot_strain_component(self,strainComponent = 0,
            save_path=None,
            fontsize=8):
    
    self.plot_elem_field(self.strainComponents[:,strainComponent], title = f'Strain: {strainComponent} ',
                          save_path=save_path,fontsize=fontsize)
#################################################################
  def plot_stress_component(self,stressComponent = 0,
            save_path=None,
            fontsize=10):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.stressComponents[:,stressComponent], title = f'Stress: {stressComponent} ',
                          save_path=save_path, fontsize=fontsize)
#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            fontsize=10,
            plotter = None):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, fontsize=fontsize,plotter = plotter)
    
#################################################################
if __name__ == "__main__":    
  from hex_structural_examples import StructuralExamples,getStructuralProblem

  problem = StructuralExamples.LBracket
  nDOFDesired = 50000
  mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
  solver = linear_solvers.Solvers.DPCG # typically DPCG or PARDISO
  
  dsolver = deflation.DeflationSolver()

  if (solver == linear_solvers.Solvers.DPCG):
    nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
    dsolver.create_deflation_groups(mesh, nGroups)
    #dsolver.plot_deflation_groups(mesh)
    dsolver.create_deflation_matrix(mesh)
   
  
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
  print(f"Maximum p-norm stress (PNORM = {PNORM_EXPONENT}): {fe_solver.pNormStress:.4e}")

  fe_solver.plot_deformation(show_geometry=True)
  fe_solver.plot_vonMisesStress()
  fe_solver.plot_stress_component(0)
  