"""Structural Finite Element Analysis."""

from topopt_material_model import *
import time
import numpy as np
import os
import pyvista as pv
import mat_lib
import bound_cond
from bound_cond import apply_dirichlet_bc_torch
import linear_solvers
import hex_element_stiffness
import deflation
import scipy.sparse as sp
import torch_spsolve
from torch_spsolve import solve as sparse_spsolve


script_dir = os.path.dirname(os.path.abspath(__file__))

class HexStructuralFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
         mesh,
         mat_prop: mat_lib.Material | list[mat_lib.Material],
         bc: bound_cond.BC,
         solver: torch_spsolve.Solvers,
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

  def solve(self, x, material_model: MaterialModel = None):
      """Solve the structural finite element problem using torch and autograd.

      Assembles the global stiffness matrix in torch, applies Dirichlet
      boundary conditions, and solves for the displacement field using an
      autograd-capable sparse linear solver.

      Args:
          x: 1D torch tensor of shape (num_elems,) containing the element
            densities / design variables.
          material_model: MaterialModel enum selecting the penalization
            scheme (e.g., SIMP, RAMP, SIMPPLUS). If None, x is used
            directly as the stiffness scaling in the material model.

      Returns:
          u: 1D torch tensor of shape (num_dofs,) containing the displacement
            field for all DOFs.
      """

      self.x = x
      device, dtype = x.device, x.dtype
      ndof = self.bc.num_dofs

      elem_material_scaling = get_structural_material_model_scaling_torch(x, material_model)  # (E,)

      if self.elem_stiff.shape[0] == 1:
          elem_stiff_mtrx_torch = torch.tensor(self.elem_stiff[0], dtype=dtype, device=device)      # (24,24)
          elem_stiff_mtrx = torch.einsum("ij,e->eij", elem_stiff_mtrx_torch, elem_material_scaling)                              # (E,24,24)
      else:
          elem_stiff_mtrx_torch = torch.tensor(self.elem_stiff, dtype=dtype, device=device)         # (M,24,24)
          elem_stiff_mtrx = torch.einsum("mij,m->mij", elem_stiff_mtrx_torch, elem_material_scaling)                             # (M,24,24)

      vals = elem_stiff_mtrx.reshape(-1)

      if not hasattr(self, "_torch_node_idx"):
          self._torch_node_idx = torch.from_numpy(self.node_idx.T).long()
      idx = self._torch_node_idx.to(device)

      self.stiff_mtrx = torch.sparse_coo_tensor(idx, vals, (ndof, ndof), device=device, dtype=dtype).coalesce()

      if getattr(self.mesh, "externalSprings", None):
          dofs, ks = zip(*self.mesh.externalSprings)
          spring_idx = torch.tensor([dofs, dofs], dtype=torch.long, device=device)
          spring_vals = torch.tensor(ks, dtype=dtype, device=device)
          spring_K = torch.sparse_coo_tensor(spring_idx, spring_vals, (ndof, ndof),
                                            device=device, dtype=dtype)
          self.stiff_mtrx = (self.stiff_mtrx + spring_K).coalesce()

      f = torch.tensor(self.bc.force, dtype=dtype, device=device)

      if self.elem_body_force is not None:
          elem_force = torch.tensor(self.elem_body_force, dtype=dtype, device=device)  # (3E,)
          elem_force = elem_force.view(-1, 3) * elem_material_scaling.view(-1, 1)                      # (E,3) scaled
          elem_force = elem_force.reshape(-1)                                         # (3E,)

          map_vec = torch.tensor(self.mesh.elem_to_node_field_mapping,
                                dtype=dtype, device=device)

          fx = map_vec * elem_force[0::3]
          fy = map_vec * elem_force[1::3]
          fz = map_vec * elem_force[2::3]

          node_forces = torch.stack([fx, fy, fz], dim=1).reshape(-1)  # (3 * num_nodes,)
          f = f + node_forces

      self.total_force = f

      K_bc, f_bc = apply_dirichlet_bc_torch(self.stiff_mtrx, f, self.bc)
      u = sparse_spsolve(K_bc, f_bc, solver=self.solver)  # (ndof,)

      self.sol = u
      self.deformation = torch.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
      self.max_deformation = self.deformation.detach().max().item()

      return u


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
      sol_np = self.sol.detach().cpu().numpy()
      x_np = self.x.detach().cpu().numpy()
      gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1], 
        [-1, -1, -1, -1, 1, 1, 1, 1]
      ])
      for i in range(3):
        gradN[i, :] = 2*gradN[i,:] / self.mesh.elem_size[i]
      # Get element degrees of freedom
      edof = self.mesh.edofMat
      
      # Compute displacement gradients
      uGrad = gradN @ sol_np[edof[:, ::3]].T
      vGrad = gradN @ sol_np[edof[:, 1::3]].T
      wGrad = gradN @ sol_np[edof[:, 2::3]].T
      
      # Compute Engineering strains
      strain = np.stack([
        uGrad[0], vGrad[1], wGrad[2],
        uGrad[1] + vGrad[0],
        uGrad[2] + wGrad[0],
        vGrad[2] + wGrad[1]
      ], axis=1)  # Shape: (num_elems, 6)

    
      # STRESS_RELAXATION method;
      # See Efficient stress-constrained topology optimizationusing inexact design sensitivities
      # by Oded Amir, 2021
      # Constitutive matrix D for each material
      q = 0.5  # SIMP like penalization for stress
     
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
      correction = (EVOID_RELATIVE + (1-EVOID_RELATIVE) * (x_np**q)).reshape((-1,1))
      eStress = correction * self.stressComponents
      self.vonMisesStress = np.sqrt(0.5*((eStress[:,0]-eStress[:,1])**2 +
                (eStress[:,1]-eStress[:,2])**2 +
                (eStress[:,2]-eStress[:,0])**2) +
                3*(eStress[:,3]**2 + eStress[:,4]**2 +
                   eStress[:,5]**2))
      self.strainComponents = strain

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
      plotter.add_mesh(geomMesh, color='white', show_edges=True)

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
                'title_font_size': 12,
                'label_font_size': 12
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
                'title_font_size': 12,
                'label_font_size': 12
            }
        )

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
    