"""Structural Finite Element Analysis."""

import time
import numpy as np
import linear_solvers as lin_sol
import hex_element_stiffness as elem_stiff
import mat_lib
import bound_cond
import linear_solvers as lin_solv
import os
import scipy.sparse
from scipy.sparse.linalg import eigsh
import pyvista as pv
script_dir = os.path.dirname(os.path.abspath(__file__))


class ModalFEA:
  """Linear Structural Modal Finite Element Analysis."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.Material,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
     # Handle single material or list of materials
    if isinstance(mat_prop, list):
      # Create element stiffness matrix for each material
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_structural(mp.youngs_modulus, mp.poissons_ratio, mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
      elem_mass_list = [elem_stiff.hex8_mass_matrix_structural(mp.mass_density, mesh.elem_size) 
            for mp in mat_prop]
      self.elem_mass = np.stack(elem_mass_list)
    else:
      self.elem_stiff = np.expand_dims(
          elem_stiff.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus, mat_prop.poissons_ratio,mesh.elem_size), axis=0)
      self.elem_mass = np.asarray(
                    elem_stiff.hex8_mass_matrix_structural(mat_prop.mass_density, mesh.elem_size))

    self.node_idx = np.stack((
                      np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
                      ).T.astype(int)
  
  def computeEigenModes(self,
            nEigenModes: int = 1,
            x: np.ndarray = None,
            elasticity_material_model: dict = None) -> np.ndarray:
    """Solve the modal structural finite element problem.

    Args:
        nEigenModes: Number of eigenmodes to compute.
      x: Array of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    if x is None:
      x = np.ones((self.mesh.num_elems,))

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


  
    if self.elem_stiff.shape[0] == 1:
      # Single material case (1,N,N)
      elem_stiff_mtrx = np.einsum('ij, e -> eij',
                    self.elem_stiff[0],
                    elem_material_scaling).flatten(order = 'C')
      elem_mass_mtrx = np.einsum('ij, e -> eij',
                                 self.elem_mass,
                                 elem_material_scaling).flatten(order = 'C')
    else:
      # Multiple materials case (M,N,N)
      # Assuming elem_mat_id contains material ID (0 to M-1) for each element
      # Randomly assign material IDs (0 or 1) to each element
      
      elem_stiff_mtrx = np.einsum('mij, e, em -> eij',
                    self.elem_stiff,
                    elem_material_scaling,
                    np.eye(self.elem_stiff.shape[0])[self.mesh.elemComponentId]).flatten(order = 'C')
      elem_mass_mtrx = np.einsum('mij, e, em -> eij',
                    self.elem_mass,
                    elem_material_scaling,
                    np.eye(self.elem_mass.shape[0])[self.mesh.elemComponentId]).flatten(order = 'C')

    K = scipy.sparse.csr_matrix(
      (elem_stiff_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
      shape=(self.bc.num_dofs, self.bc.num_dofs))
    K_tilde = (
          K[self.bc.free_dofs, :][:, self.bc.free_dofs]
          )

    
    M = scipy.sparse.csr_matrix(
      (elem_mass_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
      shape=(self.bc.num_dofs, self.bc.num_dofs))
  
    M_tilde = (
          M[self.bc.free_dofs, :][:, self.bc.free_dofs]
          )
    # Solve for eigenvalues and eigenvectors using shift-invert mode
    omega, eigenvecs = eigsh(K_tilde, k=nEigenModes, M=M_tilde, 
                                sigma=0, which='LM')
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = omega.argsort()
    eigenvals = np.sqrt(omega[idx])/(2*np.pi)
    eigenvecs = eigenvecs[:,idx]

    # Initialize full eigenvector array with zeros
    full_eigenvecs = np.zeros((self.bc.num_dofs, nEigenModes))

    # Set values at free DOFs
    full_eigenvecs[self.bc.free_dofs, :] = eigenvecs

    # Update eigenvecs to include all DOFs
    eigenvecs = full_eigenvecs
    
    self.eigenvals = eigenvals
    self.eigenvecs = eigenvecs
    return eigenvals, eigenvecs

  
  def plot_eigenmode(self, mode = 0,plotter = None):
    # Return if no solution exists yet
    if not hasattr(self, 'eigenvecs'):
      return None

    # Create vertices array
    vertices = self.mesh.node_xyz
     # Create vertices array
    vertices = self.mesh.node_xyz
  
    sol = self.eigenvecs[:,mode].copy()
    sol = sol.reshape((-1, 3))
   
    deltaMax = np.max(np.abs(sol))
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
    pv_mesh.point_data['values'] = sol
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    # Create plotter
  
    if plotter is None:
      plotter = pv.Plotter()
    
    
    plotter.add_title(f'Eigenmode: {mode}; freq: {self.eigenvals[mode]:0.3g} Hz', font_size=8)
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
    plotter.show() 
    
    return 

if __name__ == "__main__":    
  import hex_modal_fea as fea
  from hex_structural_examples import *


  problem = StructuralExamples.LBracket
  nDOFDesired = 50000
  mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
  solver = lin_solv.Solvers.PARDISO 

  startTime = time.time()

  modal_solver = fea.ModalFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = solver,
        rtol = 1e-8,
        elem_body_force = elem_body_force)

  nEigenModes = 3
  eigenvals, eigenvecs = modal_solver.computeEigenModes(nEigenModes = nEigenModes)
 
  print('-----------------------------')
  print("FEA time: ", time.time() - startTime)
  print('Eigenvalues: ', eigenvals)
  print('-----------------------------')
  for i in range(nEigenModes):
        modal_solver.plot_eigenmode(i)