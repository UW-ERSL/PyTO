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
from hex_plotter import HexFEAPlotter 

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
                      np.kron(self.mesh.edofMatStructural, np.ones((24, 1))).flatten(),
                      np.kron(self.mesh.edofMatStructural, np.ones((1, 24))).flatten())
                      ).T.astype(int)
    
    self.plotter = HexFEAPlotter(mesh)  
  
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

  
    eigenvector = self.eigenvecs[:, mode]  # ← This gives 1D
    eigenvalue = self.eigenvals[mode]
    
    return self.plotter.plot_eigenmode(
        eigenvector, eigenvalue, mode_number=mode, plotter=plotter)

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