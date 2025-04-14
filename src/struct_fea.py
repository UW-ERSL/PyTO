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
    u =  lin_sol.solve(stiff_mtrx,
                      self.total_force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return u

  def postprocess(self, u: jnp.ndarray) -> jnp.ndarray:
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
      uGrad = gradN @ u[edof[:, ::3]].T
      vGrad = gradN @ u[edof[:, 1::3]].T
      wGrad = gradN @ u[edof[:, 2::3]].T
      
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
  
if __name__ == "__main__":    
  jax.config.update("jax_enable_x64", True)
  import plots
  from examples_structural import *

  problem = StructuralExamples.MBBB
  nDOFDesired = 60000
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

  plots.plotMesh(fe_solver.mesh, bc=bc)
  u = np.asarray(fe_solver.solve())
  delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
  deltaMax = np.max(delta)
  nDOF = 3*fe_solver.mesh.num_nodes
  fe_solver.postprocess(u)
  maxStress = np.max(fe_solver.vonMisesStress)
 
  print('-----------------------------')
  print("nDof: ", nDOF)
  print('Solver: ', fe_solver.solver.name)
  print("FEA time: ", time.time() - startTime)
  print('Max displacement: ', f"{deltaMax:.2g}")
  print("Max von Mises stress: ", f"{maxStress:.2g}")
  print('-----------------------------')
  
 
  plots.plotMesh(fe_solver.mesh, bc=None, u=u, 
                 title=f'dof = {nDOF}, Max deformation: {deltaMax:.3e}')

  plots.plotElementField(fe_solver.mesh, fe_solver.vonMisesStress,
                        title='von Mises stress', cmap='jet')
  

  #plots.plotElementField(fe_solver.mesh, fe_solver.strainComponents[:,3], title=f'Strain component: {'γxy'}', cmap='jet')
  
 
  