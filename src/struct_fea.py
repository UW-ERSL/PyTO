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
import plots
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
							 mat_prop: mat_lib.StructuralMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
               elem_body_force: jnp.ndarray = None,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_structural(mat_prop, mesh.elem_size))

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
      elem_material_scaling: Array of (num_elems,) of the material scaling.
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
    elif elasticity_material_model['name'] == 'RAMP': 
      penal = elasticity_material_model['penal']
      elem_material_scaling = x/(1+penal*(1-x))
    elif elasticity_material_model['name'] == 'Custom':
      # Custom model from the paper here: https://doi.org/10.1002/nme.2499 
      alpha = elasticity_material_model['alpha']
      penal = elasticity_material_model['penal']
      elem_material_scaling = (alpha-1)/alpha * x ** penal + (1/alpha) * x


    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff,
									               elem_material_scaling).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
  
    self.total_force = self.bc.force.copy()
    if self.elem_body_force is not None:
      elem_force = self.elem_body_force.copy()
      for i in range(3):
        elem_force[i::3]  *= x
        
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

if __name__ == "__main__":    
  jax.config.update("jax_enable_x64", True)

  from examples_structural import *

  example = StructuralExamples.EdgeCantilever
  elem_body_force = None # by default no body force
  solver = lin_solv.Solvers.PARDISO # typically DPCG or PARDISO
  


  if example == StructuralExamples.EdgeCantilever:
    mesh, mat_prop, bc = createEdgeCantileverProblem(nDOFDesired=10000)
  elif example == StructuralExamples.MBB:
    mesh, mat_prop, bc = createMBBProblem(nDOFDesired=10000)   
  elif example == StructuralExamples.DistributedLoad:
    mesh, mat_prop, bc = createDistributedLoadProblem(nDOFDesired=10000)    
  elif example == StructuralExamples.Multiload:
    mesh, mat_prop, bc = createMultiloadProblem(nDOFDesired=10000)
  elif example == StructuralExamples.LBracket:
    mesh, mat_prop, bc = createLBracketProblem(nDOFDesired=10000)    
  elif example == StructuralExamples.CompliantMechanism:
    mesh, mat_prop, bc = createCompliantMechanismProblem(nDOFDesired=10000)
  elif example == StructuralExamples.BeamSurfaceLoad:
    mesh, mat_prop, bc = createBeamSurfaceLoadProblem(nDOFDesired=10000)
  elif example == StructuralExamples.FilletedBeam:
    mesh, mat_prop, bc = createFilletedBeamProblem(nDOFDesired=100000)
  elif example == StructuralExamples.CentrifugalPlate:
    mesh, mat_prop, bc, elem_body_force  = createCentrifugalPlateProblem(nDOFDesired=20000)
  elif example == StructuralExamples.GravityBar:
    mesh, mat_prop, bc, elem_body_force  = createGravityBarProblem(nDOFDesired=10000)
  elif example == StructuralExamples.GravityPlate:
    mesh, mat_prop, bc, elem_body_force  = createGravityPlateProblem(nDOFDesired=30000)
  elif example == StructuralExamples.ArrowHead:
    mesh, mat_prop, bc = createArrowHeadProblem(nDOFDesired=200000)
  elif example == StructuralExamples.BliskQuarter:
    mesh, mat_prop, bc = createBliskQuarterModelProblem(nDOFDesired=50000)
  elif example == StructuralExamples.BliskFull:
    mesh, mat_prop, bc = createBliskFullModelProblem(nDOFDesired=20000)

  
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

  u = np.asarray(fe_solver.solve())
  delta = np.sqrt(u[0::3]**2 +  u[1::3]**2 +  u[2::3]**2)
  deltaMax = np.max(delta)
  nDOF = 3*fe_solver.mesh.num_nodes

  print('-----------------------------')
  print("nDof: ", nDOF)
  print('Solver: ', fe_solver.solver.name)
  print("FEA time: ", time.time() - startTime)
  print('Max displacement: ', deltaMax)
  print('-----------------------------')

  plots.plotMesh(fe_solver.mesh, fe_solver.bc, u,
      title=f'dof = {nDOF}, Max deformation: {deltaMax:.3e}')