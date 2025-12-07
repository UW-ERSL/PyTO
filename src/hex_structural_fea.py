"""Structural Finite Element Analysis."""

from topopt_material_model import *
import time
import numpy as np
import os
import mat_lib
import bound_cond
import linear_solvers
import hex_element_stiffness
import deflation
import scipy.sparse as sp
from hex_plotter import HexFEAPlotter 

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
    self.plotter = HexFEAPlotter(mesh) 


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
    self.external_force = self.bc.force.copy()
    if self.elem_body_force is not None: # convert element body forces to nodal forces
      elem_force = self.elem_body_force.copy()
      for i in range(3):
        elem_force[i::3]  *= x
        
      node_forces = np.zeros((self.mesh.num_nodes * 3,))
      node_forces[0::3] = self.mesh.elem_to_node_field_mapping* elem_force[0::3] 
      node_forces[1::3] = self.mesh.elem_to_node_field_mapping* elem_force[1::3] 
      node_forces[2::3] = self.mesh.elem_to_node_field_mapping* elem_force[2::3] 
      self.external_force += node_forces

    
    if hasattr(self.mesh, 'externalSprings') and self.mesh.externalSprings is not None:
      # Add spring stiffnesses to diagonal terms
      # Convert to CSR format for modification
      self.stiff_mtrx  = self.stiff_mtrx.tocsr()
      for KSpring, dof in self.mesh.externalSprings:
          self.stiff_mtrx [dof,dof] += KSpring

     # purely elastic
    sol = linear_solvers.solve(self.stiff_mtrx,
                        self.external_force,
                        self.solver,
                        self.bc,
                        dsolver = self.dsolver,
                        **self.kwargs)
    self.sol = sol.copy()
    self.deformation = np.sqrt(sol[0::3]**2 + sol[1::3]**2 + sol[2::3]**2)
    self.max_deformation = np.max(self.deformation)
    self.solElastic = self.sol.copy()

    self.total_force = self.external_force.copy()
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
      r = get_stress_relaxation_factor()  # SIMP like penalization for stress
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
        
      correction = get_stress_relaxation_correction(self.x)
      self.stressComponents *= correction 
      eStress = self.stressComponents
      self.vonMisesStress = np.sqrt(0.5*((eStress[:,0]-eStress[:,1])**2 +
                (eStress[:,1]-eStress[:,2])**2 +
                (eStress[:,2]-eStress[:,0])**2) +
                3*(eStress[:,3]**2 + eStress[:,4]**2 +
                   eStress[:,5]**2))
      
      pNorm = get_pNorm_exponent()
      self.pNormStress = (np.sum(self.vonMisesStress**pNorm))**(1/pNorm)  
     

      self.elemStrainEnergy = 0.5 * np.sum(strain * eStress, axis=1)  # Element-wise strain energy
      #print(f"Maximum von Mises stress: {np.max(self.vonMisesStress):.4e}")
      return 
#################################################################
  def plot_mesh(self, title = None,plot_bc = True,rel_arrow_scale = 0.1, 
                 save_path=None,offsetArrow = False, plotter=None):
    
    return self.plotter.plot_mesh_structural(self.bc, title = title, plot_bc = plot_bc,rel_arrow_scale = rel_arrow_scale, 
                save_path=save_path,offsetArrow = offsetArrow, plotter=plotter)
  
################################################################# 
  def plot_deformation(self,show_geometry=False, auto_close = True, save_path=None, plotter=None):
    
    return self.plotter.plot_structural_deformation(self.sol,
                                                     show_geometry=show_geometry, 
                                                     auto_close = auto_close, 
                                                     save_path=save_path,
                                                       plotter=plotter)
   

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
            annotate_max_min = False,
            colors=None):  # New parameter for custom colors
  

    return self.plotter.plot_elem_field(elem_field,
            mask_low_pseudodensity = mask_low_pseudodensity,
            title = title,
            save_path=None,
            colormap = 'jet',
            auto_close = True,
            fontsize=fontsize,
            cross_section=cross_section,
            show_geometry=show_geometry,
            plotter = plotter,
            annotate_max_min = annotate_max_min)  
    
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
    self.plot_elem_field(self.stressComponents[:,stressComponent], title = f'Stress: {stressComponent} ',
                          save_path=save_path, fontsize=fontsize)
#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            fontsize=10,
            plotter = None):
    
    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, fontsize=fontsize,plotter = plotter)
    

 #################################################################
  def plot_pseudo_density_realtime(self, title='Pseudo density', iteration=0, external_plotter=None):
    """
    Real-time visualization with proper Qt event loop handling.
    
    Args:
        title: Title for the visualization
        iteration: Current iteration number
        external_plotter: Optional external plotter (e.g., from GUI)
    
    Notes:
        This method delegates to the HexFEAPlotter, which handles the actual visualization.
        When external_plotter is provided (GUI mode), it will use that plotter instead of
        creating a new BackgroundPlotter window.
    """
    return self.plotter.plot_pseudo_density_realtime(
        title=title, 
        iteration=iteration,
        external_plotter=external_plotter
    )


    
#################################################################
if __name__ == "__main__":    
  from hex_structural_examples import StructuralExamples,getStructuralProblem
 

  problem = StructuralExamples.CantileverMidLoad
  nDOFDesired = 30000
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
  print(f"Maximum p-norm stress: {fe_solver.pNormStress:.4e}")

  fe_solver.plot_deformation(show_geometry=True)
  fe_solver.plot_vonMisesStress()
  fe_solver.plot_stress_component(0)
  