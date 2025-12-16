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
from hex_plotter import HexFEAPlotter 

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
                thermoElasticReferenceTemperature = 23.0,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.thermoElasticReferenceTemperature = thermoElasticReferenceTemperature  # Reference temperature for thermal expansion
    self.solver, self.kwargs = solver, kwargs
    self.sol = None
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
                      np.kron(self.mesh.edofMatThermal, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMatThermal, np.ones((1, 8))).flatten())
                      ).T.astype(int)
    
    self.elem_body_force = elem_body_force

    self.plotter = HexFEAPlotter(mesh) 


##################################################################
  def set_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
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
        # elem_stiff_mtrx = np.zeros((self.mesh.num_elems, 8, 8))
        # for e in range(self.mesh.num_elems):
        #   mat_idx = elem_mat_id[e]
        #   elem_stiff_mtrx[e] = self.elem_stiff[mat_idx] * elem_material_scaling[e]
        # elem_stiff_mtrx = elem_stiff_mtrx.flatten(order='C')
        elem_stiff_mtrx = np.einsum('mij, m -> mij',
                    self.elem_stiff,
                    elem_material_scaling).flatten(order = 'C')

    stiff_mtrx = sp.coo_matrix((elem_stiff_mtrx, (self.node_idx[:, 0], self.node_idx[:, 1])),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    
    self.stiff_mtrx = stiff_mtrx

    sol =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    self.sol = sol.copy()
    return sol
  #################################################################
  def getHMatrix(self, dx, dy, dz, nu):
      """
      Compute H matrix for thermo-elastic coupling with constitutive matrix D included.
      
      This matrix relates nodal temperatures to thermal forces:
          f_thermal =  H @ (T_nodes - T_ref)
      
      
      Parameters:
      -----------
      
      dx, dy, dz : float
          Element dimensions (m)
      nu : float
          Poisson's ratio
      
      Returns:
      --------
      H : ndarray (24, 8)
          Thermo-elastic coupling matrix
          
      Notes:
      ------
      This function was converted from MATLAB symbolic output.
      Generated by Symbolic Math Toolbox on 27-Nov-2025 14:09:59
      """
      # Intermediate calculations
      t2 = nu * 2.0
      t3 = t2 - 1.0
      t4 = 1.0 / t3
      t5 = ( dx * dy * t4) / 18.0
      t6 = ( dx * dy * t4) / 36.0
      t7 = ( dx * dz * t4) / 18.0
      t8 = ( dx * dz * t4) / 36.0
      t9 = ( dy * dz * t4) / 18.0
      t10 = ( dy * dz * t4) / 36.0
      t13 = (dx * dy * t4) / 72.0
      t16 = (dx * dz * t4) / 72.0
      t19 = (dy * dz * t4) / 72.0
      
      # Negative terms
      t11 = -t5
      t12 = -t6
      t14 = -t7
      t15 = -t8
      t17 = -t9
      t18 = -t10
      t20 = -t13
      t21 = -t16
      t22 = -t19
      
      # Assemble H matrix (24x8) using column-major order from MATLAB
      H_flat = np.array([
          t9, t17, t18, t10, t10, t18, t22, t19,
          t7, t8, t15, t14, t8, t16, t21, t15,
          t5, t6, t13, t6, t11, t12, t20, t12,
          t9, t17, t18, t10, t10, t18, t22, t19,
          t8, t7, t14, t15, t16, t8, t15, t21,
          t6, t5, t6, t13, t12, t11, t12, t20,
          t10, t18, t17, t9, t19, t22, t18, t10,
          t8, t7, t14, t15, t16, t8, t15, t21,
          t13, t6, t5, t6, t20, t12, t11, t12,
          t10, t18, t17, t9, t19, t22, t18, t10,
          t7, t8, t15, t14, t8, t16, t21, t15,
          t6, t13, t6, t5, t12, t20, t12, t11,
          t10, t18, t22, t19, t9, t17, t18, t10,
          t8, t16, t21, t15, t7, t8, t15, t14,
          t5, t6, t13, t6, t11, t12, t20, t12,
          t10, t18, t22, t19, t9, t17, t18, t10,
          t16, t8, t15, t21, t8, t7, t14, t15,
          t6, t5, t6, t13, t12, t11, t12, t20,
          t19, t22, t18, t10, t10, t18, t17, t9,
          t16, t8, t15, t21, t8, t7, t14, t15,
          t13, t6, t5, t6, t20, t12, t11, t12,
          t19, t22, t18, t10, t10, t18, t17, t9,
          t8, t16, t21, t15, t7, t8, t15, t14,
          t6, t13, t6, t5, t12, t20, t12, t11
      ])
      
      # Reshape to 24x8 using Fortran (column-major) order to match MATLAB
      H = H_flat.reshape((24, 8), order='F')
      
      return H
  #################################################################
  def get_thermoelastic_force(self, x: np.ndarray = None, material_model: MaterialModel = None) -> np.ndarray:
    """
    Add thermal forces to the finite element system.
    Supports multi-material: uses per-element E, alpha, nu.
    """
    if self.sol is None:
        raise ValueError("Solution not computed yet. Call solve() before get_thermal_force().")

    if x is None:
        x = np.ones((self.mesh.num_elems,))

    self.x = x  # store for postprocessing
    elem_material_scaling = get_structural_material_model_scaling(x, material_model)
    dx, dy, dz = self.mesh.elem_size

    f_thermoelastic = np.zeros(3 * self.mesh.num_nodes)  # Elastic force vector

    # Multi-material support
    if isinstance(self.mat_prop, list):
        for elem in range(self.mesh.num_elems):
            mp = self.mat_prop[elem]
            E = mp.youngs_modulus
            alpha = mp.thermal_expansion_coefficient
            nu = mp.poissons_ratio
            HMatrix = self.getHMatrix(dx, dy, dz, nu)
            elem_nodes = self.mesh.elemArray[elem]
            node_temp = self.sol[elem_nodes]
            f_thermal_elem = elem_material_scaling[elem] * E * alpha * HMatrix @ (node_temp - self.thermoElasticReferenceTemperature)
            for j in range(8):
                f_thermoelastic[3 * elem_nodes[j]] += f_thermal_elem[3 * j]
                f_thermoelastic[3 * elem_nodes[j] + 1] += f_thermal_elem[3 * j + 1]
                f_thermoelastic[3 * elem_nodes[j] + 2] += f_thermal_elem[3 * j + 2]
    else:
        E = self.mat_prop.youngs_modulus
        alpha = self.mat_prop.thermal_expansion_coefficient
        nu = self.mat_prop.poissons_ratio
        HMatrix = self.getHMatrix(dx, dy, dz, nu)
        for elem in range(self.mesh.num_elems):
            elem_nodes = self.mesh.elemArray[elem]
            node_temp = self.sol[elem_nodes]
            f_thermal_elem = elem_material_scaling[elem] * E * alpha * HMatrix @ (node_temp - self.thermoElasticReferenceTemperature)
            for j in range(8):
                f_thermoelastic[3 * elem_nodes[j]] += f_thermal_elem[3 * j]
                f_thermoelastic[3 * elem_nodes[j] + 1] += f_thermal_elem[3 * j + 1]
                f_thermoelastic[3 * elem_nodes[j] + 2] += f_thermal_elem[3 * j + 2]
    return f_thermoelastic


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
      edof = self.mesh.edofMatThermal
      
      # Compute displacement gradients
      self.strain = gradN @ self.sol[edof].T

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
            mask_low_pseudodensity = True,
            title = '',
            save_path=None,
            colormap = 'jet',
            auto_close = True,
            fontsize=10,
            cross_section=None,
            show_geometry=False,
            plotter = None,
            annotate_max_min = False)  
    
  #################################################################
  def plot_temperature(self,auto_close = True,plotter=None, save_path=None, annotate_max_min = False):
    # Return if no solution exists yet
    if not hasattr(self, 'sol'):
      return None
    return self.plotter.plot_temperature(self.sol, auto_close = auto_close, plotter=plotter,
                                       save_path=save_path, annotate_max_min = annotate_max_min)

#################################################################
  def plot_mesh(self, title = None,plot_bc = True,auto_close = True, save_path=None, camera_position = None, plotter=None):
    
    return self.plotter.plot_mesh_thermal(self.bc, title = title,plot_bc = plot_bc,
                auto_close = auto_close, save_path=save_path, camera_position = camera_position, plotter=plotter)

#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            plotter = None,
            fontsize=10):

    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, plotter = plotter, fontsize=fontsize)
    
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
    import hex_thermal_fea as fea
    import linear_solvers as lin_solv
    import time	
    from hex_thermal_examples import *

    problem = HexThermalExamples.LBracket
    solver = lin_solv.Solvers.PARDISO
    mesh, mat_prop, bc, elem_body_force = getThermalProblem(problem)
    
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
 
    
   
