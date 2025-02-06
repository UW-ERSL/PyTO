"""Structural Finite Element Analysis."""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs

import mesher
import linear_solvers as lin_sol
import element_stiffness as elem_stiff
import mat_lib
import bound_cond
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

class ThermalFEA:
  """Linear Thermal Finite Element Analysis."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.ThermalMaterial,
							 bc: bound_cond.BC,
							 solver: lin_sol.Solvers,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.solver, self.kwargs = solver, kwargs
    self.elem_stiff = jnp.asarray(
                    elem_stiff.hex8_stiffness_matrix_thermal(mat_prop, mesh.elem_size))

    self.node_idx = jnp.stack((
                      np.kron(self.mesh.edofMat, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMat, np.ones((1, 8))).flatten())
                      ).T.astype(int)


  def solve(self, elem_conductivity: jnp.ndarray) -> jnp.ndarray:
    """Solve the thermal finite element problem.

    Args:
      elem_conductivity: Array of (num_elems,) of the young's modulus of each
        element.

    Returns: Array of (num_dofs,) of the solution to the finite element problem.
    """
    elem_stiff_mtrx = jnp.einsum('ij, e -> eij',
                                 self.elem_stiff,
									               elem_conductivity).flatten(order = 'C')

    stiff_mtrx = jax_sprs.BCOO((elem_stiff_mtrx, self.node_idx),
                                shape=(self.bc.num_dofs, self.bc.num_dofs))
    

    
    u =  lin_sol.solve(stiff_mtrx,
                      self.bc.force,
                      self.solver,
                      self.bc,
                      **self.kwargs)
    return u

def createMoranBenchMark(nDOFDesired: int = 10000,):
	# See Paper: "Utility of superposition-based finite element ..."  by Moran, at. al., Additive Manuf, 2018
    # We have modeled this as a static problem here. For transient see the transient_thermal.py file
   
	L: float = [0.005, 0.005, 0.002] # See fig 2 in paper
	nVoxelsDesired = nDOFDesired
	# Let the number of voxels be proportional to the length in each direction
	alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
	nelx = round(alpha*L[0])
	nely = round(alpha*L[1])
	nelz = round(alpha*L[2])
	mesh = mesher.Mesher()
	mesh.grid_mesh(num_elems = (nelx, nely, nelz),
								 elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
	mesh.createEdofMatThermal()
	

	x0_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
	xmax_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = xMax plane
	y0_nodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y = 0 plane
	ymax_nodes = mesh.getNodesOnBoundingBoxPlane(1,False) # y = yMax plane 
	zmax_nodes = mesh.getNodesOnBoundingBoxPlane(2,True) # z = 0 plane

	fixed_nodes = np.union1d(x0_nodes, np.union1d(xmax_nodes, np.union1d(y0_nodes, 
                            np.union1d(ymax_nodes, zmax_nodes))))

	fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
	dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
	mesh.node_indices[fixed_nodes, 3] = 1
	# see Fig 2 in Paper for the heat load
	nSamples = 100
	xStart = 0.0025
	xWidth = 0.00238
	x = np.linspace(xStart, xStart + xWidth, nSamples)
	line_locs = np.column_stack((x, 0.002275*np.ones_like(x), 0.002*np.ones_like(x)))
	line_nodes = mesh.get_nodes_from_locations(line_locs)
	load_dofs = line_nodes   
	mesh.node_indices[line_nodes, 3] = 2
	Q = 1 # total heat load
	load_per_dof = Q/len(line_nodes)

	force = np.zeros(mesh.num_nodes)
	force[load_dofs] = load_per_dof

	bc = bound_cond.BC(force = force,
						fixed_dofs = fixed_dofs,
						dirichlet_values = dirichlet_values) 

	mesh.translate(0, 0, -0.002)
    # see Table 1 in Paper 
	mat_prop = mat_lib.ThermalMaterial(thermal_conductivity = 27, mass_density = 4420,specific_heat = 750,)
	return mesh, mat_prop, bc

def createLBracketThermalProblem(nDOFDesired: int = 10000,thermal_conductivity = 45, heat_load = 1000):
    """Creates a thermal problem setup for an L-bracket topology optimization.
    This function sets up a finite element mesh and boundary conditions for an L-bracket
    thermal problem from an STL file. The mesh is created with approximately the desired
    number of degrees of freedom. The problem includes fixed temperature boundary conditions
    on the top surface and a heat load on a portion of the right surface.

    Args:
        nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                                    Defaults to 10000.
    Returns:
        tuple: A tuple containing:
            - mesh (Mesher): Mesh object with the L-bracket discretization
            - mat_prop (ThermalMaterial): Material properties object with thermal parameters
            - bc (BC): Boundary conditions object with heat loads and temperature constraints

    Notes:
        - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
        - Fixed temperature boundary condition (T=0) applied at y = yMax
        - Heat load is applied where y > 0.039 and x > 0.09
        - Total heat load is 1000 W distributed equally among loaded nodes
        - Material properties are set to k = 45 W/mK
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
    nElemsDesired = nDOFDesired	# estimate
    mesh = mesher.Mesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    
    mesh.createEdofMatThermal()

    node_pts = mesh.node_indices[:, :3]*mesh.elem_size +mesh.origin
	
    fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    load_nodes = np.where((node_pts[:, 1] > 0.039) & (node_pts[:, 0] > 0.09))[0] # hard coded	
    load_dofs = load_nodes
    mesh.node_indices[load_nodes, 3] = 2 # for plotting
    totalHeat= heat_load

    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = totalHeat/len(load_nodes)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity)
    return mesh, mat_prop, bc

if __name__ == "__main__":
    import plots
    import thermal_fea as fea
    import linear_solvers as lin_solv
    import jax # import jax to enable 64 bit precision
    import time	

    jax.config.update("jax_enable_x64", True)
    example = 2
    if example == 1:
        mesh, mat_prop, bc = createLBracketThermalProblem(nDOFDesired=10000,thermal_conductivity=45, heat_load=1000)    
    else:   
        mesh, mat_prop, bc = createMoranBenchMark(nDOFDesired=100000)
    
    fe_solver = fea.ThermalFEA(mesh = mesh,
                          mat_prop = mat_prop,
                          bc = bc,
                          solver = lin_solv.Solvers.PARDISO)

    thermal_conductivity = np.ones((fe_solver.mesh.num_elems,)) # This is really material scaling for SIMP
    startTime = time.time()
    u = np.asarray(fe_solver.solve(elem_conductivity= thermal_conductivity))
    
    uMax = np.max(np.abs(u))
    nDOF = fe_solver.mesh.num_nodes
   
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max u: ', uMax)
    print('-----------------------------')
	
    plots.plotMesh(fe_solver.mesh, None, u,title=f'Dof = {nDOF}, max u: {uMax:.3e}',show_edges=False,)
