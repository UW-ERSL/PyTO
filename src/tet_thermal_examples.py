import numpy as np
import os
from tet_mesher import TetMesher
import mat_lib
import bound_cond
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class TetThermalExamples(enum.Enum):
	ThickPlate = enum.auto()
	AnnularPlate = enum.auto()
	LBracket = enum.auto()

def getTetThermalProblem(problem: TetThermalExamples, **kwargs):
  """Returns a structural problem based on the given problem name.

  Parameters:
  ----------
  problem : StructuralExamples
    The name of the problem to return.
  **kwargs : dict
    Additional keyword arguments to pass to the problem creation function.

  Returns:
  -------
  tuple
    A tuple containing the mesh, material properties, and boundary conditions for the problem.
  """
  if problem == TetThermalExamples.ThickPlate:
    return createThickPlateThermalProblemTet(**kwargs)
  elif problem == TetThermalExamples.AnnularPlate:
    return createAnnularPlateThermalProblemTet(**kwargs)
  elif problem == TetThermalExamples.LBracket:
    return createLBracketThermalProblemTet(**kwargs)
  else:
    raise ValueError("Invalid structural tet example name.")
  


def createThickPlateThermalProblemTet(nDOFDesired: int = 10000,thermal_conductivity = 50, heat_load = 100000):
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
        - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
        - Fixed temperature boundary condition (T=0) applied at y = yMax
        - Heat load is applied where y > 0.039 and x > 0.09
        - Total heat load is 1000 W distributed equally among loaded nodes
        - Material properties are set to k = 45 W/mK
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    nElemsDesired = nDOFDesired*3	# estimate
    tetmesh = TetMesher()
	
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    

    fixed_nodes = np.where(tetmesh.node_xyz[:, 0] == np.min(tetmesh.node_xyz[:, 0]) )[0] # x = xMin plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values =23*np.ones_like(fixed_dofs, dtype = float)
    load_nodes = np.where(tetmesh.node_xyz[:, 0] == np.max(tetmesh.node_xyz[:, 0]) )[0] # x = xMax plane
    tri_surface_indices = tetmesh.get_surface_triangles_with_all_nodes_in_node_set(load_nodes)
    #tri_surface_indices =  tetmesh.get_surface_triangles_on_bounding_box(axis_dir = 0,min_plane = False)
 
    force = tetmesh.integrate_over_surface_triangles(heat_load, tri_surface_indices)
   
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity,mass_density=7850,specific_heat=500)
    return tetmesh, mat_prop, bc


def createAnnularPlateThermalProblemTet(nDOFDesired: int = 10000,thermal_conductivity = 50, heat_load = 100):
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
        - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
        - Fixed temperature boundary condition (T=0) applied at y = yMax
        - Heat load is applied where y > 0.039 and x > 0.09
        - Total heat load is 1000 W distributed equally among loaded nodes
        - Material properties are set to k = 45 W/mK
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')
    nElemsDesired = nDOFDesired*3	# estimate
    tetmesh = TetMesher()
	
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)

    centerPt = [0,0,0]
    axis = [0,0,1]
    innerRadius = 0.01
    fixed_nodes = tetmesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-tetmesh.elem_size*0.01,
                                                     innerRadius+tetmesh.elem_size*0.01) 
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  

    outerRadius = 0.05
    load_nodes = tetmesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-tetmesh.elem_size*0.1,
                                                    outerRadius+tetmesh.elem_size*0.1)  
    tri_surface_indices = tetmesh.get_surface_triangles_with_all_nodes_in_node_set(load_nodes)
    # tri_surface_indices =  tetmesh.get_surface_triangles_within_annular_region(centerPt,axis,outerRadius-tetmesh.elem_size*0.1,
    #                                                 outerRadius+tetmesh.elem_size*0.1)
   
    force = tetmesh.integrate_over_surface_triangles(heat_load, tri_surface_indices)
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity,mass_density=7850,specific_heat=500)
    return tetmesh, mat_prop, bc

def createLBracketThermalProblemTet(nDOFDesired = 10000,thermal_conductivity = 50, heat_load = 10):
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
        - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
        - Fixed temperature boundary condition (T=0) applied at y = yMax
        - Heat load is applied where y > 0.039 and x > 0.09
        - Total heat load is 1000 W distributed equally among loaded nodes
        - Material properties are set to k = 45 W/mK
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')

    tetmesh = TetMesher()
    nElemsDesired = 3*nDOFDesired # estimate
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired)
    #tetmesh.plot()

    yMax = np.max(tetmesh.node_xyz[:, 1])
    fixed_nodes = np.where(tetmesh.node_xyz[:, 1] == yMax )[0] 
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)


    load_nodes = np.where(tetmesh.node_xyz[:, 0] == np.max(tetmesh.node_xyz[:, 0]) )[0] # x = xMax plane
    tri_surface_indices = tetmesh.get_surface_triangles_with_all_nodes_in_node_set(load_nodes)

    force = tetmesh.integrate_over_surface_triangles(heat_load, tri_surface_indices)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity,mass_density=7850,specific_heat=500)
    return tetmesh, mat_prop, bc
