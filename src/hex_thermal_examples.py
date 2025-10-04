import numpy as np
import mat_lib
import bound_cond
import hex_mesher
import mat_lib
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class HexThermalExamples(enum.Enum):
    HeatPlate = enum.auto()
    FourCornersThermal = enum.auto()
    BridgeThermal = enum.auto()
    ThickPlate = enum.auto()
    AnnularPlate = enum.auto()
    LBracket = enum.auto()
    Moran = enum.auto()
    BliskWithBlade = enum.auto()

def getThermalProblem(problem: HexThermalExamples, **kwargs):
  """Returns a thermal problem based on the given problem name.

  Parameters:
  ----------
  problem : ThermalExamples
    The name of the problem to return.
  **kwargs : dict
    Additional keyword arguments to pass to the problem creation function.

  Returns:
  -------
  tuple
    A tuple containing the mesh, material properties, and boundary conditions for the problem.
  """
  print("problem", problem)
  if problem == HexThermalExamples.HeatPlate:
    return createHeatPlateThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.FourCornersThermal:
    return createFourCornersThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.BridgeThermal:
    return createBridgeThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.LBracket:
    return createLBracketThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.LBracket:
    return createLBracketThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.ThickPlate:
    return createThickPlateThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.AnnularPlate:
    return createAnnularPlateThermalProblem(**kwargs)
  
  elif problem == HexThermalExamples.Moran:
    return createMoranBenchMark(**kwargs)
  
  elif problem == HexThermalExamples.BliskWithBlade:
    return createBliskWithBladeProblem(**kwargs)
  
  else:
    raise ValueError("Invalid thermal example name.")



def createHeatPlateThermalProblem(nDOFDesired: int = 25000,
                                         heat_load = 1, T0 = 23):
    """
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/HeatPlate/HeatPlate.STL')
    nElemsDesired = nDOFDesired //3	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatThermal()
    node_pts = mesh.node_xyz
	
    fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
  
    load_nodes = np.where((node_pts[:, 1] == np.min(node_pts[:, 1])) & 
                (node_pts[:, 0] >= 0.475) & 
                (node_pts[:, 0] <= 0.525))[0] 
    
    load_dofs = load_nodes
    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = heat_load/len(load_nodes)

    dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force


def createFourCornersThermalProblem(nDOFDesired: int = 25000,T0 = 100):
    """
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/FourCornersThermal/FourCornersThermal.STL')
    nElemsDesired = nDOFDesired //3	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatThermal()
    node_pts = mesh.node_xyz
	
    tol = 1e-6

    xmin = np.min(node_pts[:, 0])
    xmax = np.max(node_pts[:, 0])
    ymin = np.min(node_pts[:, 1])
    ymax = np.max(node_pts[:, 1])
    
    fixed_nodes_1 = np.where((np.abs(node_pts[:, 0] - xmin) < tol) & 
                 (node_pts[:, 1] < 0.04))[0].astype(int)  # x = xmin and y < 0.1
    
    fixed_nodes_2 = np.where((np.abs(node_pts[:, 1] - ymax) < tol) & 
                 (node_pts[:, 0] < 0.04))[0].astype(int)  # y = ymax and x < 0.1
     
    fixed_nodes_3 = np.where((np.abs(node_pts[:, 0] - xmax) < tol) & 
                 (node_pts[:, 1] > 0.06))[0].astype(int)  # x = xmax and y > 0.9
    
    fixed_nodes_4 = np.where((np.abs(node_pts[:, 1] - ymin) < tol) & 
                 (node_pts[:, 0] > 0.06))[0].astype(int)  # y = ymin and x > 0.9
  

    fixed_dofs = np.concatenate([fixed_nodes_1, fixed_nodes_2, fixed_nodes_3, fixed_nodes_4]).astype(int)
    dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

    dirichlet_values[0:len(fixed_nodes_1)] = 0
    dirichlet_values[len(fixed_nodes_1):len(fixed_nodes_1)+len(fixed_nodes_2)] = T0
    dirichlet_values[len(fixed_nodes_1)+len(fixed_nodes_2):len(fixed_nodes_1)+len(fixed_nodes_2)+len(fixed_nodes_3)] = 0 
    dirichlet_values[len(fixed_nodes_1)+len(fixed_nodes_2)+len(fixed_nodes_3):] = T0
  

    force = np.zeros(mesh.num_nodes)
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force



def createBridgeThermalProblem(nDOFDesired: int = 25000):
    """
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/BridgeThermal/BridgeThermal.STL')
    nElemsDesired = nDOFDesired //3	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatThermal()
    node_pts = mesh.node_xyz
	
    tol = 1e-6
    xmin = np.min(node_pts[:, 0])
    xmax = np.max(node_pts[:, 0])
    fixed_nodes_1 = np.where((np.abs(node_pts[:, 0] - xmin) < tol) & 
                 (node_pts[:, 1] < 0.02))[0] # x = xMin within tolerance and y < 0.2
    fixed_dofs_1 = np.array([fixed_nodes_1]).flatten().astype(int)

    fixed_nodes_2 = np.where((np.abs(node_pts[:, 0] - xmax) < tol) & 
                 (node_pts[:, 1] < 0.02))[0] # x = xMin within tolerance and y < 0.2
    
    fixed_dofs_2 = np.array([fixed_nodes_2]).flatten().astype(int)


    fixed_dofs = np.concatenate([fixed_nodes_1, fixed_nodes_2]).astype(int)
    dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

    dirichlet_values[0:len(fixed_nodes_1)] = 0
    dirichlet_values[len(fixed_nodes_1):len(fixed_nodes_1)+len(fixed_nodes_2)] = 100
    force = np.zeros(mesh.num_nodes)
 
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force



def createThickPlateThermalProblem(nDOFDesired: int = 10000, heat_load = 1000, T0 = 23):
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
    nElemsDesired = nDOFDesired	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    
    mesh.createEdofMatThermal()

    node_pts = mesh.node_indices[:, :3]*mesh.elem_size +mesh.origin
	
    fixed_nodes = np.where(node_pts[:, 0] == np.min(node_pts[:, 0]) )[0] # x = xMin plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
  
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    load_nodes = np.where(node_pts[:, 0] == np.max(node_pts[:, 0]) )[0] # x = xMax plane
    load_dofs = load_nodes
    mesh.node_indices[load_nodes, 3] = 2 # for plotting
    totalHeat= heat_load

    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = totalHeat/len(load_nodes)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force


def createAnnularPlateThermalProblem(nDOFDesired: int = 10000, heat_load = 100, T0 = 23):
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
    nElemsDesired = nDOFDesired	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    
    mesh.createEdofMatThermal()
    centerPt = [0,0,0]
    axis = [0,0,1]
    innerRadius = 0.01
    fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707) 
	
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
  
 

    outerRadius = 0.05
    load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
    load_dofs = load_nodes
    totalHeat= heat_load

    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = totalHeat/len(load_nodes)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force


def createLBracketThermalProblem(nDOFDesired: int = 10000, heat_load = 10, T0 = 23):
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
    nElemsDesired = nDOFDesired	# estimate
    mesh = hex_mesher.HexMesher()
	
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)

    mesh.createEdofMatThermal()

    node_pts = mesh.node_xyz
	
    fixed_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]) )[0] # y = yMax plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    
    dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
  
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    load_nodes = np.where(node_pts[:, 0] == np.max(node_pts[:, 0]) )[0] # x = xMax plane
    load_dofs = load_nodes
    mesh.node_indices[load_nodes, 3] = 2 # for plotting
    totalHeat= heat_load

    force = np.zeros(mesh.num_nodes)
    force[load_dofs] = totalHeat/len(load_nodes)

    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force

def createBliskWithBladeProblem(nDOFDesired: int = 10000, heat_load = 10, T0 = 23):
   # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskSectionWithBlade.STL')

  nElemsDesired = nDOFDesired    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatThermal()
  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01085
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
  dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)

  bladeStartRadius = 0.055
  bladeEndRadius = 0.07
 
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,bladeStartRadius,
                                                     bladeEndRadius)  
  load_dofs = load_nodes

  totalHeat= heat_load

  force = np.zeros(mesh.num_nodes)
  force[load_dofs] = totalHeat/len(load_nodes)
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

def createMoranBenchMark(nDOFDesired: int = 10000,T0 = 23):
	# See Paper: "Utility of superposition-based finite element ..."  by Moran, at. al., Additive Manuf, 2018
    # We have modeled this as a static problem here. For transient see the transient_thermal.py file
   
	L: float = [0.005, 0.005, 0.002] # See fig 2 in paper
	nVoxelsDesired = nDOFDesired
	# Let the number of voxels be proportional to the length in each direction
	alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
	nelx = round(alpha*L[0])
	nely = round(alpha*L[1])
	nelz = round(alpha*L[2])
	mesh = hex_mesher.HexMesher()
	mesh.grid_mesh(num_elems = (nelx, nely, nelz),
								 elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
	mesh.createEdofMatThermal()
	
	x0_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
	xmax_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
	y0_nodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y = 0 plane
	ymax_nodes = mesh.getNodesOnBoundingBoxPlane(1,False) # y = yMax plane 
	zmax_nodes = mesh.getNodesOnBoundingBoxPlane(2,True) # z = 0 plane
	# apply Dirichelt on xMax nodes
	fixed_nodes = np.union1d(x0_nodes, np.union1d(xmax_nodes, np.union1d(y0_nodes, np.union1d(ymax_nodes, zmax_nodes))))
	# don't apply Dirichelt on xMax nodes
	#fixed_nodes = np.union1d(x0_nodes,  np.union1d(y0_nodes,np.union1d(ymax_nodes, zmax_nodes)))
	   
	fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
	dirichlet_values = T0*np.ones_like(fixed_dofs, dtype = float)
	mesh.node_indices[fixed_nodes, 3] = 1
	# see Fig 2 in Paper for the heat load
	
	xStart = 0.0025
	xWidth = 0.00238
	nSamples = 2*int(xWidth/mesh.elem_size[0])
	x = np.linspace(xStart, xStart + xWidth, nSamples)
	line_locs = np.column_stack((x, 0.002275*np.ones_like(x), 0.002*np.ones_like(x)))
	line_nodes = np.unique(mesh.get_nodes_from_locations(line_locs))
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
	mat_prop = mat_lib.get_material("Steel")
	elem_body_force = None
	return mesh, mat_prop, bc, elem_body_force
