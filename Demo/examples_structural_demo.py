import numpy as np
import mat_lib
import bound_cond
import mesher
import mat_lib
import os
import enum
from mat_xml_parser import *
script_dir = os.path.dirname(os.path.abspath(__file__))


class StructuralExamplesDemo(enum.Enum):
	KnuckleAssemblyDemo = enum.auto()
	NoseCone = enum.auto()
	NoseConeAnglularSym = enum.auto()
	BasePlate = enum.auto()
	EdgeCantileverDemo = enum.auto() 
	BridgeDemo = enum.auto()
	LongBeamDemo = enum.auto()
	SimpleBracketDemo = enum.auto()
	LongBeamTopBottomLoadDemo = enum.auto()
  
  
  


def getStructuralProblem(problem: StructuralExamplesDemo, **kwargs):
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
  if problem == StructuralExamplesDemo.NoseCone:
    return createNoseconeProblem(**kwargs)
  elif problem == StructuralExamplesDemo.NoseConeAnglularSym:
    return createNoseconeProblemZAxisAngularSymmetry(**kwargs)
  elif problem == StructuralExamplesDemo.BasePlate:
    return createBasePlateProblem(**kwargs)
  elif problem == StructuralExamplesDemo.EdgeCantileverDemo:
    return createEdgeCantileverDemoProblem(**kwargs)
  elif problem == StructuralExamplesDemo.BridgeDemo:
    return createBridgeDemoProblem(**kwargs)
  elif problem == StructuralExamplesDemo.LongBeamDemo:
    return createLongBeamDemoProblem(**kwargs)
  elif problem == StructuralExamplesDemo.SimpleBracketDemo:
    return createSimpleBracketDemoProblem(**kwargs)
  elif problem == StructuralExamplesDemo.LongBeamTopBottomLoadDemo:
    return createLongBeamTopBottomLoadDemoProblem(**kwargs)
  else:
    raise ValueError("Invalid structural example name.")
  

def createKnuckleAssemblyProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                               poissons_ratio = 0.28, totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/KnuckleAssembly/KnuckleAssembly.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  fixed_nodes_1 = np.where(node_pts[:, 0] == np.min(node_pts[:, 0]))[0]
  fixed_nodes_2 = np.where(node_pts[:, 0] == np.max(node_pts[:, 0]))[0]
  fixed_nodes = np.union1d(fixed_nodes_1,fixed_nodes_2)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]))[0]       
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createTableProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/Table/Table.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  fixed_nodes = np.where(node_pts[:, 1] == np.min(node_pts[:, 1]))[0]

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]))[0]       
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force



def createNoseconeProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/HollowNoseConeWithSolidBaseNew.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  
  load_nodes = np.where(node_pts[:, 1] > 0.033)[0]       
  
  #mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  
  # fix inner radius
  centerPt = [0.06502,0,0.06502]
  axis = [0,1,0]
  outerRadius = 0.06
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*1.5,
                                                     outerRadius+mesh.elem_size[0]*1.5)  
  
  nodes_to_optimize = np.where(node_pts[:, 1] > 0.01)[0]       
  
  fixed_nodescrossX = np.where((node_pts[:, 0] > 0.06345) & (node_pts[:, 0] < 0.07155))[0]
  fixed_nodescrossZ = np.where((node_pts[:, 2] > 0.06345) & (node_pts[:, 2] < 0.07155))[0]

  fixed_nodes = np.union1d(fixed_nodes, fixed_nodescrossX)
  fixed_nodes = np.union1d(fixed_nodes, fixed_nodescrossZ)
  fixed_nodes = np.setdiff1d(fixed_nodes, nodes_to_optimize)

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  #

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force



def createNoseconeProblemZAxisAngularSymmetry(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/SolidNoseConeForZAngularSymmetry.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz

  # mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 2] > 0.3)[0]       
  
  #mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 2  # z direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  
  # fix inner radius
  centerPt = [0.08131,0.08131,0.0]
  axis = [0,0,1]
  outerRadius = 0.06
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*1.0,
                                                     outerRadius+mesh.elem_size[0]*2.1)  
  
  nodes_to_optimize = np.where(node_pts[:, 2] > 0.05)[0]       
  
  fixed_nodes = np.setdiff1d(fixed_nodes, nodes_to_optimize)

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  #

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createBasePlateProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, 'C:/Users/pthombre/Downloads/RocketPy_PyTO/Models/Rocket/PayloadBaseRecreatedForDemoSTL.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz

  # fix inner radius
  centerPt = [0,63,63.5]
  axis = [1,0,0]
  innerRadius = 20  # 20 mm
  outerRadius = 65  # 65 mm
  outerBottomRadius = 60  # 60 mm
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerBottomRadius-mesh.elem_size[0]*3,
                                                     outerBottomRadius+mesh.elem_size[0]*0.907)  
  
  nodes_to_optimize = np.where(node_pts[:, 0] >= 10)[0]       
  
  fixed_nodes = np.setdiff1d(fixed_nodes, nodes_to_optimize)

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  # line defined by x = xMax
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
  load_dofs = 3 * load_nodes  # z direction
  #mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof 
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createEdgeCantileverDemoProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
                                youngs_modulus = 2.1e11, poissons_ratio = 0.28,totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = mesher.Mesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and y = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  # Get material data from XML if it exists
  fp_materialXML = os.path.join(script_dir, './material.xml')
  if fp_materialXML is not None and os.path.isfile(fp_materialXML):
      youngs_modulus, poissons_ratio = parse_material_properties(fp_materialXML)

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createBridgeDemoProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
                                youngs_modulus = 2.1e11, poissons_ratio = 0.28,totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = mesher.Mesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodesX0 = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_nodesXMax = mesh.getNodesOnBoundingBoxPlane(0,False) # x = 0 plane
  fixed_nodes = np.union1d(fixed_nodesX0, fixed_nodesXMax)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and y = 0 
  load_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  # Get material data from XML if it exists
  fp_materialXML = os.path.join(script_dir, './material.xml')
  if fp_materialXML is not None and os.path.isfile(fp_materialXML):
      youngs_modulus, poissons_ratio = parse_material_properties(fp_materialXML)

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createLongBeamDemoProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
                                youngs_modulus = 2.1e11, poissons_ratio = 0.28,totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = mesher.Mesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodesX0 = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_nodesXMax = mesh.getNodesOnBoundingBoxPlane(0,False) # x = 0 plane
  fixed_nodes = np.union1d(fixed_nodesX0, fixed_nodesXMax)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by at plane y = 0 and between 45 to 55 % of the length in x direction
  load_nodesonY0 = mesh.getNodesOnBoundingBoxPlane(1, True)
  load_nodesonYMax = mesh.getNodesOnBoundingBoxPlane(1, False)
  load_nodesY = np.union1d(load_nodesonY0, load_nodesonYMax)

  x_coords = mesh.node_xyz[load_nodesY, 0]
  mask = (x_coords > 0.45 * L[0]) & (x_coords < 0.55 * L[0])
  load_nodes = load_nodesY[mask]


  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  # Get material data from XML if it exists
  fp_materialXML = os.path.join(script_dir, './material.xml')
  if fp_materialXML is not None and os.path.isfile(fp_materialXML):
      youngs_modulus, poissons_ratio = parse_material_properties(fp_materialXML)

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createSimpleBracketDemoProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = '../Models/SimpleBracket/SimpleBracketmm.STL'

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
  #mesh.plot()

  # fix inner radius
  centerPt = [0.0077,0.01777,0.003]
  axis = [0,0,1]
  outerBottomRadius = 0.003  # 3 mm
  fixed_nodes1 = mesh.get_nodes_within_annular_region(centerPt,axis,outerBottomRadius-mesh.elem_size[0]*2.707,
                                                     outerBottomRadius+mesh.elem_size[0]*0.907)  
  
  #nodes_to_optimize = np.where(node_pts[:, 0] >= 10)[0]       
  
  # fix inner radius
  centerPt = [0.0077,0.04,0.003]
  axis = [0,0,1]
  outerBottomRadius = 0.003  
  fixed_nodes2 = mesh.get_nodes_within_annular_region(centerPt,axis,outerBottomRadius-mesh.elem_size[0]*2.707,
                                                     outerBottomRadius+mesh.elem_size[0]*0.907)  
  
  fixed_nodes = np.union1d(fixed_nodes1, fixed_nodes2)

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  # line defined by x = xMax
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
  load_dofs = 3 * load_nodes + 1  # y direction
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof 
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createLongBeamTopBottomLoadDemoProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
                                youngs_modulus = 2.1e11, poissons_ratio = 0.28,totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = '../Models/EdgeCantilever/EdgeCantilever.STL'

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodesX0 = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_nodesXMax = mesh.getNodesOnBoundingBoxPlane(0,False) # x = 0 plane
  fixed_nodes = np.union1d(fixed_nodesX0, fixed_nodesXMax)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by at plane y = 0 and between 45 to 55 % of the length in x direction
  load_nodesonY0 = mesh.getNodesOnBoundingBoxPlane(1, True)
  load_nodesonYMax = mesh.getNodesOnBoundingBoxPlane(1, False)
  load_nodes = np.union1d(load_nodesonY0, load_nodesonYMax)

  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  # Get material data from XML if it exists
  fp_materialXML = os.path.join(script_dir, './material.xml')
  if fp_materialXML is not None and os.path.isfile(fp_materialXML):
      youngs_modulus, poissons_ratio = parse_material_properties(fp_materialXML)

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
