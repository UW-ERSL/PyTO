import numpy as np
import mat_lib
import bound_cond
import hex_mesher
import mat_lib
import os
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))


class StructuralExamples(enum.Enum):
	TensileBar = enum.auto()
	TorsionBar = enum.auto()
	BeamBending = enum.auto()
	ShearBlock = enum.auto()
	Mitchell = enum.auto()
	EdgeCantilever = enum.auto()
	ShortCantileverTipLoad = enum.auto()
	ShortCantileverMidLoad = enum.auto()
	CantileverTipLoad = enum.auto()
	CantileverMidLoad = enum.auto()
	TwoBar = enum.auto()
	ThreeHoleBracket = enum.auto()
	MBBB = enum.auto()
	DistributedLoad = enum.auto()
	Multiload = enum.auto()
	GravityBar = enum.auto() 
	GravityPlate = enum.auto()
	LBracket = enum.auto()
	ArrowHead = enum.auto()
	CompliantMechanism = enum.auto()
	FilletedBeam = enum.auto()
	BeamSurfaceLoad = enum.auto()
	CentrifugalPlate = enum.auto()
	TorquePlate = enum.auto()
	KnuckleAssembly =  enum.auto()
	Table =  enum.auto()
	LBracketThick = enum.auto()
	BliskQuarter = enum.auto()
	BliskWithBlade =  enum.auto()
	NoseCone = enum.auto()
	


def getStructuralProblem(problem: StructuralExamples, **kwargs):
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
  if problem == StructuralExamples.TensileBar:
    return createTensileBarProblem(**kwargs)
  elif problem == StructuralExamples.TorsionBar:
    return createTorsionBarProblem(**kwargs)
  elif problem == StructuralExamples.BeamBending:
    return createBeamBendingProblem(**kwargs)
  elif problem == StructuralExamples.ShearBlock:
    return createShearBlockProblem(**kwargs)
  elif problem == StructuralExamples.Mitchell:
    return createMitchellProblem(**kwargs)
  elif problem == StructuralExamples.EdgeCantilever:
    return createEdgeCantileverProblem(**kwargs)
  elif problem == StructuralExamples.ShortCantileverTipLoad:
    return createShortCantileverTipLoadProblem(**kwargs)
  elif problem == StructuralExamples.ShortCantileverMidLoad:
    return createShortCantileverMidLoadProblem(**kwargs)
  elif problem == StructuralExamples.CantileverTipLoad:
    return createCantileverTipLoadProblem(**kwargs)
  elif problem == StructuralExamples.CantileverMidLoad:
    return createCantileverMidLoadProblem(**kwargs)
  elif problem == StructuralExamples.MBBB:
    return createMBBBProblem(**kwargs)
  elif problem == StructuralExamples.LBracket:
    return createLBracketProblem(**kwargs)
  elif problem == StructuralExamples.TwoBar:
    return createTwoBarProblem(**kwargs)
  elif problem == StructuralExamples.DistributedLoad:
    return createDistributedLoadProblem(**kwargs)
  elif problem == StructuralExamples.ThreeHoleBracket:
    return createThreeHoleBracketProblem(**kwargs)
  elif problem == StructuralExamples.Multiload:
    return createMultiloadProblem(**kwargs)
  elif problem == StructuralExamples.GravityBar:
    return createGravityBarProblem(**kwargs)
  elif problem == StructuralExamples.GravityPlate:
    return createGravityPlateProblem(**kwargs)
  elif problem == StructuralExamples.CompliantMechanism:
    return createCompliantMechanismProblem(**kwargs)
  elif problem == StructuralExamples.FilletedBeam:
    return createFilletedBeamProblem(**kwargs)
  elif problem == StructuralExamples.BeamSurfaceLoad:
    return createBeamSurfaceLoadProblem(**kwargs)
  elif problem == StructuralExamples.CentrifugalPlate:
    return createCentrifugalPlateProblem(**kwargs)
  elif problem == StructuralExamples.TorquePlate:
    return createTorquePlateProblem(**kwargs)
  elif problem == StructuralExamples.BliskQuarter:
    return createBliskQuarterProblem(**kwargs)
  elif problem == StructuralExamples.BliskWithBlade:
    return createBliskSectionWithBlade(**kwargs)
  elif problem == StructuralExamples.LBracketThick:
    return createLBracketThickProblem(**kwargs)
  elif problem == StructuralExamples.KnuckleAssembly:
    return createKnuckleAssemblyProblem(**kwargs)
  elif problem == StructuralExamples.Table:
    return createTableProblem(**kwargs)
  elif problem == StructuralExamples.ArrowHead:
    return createArrowHeadProblem(**kwargs)
  elif problem == StructuralExamples.NoseCone:
    return createNoseconeProblem(**kwargs)
  else:
    raise ValueError("Invalid structural example name.")
  

def createTensileBarProblem(nDOFDesired: int = 10000, L: float = [10, 1, 1],youngs_modulus = 2e11, 
                            poissons_ratio = 0.3,tensileForce = 10000):
  """Creates a tensile problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
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
  print('-----------------------------')
  print("Theoretical max displacement: {:.2g}".format(tensileForce*L[0]/(youngs_modulus*L[1]*L[2])))
  print("Theoretical max stress: {:.2g}".format(tensileForce/(L[1]*L[2])))
  print('-----------------------------')
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes]).flatten().astype(int) # fixed in x direction
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  # Add forces on x=xMax plane with different magnitudes for edges and corners
  # that is consistent with numerical integration over the surface
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
  # Find edge and corner nodes
  edge_nodes = []
  corner_nodes = []
  face_nodes = [] 

  for node in load_nodes:
    coords = mesh.node_xyz[node]
    num_extremes = 0
    if abs(coords[1] - min(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2 or abs(coords[1] - max(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2:
      num_extremes += 1
    if abs(coords[2] - min(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2 or abs(coords[2] - max(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2:
      num_extremes += 1
    
    if num_extremes == 2:
      corner_nodes.append(node)
    elif num_extremes == 1:
      edge_nodes.append(node) 
    else:
      face_nodes.append(node)
  
  # Split into lists for clarity
  corner_nodes = np.array(corner_nodes)
  edge_nodes = np.array(edge_nodes)
  face_nodes = np.array(face_nodes)
  # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  force[3*face_nodes] = 4.0  
  force[3*edge_nodes] = 2.0  
  force[3*corner_nodes] = 1.0 
  
  # Normalize forces to achieve desired total load
  total_load = np.sum(force[3*load_nodes])
  force[3*load_nodes] *= tensileForce/total_load
  print("Total force:", np.sum(force[3*load_nodes]))
  load_nodes = np.union1d(np.union1d(face_nodes, edge_nodes), corner_nodes)
  mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

def createTorsionBarProblem(nDOFDesired: int = 10000, L: float = [1, 0.2, 0.2], 
                youngs_modulus = 2e7, poissons_ratio = 0.3, totalLoad = 1000):
    """Creates a torsion bar problem with approximate desired DOFs.

    Parameters:
    ----------
    nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
    L : list of float  
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
    youngs_modulus : float
    Young's modulus of material (default 2e11)
    poissons_ratio : float
    Poisson's ratio of material (default 0.3)
    torque : float
    Applied torque magnitude (default 1000)

    Returns:
    -------
    tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and torque on right face
    """
    nVoxelsDesired = nDOFDesired/3    
    # Let the number of voxels be proportional to the length in each direction
    alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
    nelx = round(alpha*L[0])
    nely = round(alpha*L[1])
    nelz = round(alpha*L[2])
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems = (nelx, nely, nelz),
            elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
    mesh.createEdofMatStructural()

    # Fix all DOFs at x=0 plane
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True)
    fixed_dofs = np.array([3 * fixed_nodes,
          3 * fixed_nodes + 1,
          3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
    mesh.node_indices[fixed_nodes, 3] = 1

    # Get nodes on x=xMax plane for applying torque
    load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
    load_dofs = []
    force_values = []

    # Calculate center of right face
    right_face_nodes = mesh.node_xyz[load_nodes]
    face_center = np.mean(right_face_nodes, axis=0)
    # Apply torque forces on each node
    for node in load_nodes:
    # Get node position relative to face center
      node_pos = mesh.node_xyz[node] - face_center
      # Calculate radius in yz plane
      r = np.sqrt(node_pos[1]**2 + node_pos[2]**2)
     
      # Unit vector in tangential direction
      tangent_dir = np.array([0, -node_pos[2], node_pos[1]])/r
      # Force magnitude proportional to radius
      force_mag = totalLoad / (len(load_nodes) * r)
      # Add force components
      load_dofs.extend([3*node + i for i in range(3)])
      force_values.extend(force_mag * tangent_dir)

    mesh.node_indices[load_nodes, 3] = 2

    force = np.zeros(3*mesh.num_nodes)
    force[load_dofs] = force_values

    bc = bound_cond.BC(force = force,
        fixed_dofs = fixed_dofs,
        dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
              poissons_ratio=poissons_ratio)
    
    elem_body_force = None

    return mesh, mat_prop, bc, elem_body_force
  # ----------------------------------------

def createBeamBendingProblem(nDOFDesired: int = 10000, L: float = [10, 1, 1],youngs_modulus = 2e11, 
                            poissons_ratio = 0.3,tensileForce = 10000):
  """Creates a tensile problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
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
  print('-----------------------------')
  # For a beam with rectangular cross-section under end load P
  # Maximum deflection = PL^3/(3EI) where I = (w*h^3)/12
  I = (L[1]*L[2]**3)/12
  print("Theoretical max deflection: {:.2g}".format(tensileForce*L[0]**3/(3*youngs_modulus*I)))
  # Maximum bending stress = My/I where M = PL and y = h/2
  print("Theoretical max stress: {:.2g}".format(tensileForce*L[0]*L[2]/(2*I)))
  print('-----------------------------')
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  # Fix all DOFs at x=0 plane (cantilever)
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  # Add downward forces at x=xMax plane
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
  # Find edge and corner nodes
  edge_nodes = []
  corner_nodes = []
  face_nodes = [] 

  for node in load_nodes:
    coords = mesh.node_xyz[node]
    num_extremes = 0
    if abs(coords[1] - min(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2 or abs(coords[1] - max(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2:
      num_extremes += 1
    if abs(coords[2] - min(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2 or abs(coords[2] - max(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2:
      num_extremes += 1
    
    if num_extremes == 2:
      corner_nodes.append(node)
    elif num_extremes == 1:
      edge_nodes.append(node) 
    else:
      face_nodes.append(node)
  
  # Split into lists for clarity
  corner_nodes = np.array(corner_nodes)
  edge_nodes = np.array(edge_nodes)
  face_nodes = np.array(face_nodes)
  
  # Apply downward forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  force[3*face_nodes + 2] = -4.0  # Apply force in z direction (downward)
  force[3*edge_nodes + 2] = -2.0  
  force[3*corner_nodes + 2] = -1.0 
  
  # Normalize forces to achieve desired total load
  load_nodes = np.union1d(np.union1d(face_nodes, edge_nodes), corner_nodes)
  total_load = np.sum(force[3*load_nodes + 2])
  force[3*load_nodes + 2] *= tensileForce/abs(total_load)
  print("Total force:", np.sum(force[3*load_nodes + 2]))
  mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
 
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createShearBlockProblem(nDOFDesired: int = 10000, L: float = [1, 1, 1],youngs_modulus = 2e11, 
                            poissons_ratio = 0.3,shearForce = 1000):
  """Creates a shear problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
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
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes]).flatten().astype(int) # fixed in x direction
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  # Add forces on x=xMax plane with different magnitudes for edges and corners
  # that is consistent with numerical integration over the surface
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane

  load_dof = 3*load_nodes + 1 # y direction

  # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  force[load_dof] = shearForce/len(load_nodes)


  mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force



def createMitchellProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                            poissons_ratio = 0.26,load1 = 5.6e4, load2 = 0):
  """
  See: Topology Optimization Benchmarks in 2D: Results for Minimum Compliance and Minimum Volume in Planar Stress Problems
S. Ivvan Valdez, et al. Arch Computat Methods Eng (2017) 24:803–839, DOI 10.1007/s11831-016-9190-3
  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/Mitchell/Mitchell.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  left_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  left_dofs = np.array([3 * left_nodes]).flatten().astype(int) # fixed in x direction

  bottom_nodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y = 0 plane
  right_nodes = np.intersect1d(bottom_nodes, np.where(node_pts[:, 0] >= 0.9)[0]) # hard coded
  right_dofs = np.array([3 * right_nodes+1]).flatten().astype(int)# fixed in y direction
  
  fixed_dofs = np.union1d(left_dofs,right_dofs)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[left_nodes, 3] = 1
  mesh.node_indices[right_nodes, 3] = 1
  
  
  load_nodes_1 = np.intersect1d(bottom_nodes, np.where(node_pts[:, 0] <= 0.1))
  load_dof_1 = 3*load_nodes_1 + 1 # y direction

  load_nodes_2 = np.intersect1d(bottom_nodes, np.where((node_pts[:, 0] >= 0.25) & (node_pts[:, 0] <= 0.35))[0])
  load_dof_2 = 3*load_nodes_2 + 1 # y direction

  # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)

  force[load_dof_1] = -load1/len(load_nodes_1)
  force[load_dof_2] = -load2/len(load_nodes_2)

 
  mesh.node_indices[load_nodes_1, 3] = 2
  mesh.node_indices[load_nodes_2, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force


  # ----------------------------------------


def createEdgeCantileverProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],
                                youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 10000):
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
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and z = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(2,True))
  load_dofs = 3 * load_nodes + 2  # z direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

  
def createShortCantileverMidLoadProblem(nDOFDesired: int = 10000,
                               youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/ShortCantilever/ShortCantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.05)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createShortCantileverTipLoadProblem(nDOFDesired: int = 10000, 
                               youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 5.8e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/ShortCantilever/ShortCantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
 
  y_tip_nodes = np.where(abs(node_pts[:,1]) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_tip_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  

def createCantileverMidLoadProblem(nDOFDesired: int = 10000,
                               youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/Cantilever/Cantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.05)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createCantileverTipLoadProblem(nDOFDesired: int = 10000,
                               youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 5.8e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/Cantilever/Cantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
 
  y_tip_nodes = np.where(abs(node_pts[:,1]) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_tip_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createTwoBarProblem(nDOFDesired: int = 10000,
                               youngs_modulus = 2e11, poissons_ratio = 0.3,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
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
  stl_file = os.path.join(script_dir, '../Models/TwoBar/TwoBar.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------



def createMBBBProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                      poissons_ratio = 0.3,load = 2.7e4):
  ''' 
    See: Topology Optimization Benchmarks in 2D: Results for Minimum Compliance and Minimum Volume in Planar Stress Problems
  S. Ivvan Valdez, et al. Arch Computat Methods Eng (2017) 24:803–839, DOI 10.1007/s11831-016-9190-3
'''
  stl_file = os.path.join(script_dir, '../Models/MBBB/MBBB.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  symmetry_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  symmetry_dofs = np.array([3 * symmetry_nodes]).flatten().astype(int)

  right_nodes=np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,True), np.where(mesh.node_xyz[:,0] >= 2.7)[0])
  right_dofs = np.array([3 * right_nodes+1]).flatten().astype(int)
  
  fixed_dofs = np.union1d(symmetry_dofs,right_dofs)
  fixed_nodes = np.union1d(symmetry_nodes,right_nodes)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1

 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,False), np.where(mesh.node_xyz[:,0] <= 0.3)[0])
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -load/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createDistributedLoadProblem(nDOFDesired: int = 10000, L: float = [1.0, 0.5, 0.025],youngs_modulus = 2e11, poissons_ratio = 0.3):
  stl_file = os.path.join(script_dir, '../Models/DistributedLoad/DistributedLoad.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  left_node =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,True))
  right_node =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  fixed_nodes = np.union1d(left_node,right_node)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1

 
  load_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -10000/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createMultiloadProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],youngs_modulus = 2e11, poissons_ratio = 0.3):
  # This is an example where a grid mesh is created, and a structural problem is posed on it.
  # For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  load_node1 = mesh.get_nodes_from_locations([L[0], L[1]/2, L[2]/2])  
  load_dof1 = 3 * load_node1 + 1  
  load_node2 = mesh.get_nodes_from_locations([L[0]/2, L[1], L[2]/2])  
  load_dof2 = 3 * load_node2 + 1  

  force = np.zeros(3*mesh.num_nodes)
  force[load_dof1] = -1000
  force[load_dof2] = -10000
  mesh.node_indices[load_node1, 3] = 2
  mesh.node_indices[load_node2, 3] = 2 # for plotting

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createLBracketProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,topload = 1000,midload = 0):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for an L-bracket
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. The problem includes fixed boundary conditions on the top
  surface and a distributed load on a portion of the right surface.
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)  # y = yMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)
  node_pts = mesh.node_xyz
  if(abs(topload) > 0):
    topload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False) , np.where((node_pts[:, 1] >= 0.36))[0]) # hard coded    
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)

  if(abs(midload) > 0):
    midload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), np.where((node_pts[:, 1] >= 0.18) & (node_pts[:, 1] <= 0.22))[0]) # hard coded    
    midload_dofs = 3 * midload_nodes + 1  
    mesh.node_indices[midload_nodes, 3] = 2 # for plotting
    
    force[midload_dofs] = -midload/len(midload_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createGravityBarProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.28, material_density = 7700):
  """Creates a structural problem setup for a vertical bar under gravity loading.
  This function sets up a finite element analysis problem for a vertical bar by:
  1. Reading an STL file of a vertical bar
  2. Creating a mesh with desired number of degrees of freedom
  3. Setting up fixed boundary conditions at the top surface
  4. Applying gravitational body forces
  Parameters
  ----------
  nDOFDesired : int, optional
    Desired number of degrees of freedom in the mesh (default is 10000)
  youngs_modulus : float, optional
    Young's modulus of the material in Pa (default is 2.1e11)
  poissons_ratio : float, optional  
    Poisson's ratio of the material (default is 0.28)
  material_density : float, optional
    Density of the material in kg/m³ (default is 7700)
  Returns
  -------
  tuple
    A tuple containing:
    - mesh: Mesh object with the discretized geometry
    - mat_prop: Material properties object
    - bc: Boundary conditions object
    - elem_body_force: Array of body forces on elements
  Notes
  -----
  - The STL file must be located at '../Models/VerticalBar/VerticalBar.STL'
  - The problem fixes all DOFs at the top surface (z = zMax plane)
  - Gravity acts in the negative z direction
  - SolidWorks validation shows maximum displacement of 1.7906e-9 m
  """

  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/VerticalBar/VerticalBar.STL')
  print("SolidWorks maximum displacement: 1.7906e-9 m")
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
 
  fixed_nodes =  mesh.getNodesOnBoundingBoxPlane(2,False)  # z = zMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = np.zeros(3*mesh.num_elems)
  elem_body_force[2::3] = -9.81*material_density*np.prod(mesh.elem_size)    # Apply gravity in -z direction to each element

  boundaryForce = np.zeros(3*mesh.num_nodes) # no boundary force
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createGravityPlateProblem(nDOFDesired: int = 10000, L: float = [1.0, 0.5, 0.01],
                               youngs_modulus = 2e11, poissons_ratio = 0.3,material_density = 7700,
                               verticalForcePercent = 0):
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()


  left_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,True))
  right_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  fixed_nodes = np.union1d(left_nodes,right_nodes)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  elem_body_force = np.zeros(3*mesh.num_elems)
  elem_body_force[1::3] = -9.81*material_density*np.prod(mesh.elem_size)    # Apply gravity in -y direction to each element

  verticalForce = verticalForcePercent*np.linalg.norm(elem_body_force)
  boundary_force = np.zeros(3*mesh.num_nodes)
  node_pts = mesh.node_xyz
  # Apply a small force in the middle of the plate
  load_nodes = np.where((np.abs(node_pts[:, 0]-L[0]/2) < mesh.elem_size[0]/2) & (np.abs(node_pts[:, 1] - L[1]/2) <mesh.elem_size[1]/2))[0] # hard coded    
  load_dof = 3 * load_nodes + 1  # y direction
  boundary_force[load_dof] = -verticalForce/len(load_nodes)
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  bc = bound_cond.BC(force = boundary_force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createArrowHeadProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,totalLoad = 1000):
  """Creates a structural problem setup for an arrowhead-shaped bracket.

  This function sets up a finite element analysis problem for an arrowhead bracket by:
  1. Reading an STL file of the arrowhead geometry 
  2. Creating a mesh with desired number of degrees of freedom
  3. Setting fixed boundary conditions on the bottom surface
  4. Applying distributed load on the top surface

  Parameters
  ----------
  nDOFDesired : int, optional
    Desired number of degrees of freedom in the mesh (default is 10000)
  youngs_modulus : float, optional 
    Young's modulus of the material in Pa (default is 2.1e11)
  poissons_ratio : float, optional
    Poisson's ratio of the material (default is 0.3)
  totalLoad : float, optional
    Total applied force in N (default is 1000)

  Returns
  -------
  tuple
    A tuple containing:
    - mesh: Mesh object with the discretized geometry 
    - mat_prop: Material properties object
    - bc: Boundary conditions object

  Notes
  -----
  - The STL file must be located at '../Models/ArrowHead/ArrowHead3x3.STL'
  - The problem fixes all DOFs at z=0 plane
  - Load is distributed uniformly on the top surface (z=zMax plane)
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/ArrowHead/ArrowHead3x3.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(2,True)  # z = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
 
  load_nodes = mesh.getNodesOnBoundingBoxPlane(2,False)  # z = zMax plane
  load_dofs = 3 * load_nodes + 2
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createCompliantMechanismProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e5, poissons_ratio = 0.3,totalLoad = 1e3):
  """Creates a structural problem setup for an Compliant Mechanism 
  This function sets up a finite element mesh and boundary conditions for an Compliant Mechanism
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. 
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CompliantMechanism/CompliantMechanism.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts =mesh.node_indices[:, :3]*mesh.elem_size + mesh.origin
  fixed_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) > 20))[0] # the two end faces of the mechanism
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  load_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) < 20))[0] # the middle face of the mechanism    
  load_dofs = 3 * load_nodes  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ---------------------------------------
  
def createBeamSurfaceLoadProblem(nDOFDesired: int = 20000, L: float = [0.1, 0.01, 0.01],
                  youngs_modulus = 3e7, poissons_ratio = 0.3,totalLoad = 30000):
  # This is for large deformation
  nVoxelsDesired = int(nDOFDesired/5)
  print(nVoxelsDesired)

  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])+1
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))

  mesh.createEdofMatStructural()
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1 # this is for plotting

  # line defined by x = xMax
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
  load_dofs = 3 * load_nodes + 2  # z direction
  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force
  # ----------------------------------------
  
def createFilletedBeamProblem(nDOFDesired=50000, youngs_modulus = 2.1e5, poissons_ratio = 0.3,totalLoad = 1):
  stl_file = os.path.join(script_dir, '../Models/FilletedBeam/FilletedBeam.STL')

  mesh = hex_mesher.HexMesher()
  nElemsDesired = round(nDOFDesired/3)    # estimate
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane    
  # Get node coordinates
  node_centers = mesh.node_xyz[load_nodes]

  # Calculate center of face 
  face_center = np.mean(node_centers, axis=0)

  # Calculate torque vector for each node
  load_dofs = []
  force_values = []

  for node in load_nodes:
    node_pos = mesh.node_xyz[node] - face_center
    # Cross product with [1,0,0] to get perpendicular direction
    torque_dir = np.cross([1,0,0], node_pos)
    # Normalize
    if np.linalg.norm(torque_dir) > 0:
      torque_dir = torque_dir / np.linalg.norm(torque_dir)
      # Add force components
      load_dofs.extend([3*node + i for i in range(3)])
      force_values.extend(totalLoad/len(load_nodes) * torque_dir)
 
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = force_values

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createCentrifugalPlateProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                               poissons_ratio = 0.28, material_density = 7700,
                               rpm = 10000, verticalLoad = 100):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting


   # Apply centrifugal and gravity on all elements
  elem_body_force = np.zeros(3*mesh.num_elems)
  omega = 2*np.pi*rpm/60
  for e in range(mesh.num_elems):
    center = mesh.elem_centers[e]
    elem_body_force[3*e:3*e+2] = (material_density * np.prod(mesh.elem_size)) * omega**2 * center[:2]
  
  boundaryForce = np.zeros(3*mesh.num_nodes)   
  outerRadius = 0.05
  # Apply vertical load on the outer circumference elements
  vertical_load_nodes = mesh.get_nodes_within_annular_region(centerPt, axis, outerRadius - mesh.elem_size[0] * 0.707,
                                                             outerRadius + mesh.elem_size[0] * 0.707)
  vertical_load_dofs = 3 * vertical_load_nodes + 2  # z direction
  boundaryForce[vertical_load_dofs] = -verticalLoad / len(vertical_load_nodes)

  centrifugal_force_norm = np.linalg.norm(elem_body_force[::3])
  print("Centrifugal force norm:", centrifugal_force_norm)

  
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  return mesh, mat_prop, bc, elem_body_force

# ----------------------------------------

def createTorquePlateProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                               poissons_ratio = 0.28, torque =  500):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = None
  
  outerRadius = 0.05
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply torque force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates 
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    if r > 0:
      # Unit vector in tangential direction (perpendicular to radial)
      tangent_dir = np.array([-node_pos[1], node_pos[0]])/r
      # Apply tangential force to create torque
      boundaryForce[3*node] = torque/(r*len(load_nodes)) * tangent_dir[0]
      boundaryForce[3*node + 1] = torque/(r*len(load_nodes)) * tangent_dir[1]
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  return mesh, mat_prop, bc, elem_body_force

# ----------------------------------------

def createThreeHoleBracketProblem(nDOFDesired: int = 10000, youngs_modulus = 2e11, 
                               poissons_ratio = 0.28, totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/ThreeHoleBracket/ThreeHoleBracket.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix top hole
  centerPt = [0.015,0.065,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_1 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  # fix bottom hole
  centerPt = [0.015,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_2 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  fixed_nodes = np.union1d(fixed_nodes_1,fixed_nodes_2)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = None
  
  # load on right hole
  centerPt = [0.065,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)     
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  return mesh, mat_prop, bc, elem_body_force


# ----------------------------------------

def createBliskQuarterProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, 
                               poissons_ratio = 0.28, material_density = 7700,rpm = 10000,radialForce =10000,
																	   downwardForce = 0):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskQuarter.STL')


  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01085
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = np.zeros(3*mesh.num_elems)
  omega = 2*np.pi*rpm/60
  for e in range(mesh.num_elems):
    center = mesh.elem_centers[e]
    elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]

  print("total body force ",np.sum(elem_body_force[3::3]))
  # Apply centrifugal force on each node on the circumference
  # this is in addition to the body force
  outerRadius = 0.0565
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply radial force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    if r > 0:
      # Unit vector in radial direction
      radial_dir = node_pos/r
      # Add x and y dofs with force components
      boundaryForce[3*node] = radialForce/len(load_nodes) * radial_dir[0]  
      boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[1]
      boundaryForce[3*node + 2] = -downwardForce/len(load_nodes) 
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  return mesh, mat_prop, bc, elem_body_force


# ----------------------------------------

def createBliskSectionWithBlade(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, 
                               poissons_ratio = 0.28, material_density = 7700,rpm = 10000,radialForce =0):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskSectionWithBlade.STL')


  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01085
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting


  elem_body_force = np.zeros(3*mesh.num_elems)
  omega = 2*np.pi*rpm/60
  for e in range(mesh.num_elems):
    center = mesh.elem_centers[e]
    # Add centrifugal force to each element in xy plane
    elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]

  print("total body force ",np.linalg.norm(elem_body_force))
  outerRadius = 0.0558
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply radial force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    if r > 0:
      # Unit vector in radial direction
      radial_dir = node_pos/r
      # Add x and y dofs with force components
      boundaryForce[3*node] = radialForce/len(load_nodes) * radial_dir[0]  
      boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[1]
  
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
   
  return mesh, mat_prop, bc, elem_body_force


  # ----------------------------------------

def createKnuckleAssemblyProblem(nDOFDesired: int = 10000, youngs_modulus = [2e11,2e11], 
                               poissons_ratio = [0.28,0.28], totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/KnuckleAssembly/KnuckleAssembly.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
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

  # There are 2 components in the assembly
  # Assign material properties to each component
  mat_prop = 2*[None]
  mat_prop[0] = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus[0],
                      poissons_ratio=poissons_ratio[0]) # Knuckle
  mat_prop[1] = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus[1],
                      poissons_ratio=poissons_ratio[1]) # Shaft
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createTableProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/Table/Table.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
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

  load_per_dof = totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force



def createLBracketThickProblem(nDOFDesired: int = 80000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,topload = 1000,midload = 0):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for an L-bracket
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. The problem includes fixed boundary conditions on the top
  surface and a distributed load on a portion of the right surface.
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/LBracketThick/LBracketThick.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)  # y = yMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)
  node_pts = mesh.node_xyz
  if(abs(topload) > 0):
    topload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False) , np.where((node_pts[:, 1] >= 0.36))[0]) # hard coded  
    topload_nodes = topload_nodes[(node_pts[topload_nodes, 2] >= 0.23) & (node_pts[topload_nodes, 2] <= 0.27)]
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)

  if(abs(midload) > 0):
    midload_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), np.where((node_pts[:, 1] >= 0.18) & (node_pts[:, 1] <= 0.22))[0]) # hard coded    
    midload_nodes = midload_nodes[(node_pts[midload_nodes, 2] >= 0.23) & (node_pts[midload_nodes, 2] <= 0.27)]
    
    midload_dofs = 3 * midload_nodes + 1  
    mesh.node_indices[midload_nodes, 3] = 2 # for plotting
    
    force[midload_dofs] = -midload/len(midload_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
def createNoseconeProblem(nDOFDesired: int = 10000, youngs_modulus = 1e7, 
                               poissons_ratio = 0.28, totalLoad =  1000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/Nosecone/HollowNoseCone.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  fixed_nodes = np.where(node_pts[:, 1] == np.min(node_pts[:, 1]))[0]

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 1] > 0.03)[0]       
  
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


