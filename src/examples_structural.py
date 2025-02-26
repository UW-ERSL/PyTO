import numpy as np
import mat_lib
import bound_cond
import mesher
import mat_lib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def createCantileverProblem(nDOFDesired: int = 10000, L: float = [0.1, 0.1, 0.1],youngs_modulus = 2e11, poissons_ratio = 0.3):
  # This is an example where a grid mesh is created, and a structural problem is posed on it.
  # For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
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
  # line defined by x = xMax, and z = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(2,True))
  load_dofs = 3 * load_nodes + 2  # z direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -10000/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  return mesh, mat_prop, bc

def createMBBProblem(nDOFDesired: int = 10000, L: float = [0.5, 0.167, 0.01],youngs_modulus = 2e11, poissons_ratio = 0.3):
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

  symmetry_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  symmetry_dofs = np.array([3 * symmetry_nodes]).flatten().astype(int)

  fixed_nodes=np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  fixed_dofs = np.union1d(symmetry_dofs,np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int))
  
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1

 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,False))
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
  return mesh, mat_prop, bc

def createDistributedLoadProblem(nDOFDesired: int = 10000, L: float = [1.0, 0.5, 0.01],youngs_modulus = 2e11, poissons_ratio = 0.3):
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
  return mesh, mat_prop, bc

def createMultiloadProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],youngs_modulus = 2e11, poissons_ratio = 0.3):
  # This is an example where a grid mesh is created, and a structural problem is posed on it.
  # For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
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
  return mesh, mat_prop, bc

# %%
def createLBracketProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,totalLoad = 1000):
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
    - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../TOExamples/LBracket/LBracket.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  
  
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)  # y = yMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  node_pts = mesh.node_xyz
  load_nodes = np.where((node_pts[:, 1] > 0.039) & (node_pts[:, 0] > 0.09))[0] # hard coded    
  load_dofs = 3 * load_nodes + 1  # z direction
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  return mesh, mat_prop, bc

def createCircularPlateProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,totalLoad = 1000):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for a circular plate
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
    - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../TOExamples/CircularPlateHole/CircularPlateHole.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01-mesh.elem_size[0]/2
  outerRadius = 0.01+mesh.elem_size[0]/2
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius,outerRadius)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.05-mesh.elem_size[0]/2
  outerRadius = 0.05+mesh.elem_size[0]/2
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius,outerRadius) 
  load_dofs = 3 * load_nodes + 2
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  return mesh, mat_prop, bc


def createArrowHeadProblem(nDOFDesired: int = 10000, youngs_modulus = 2.1e11, poissons_ratio = 0.3,totalLoad = 1000):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for a arrow head 
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
    - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../TOExamples/ArrowHead/ArrowHead3x3.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
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
  return mesh, mat_prop, bc

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
    - The mesh is created from an STL file located at '../TOExamples/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../TOExamples/CompliantMechanism/CompliantMechanism.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = mesher.Mesher()
  
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
  return mesh, mat_prop, bc
    
# %%
def createAlcoaProblem(youngs_modulus = 2.1e11, poissons_ratio = 0.3):
  # This is an example where an existing mesh is read, and a structural problem is posed on it.
  mesh = mesher.Mesher()
  mesh.read_pareto_mesh("../meshFiles/AlcoaGrabCAD.msh")
  mesh.createEdofMatStructural()
  node_indices = mesh.node_indices

  fixed_nodes = np.where(node_indices[:, 3] == 1)[0]
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
                3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = np.zeros_like(fixed_dofs, dtype = float)

  load_nodes = np.where(node_indices[:, 3] == 2)[0]
  load_dofs = 3 * load_nodes + 1  # y direction
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -1000.

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values)

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  return mesh, mat_prop, bc

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
  mesh = mesher.Mesher()
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
  return mesh, mat_prop, bc

# %%
def createFilletedBeamProblem(nDOFDesired=50000, youngs_modulus = 2.1e5, poissons_ratio = 0.3,totalLoad = 1000):
  stl_file = os.path.join(script_dir, '../TOExamples/FilletedBeam/FilletedBeam.STL')

  mesh = mesher.Mesher()
  nElemsDesired = round(nDOFDesired/3)    # estimate
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_indices = mesh.node_indices
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = 0 plane     
  load_dofs = 3 * load_nodes + 2  # z direction
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=youngs_modulus,
                      poissons_ratio=poissons_ratio)
  return mesh, mat_prop, bc

