import numpy as np
import os
from tet_mesher import TetMesher
import mat_lib
import bound_cond
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class TetStructuralExamples(enum.Enum):
	TensileBar = enum.auto()
	BeamBending = enum.auto()
	CubeCompression = enum.auto()
	Arrowhead = enum.auto()
	BliskModel = enum.auto()

def getTetStructuralProblem(problem: TetStructuralExamples, **kwargs):
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
  if problem == TetStructuralExamples.TensileBar:
    return createTensileBarTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.BeamBending:
    return createBeamBendingTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.CubeCompression:
    return createCubeCompressionTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.BliskModel:
    return createBliskModelTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.GEGrabCAD:
    return createGEGrabCADTetStructuralProblem(**kwargs)
  else:
    raise ValueError("Invalid structural tet example name.")
  

def createTensileBarTetStructuralProblem(nDOFDesired: int = 10000, totalLoad = 100000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Beam/Beam.STL')
    nElemsDesired = nDOFDesired//6    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.node_xyz[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.node_xyz[:, 0])
 
    load_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes

    force = np.zeros(3 * quadratic_mesh.node_xyz.shape[0])
    force[load_dof] = totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createBeamBendingTetStructuralProblem(nDOFDesired: int = 10000, totalLoad = 10000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Beam/Beam.STL')
    nElemsDesired = nDOFDesired // 6 # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()

    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.node_xyz[:, 0])
    tol = 1e-6
    fixed_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

 
    xmax = np.max(quadratic_mesh.node_xyz[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmax) <= tol)[0]
    load_dof = 3 * load_nodes + 1  # y-direction DOFs

    force = np.zeros(3 * quadratic_mesh.node_xyz.shape[0])
    force[load_dof] = -totalLoad / len(load_nodes)  # y-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createCubeCompressionTetStructuralProblem(nDOFDesired: int = 10000,  totalLoad = 10000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Cube/Cube.STL')
    nElemsDesired = nDOFDesired//6    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.node_xyz[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.node_xyz[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes

    force = np.zeros(3 * quadratic_mesh.node_xyz.shape[0])
    force[load_dof] = -totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createThickPlateTetStructuralProblem(nDOFDesired: int = 10000, totalLoad = 100000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    nElemsDesired = nDOFDesired//6    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.node_xyz[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
  
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.node_xyz[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.node_xyz[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes+2  

    force = np.zeros(3 * quadratic_mesh.node_xyz.shape[0])
    force[load_dof] = totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createBliskModelTetStructuralProblem(nDOFDesired: int = 100000, pressure = 1e6):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskSectionWithBlade.STL')
    nElemsDesired = nDOFDesired//6    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
  
    tol = 1e-8
  
     # fix inner radius
    centerPt = [0,0,0]
    axis = [0,0,1]
    innerRadius = 0.05
    fixed_nodes = tetmesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-tetmesh.elem_size*0.707,
                                                      innerRadius+tetmesh.elem_size*0.707)  
    fixed_dofs = np.array([3 * fixed_nodes,
                3 * fixed_nodes + 1,
                3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
   
    # load on the blade surface
    bladeStartRadius = 0.055
    bladeEndRadius = 0.075
    load_nodes = tetmesh.get_nodes_within_annular_region(centerPt, axis, bladeStartRadius,
                                                      bladeEndRadius)
    load_dof = 3*load_nodes+1  

    force = np.zeros(3 * quadratic_mesh.node_xyz.shape[0])
    force[load_dof] = -1000 / len(load_nodes)  

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createGEGrabCADTetStructuralProblem(nDOFDesired: int = 10000, totalLoad = 100000):
    
    tetmesh = TetMesher()
    
    tetmesh.read_Abaqus_linear_tetmesh(os.path.join(script_dir, '../Models/GEGrabCAD/GEGrabCADLinearTetMesh.inp'))
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    
    mat_prop = mat_lib.get_material("Steel")
    bc = None
    elem_body_force = None
   
    return quadratic_mesh, mat_prop, bc, elem_body_force