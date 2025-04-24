import numpy as np
import os
from tet_mesher import TetMesher
import mat_lib
import bound_cond
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class TetStructuralExamples(enum.Enum):
	TensileBar = enum.auto()
	TorsionBar = enum.auto()
	BeamBending = enum.auto()
	CubeCompression = enum.auto()

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
  elif problem == TetStructuralExamples.TorsionBar:
    return createTorsionBarTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.BeamBending:
    return createBeamBendingTetStructuralProblem(**kwargs)
  elif problem == TetStructuralExamples.CubeCompression:
    return createCubeCompressionTetStructuralProblem(**kwargs)
  else:
    raise ValueError("Invalid structural tet example name.")
  

def createTensileBarTetStructuralProblem(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 100000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Beam/Beam.STL')
    nElemsDesired = nDOFDesired//3    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.nodes[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.nodes[:, 0])
 
    load_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes

    force = np.zeros(3 * quadratic_mesh.nodes.shape[0])
    force[load_dof] = totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=E,poissons_ratio=nu)
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force

def createTorsionBarTetStructuralProblem(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 100000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Beam/Beam.STL')
    nElemsDesired = nDOFDesired // 3  # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()

    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.nodes[:, 0])
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

    # Load application as a torsional load around the x-axis
    xmax = np.max(quadratic_mesh.nodes[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmax) <= tol)[0]
    load_dof_y = 3 * load_nodes + 1  # y-direction DOFs
    load_dof_z = 3 * load_nodes + 2  # z-direction DOFs

    force = np.zeros(3 * quadratic_mesh.nodes.shape[0])
    force[load_dof_y] = totalLoad / len(load_nodes)  # y-direction load
    force[load_dof_z] = totalLoad / len(load_nodes)  # z-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=E, poissons_ratio=nu)
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force

def createBeamBendingTetStructuralProblem(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 10000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Beam/Beam.STL')
    nElemsDesired = nDOFDesired // 3  # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()

    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.nodes[:, 0])
    tol = 1e-6
    fixed_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

 
    xmax = np.max(quadratic_mesh.nodes[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmax) <= tol)[0]
    load_dof = 3 * load_nodes + 1  # y-direction DOFs

    force = np.zeros(3 * quadratic_mesh.nodes.shape[0])
    force[load_dof] = -totalLoad / len(load_nodes)  # y-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=E, poissons_ratio=nu)
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createCubeCompressionTetStructuralProblem(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 10000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/Cube/Cube.STL')
    nElemsDesired = nDOFDesired//3    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.nodes[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.nodes[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes

    force = np.zeros(3 * quadratic_mesh.nodes.shape[0])
    force[load_dof] = -totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=E,poissons_ratio=nu)
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force


def createThickPlateTetStructuralProblem(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 100000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    nElemsDesired = nDOFDesired//3    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    quadratic_mesh = tetmesh.createQuadraticTetMesh()    

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    xmin = np.min(quadratic_mesh.nodes[:, 0])
   
    tol = 1e-8
    fixed_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmin) <= tol)[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
  
    # Load application at x = xMax plane
    xmax = np.max(quadratic_mesh.nodes[:, 0])
    load_nodes = np.where(np.abs(quadratic_mesh.nodes[:, 0] - xmax) <= tol)[0]
    load_dof = 3*load_nodes+2  

    force = np.zeros(3 * quadratic_mesh.nodes.shape[0])
    force[load_dof] = totalLoad / len(load_nodes)  # x-direction load

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.StructuralMaterial(youngs_modulus=E,poissons_ratio=nu)
    elem_body_force = None
    return quadratic_mesh, mat_prop, bc, elem_body_force

