import numpy as np
import os
from tet_mesher import TetMesher
import mat_lib
import bound_cond
script_dir = os.path.dirname(os.path.abspath(__file__))

def createThickPlateStructuralProblemTet(nDOFDesired: int = 10000, E = 2e11, nu = 0.3, totalLoad = 1000):
    # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    nElemsDesired = nDOFDesired//3    # estimate (3 DOFs per node for structural)
    tetmesh = TetMesher()
    
    tetmesh.createTetMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
   

    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
    fixed_nodes = np.where(tetmesh.nodes[:, 0] == np.min(tetmesh.nodes[:, 0]))[0]
    fixed_dofs = np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2])
    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

    # Load application at x = xMax plane
    load_nodes = np.where(tetmesh.nodes[:, 0] == np.max(tetmesh.nodes[:, 0]))[0]
    tri_surface_indices = tetmesh.get_surface_triangles_with_all_nodes_in_node_set(load_nodes)
    
    # Create force vector (only x-direction load)
    force = np.zeros(3 * tetmesh.nodes.shape[0])
    load_per_node = totalLoad / len(load_nodes)
    force[3*load_nodes] = load_per_node  # x-direction load
    
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    
    mat_prop = mat_lib.ElasticMaterial(E=E, nu=nu)  # Changed to elastic material
    return tetmesh, mat_prop, bc
