import numpy as np
import mat_lib
import bound_cond
import hex_mesher
import os
import enum
import scipy.sparse as spy_sprs
from hex_structural_examples import StructuralExamples
from stl_reader import STLGeom
from scipy.sparse import lil_matrix
script_dir = os.path.dirname(os.path.abspath(__file__))


class ThermoStructuralExamples(enum.Enum):
    BiClamp = enum.auto()

def getThermoStructuralProblem(problem: ThermoStructuralExamples, **kwargs):
    if problem == ThermoStructuralExamples.BiClamp:
        return createBiClampProblem(**kwargs)
    
def createBiClampProblem(nDOFDesired=25000, structural_load = 1e5,TWall = 28):
    stl_file = os.path.join(script_dir, '../Models/BiClamp/BiClamp.STL')

    mesh = hex_mesher.HexMesher()
    nElemsDesired = round(nDOFDesired/3)    # estimate
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()
    mesh.createEdofMatThermal()


    fixed_nodes_1 = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
    fixed_nodes_2 = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
    fixed_nodes = np.union1d(fixed_nodes_1, fixed_nodes_2)
    fixed_dofs = np.array([3 * fixed_nodes,
            3 * fixed_nodes + 1,
            3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

    # Get nodes on y = 0 plane
    y_plane_nodes = mesh.getNodesOnBoundingBoxPlane(1, True)  # y = 0 plane

    # Calculate x midpoint
    node_pts = mesh.node_xyz
    x_mid = (np.max(node_pts[:, 0]) + np.min(node_pts[:, 0])) / 2

    # Filter nodes within distance 0.01 from x_mid
    load_nodes = y_plane_nodes[np.abs(node_pts[y_plane_nodes, 0] - x_mid) < 0.01]
    mesh.node_indices[load_nodes, 3] = 2 # for plotting

    force = np.zeros(3*mesh.num_nodes)
    for node in load_nodes:
        force[3 * node + 1] = -structural_load / len(load_nodes)

    bcStructural = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    thermal_fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    thermal_dirichlet_values = TWall*np.ones_like(thermal_fixed_dofs, dtype = float)
    
    thermal_force = np.zeros(mesh.num_nodes) # no heat load

    bcThermal = bound_cond.BC(force = thermal_force,
						fixed_dofs = thermal_fixed_dofs,
						dirichlet_values = thermal_dirichlet_values) 
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None

    return mesh, mat_prop, bcStructural,bcThermal, elem_body_force