import numpy as np
import os
from tet_mesher import TetMesher
import mat_lib
import bound_cond
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class TetTransientThermalExamples(enum.Enum):
	ThickPlate = enum.auto()

def getTetTransientThermalProblem(problem: TetTransientThermalExamples, **kwargs):
  """
  """
  if problem == TetTransientThermalExamples.ThickPlate:
    return createThickPlateTransientThermalProblemTet(**kwargs)
  else:
    raise ValueError("Invalid tet transient thermal example name.")


def createThickPlateTransientThermalProblemTet(nDOFDesired: int = 10000,thermal_conductivity = 50, heat_load = 100000):
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
    

    fixed_nodes = np.where(tetmesh.nodes[:, 0] == np.min(tetmesh.nodes[:, 0]) )[0] # x = xMin plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 23*np.ones_like(fixed_dofs, dtype = float)
    load_nodes = np.where(tetmesh.nodes[:, 0] == np.max(tetmesh.nodes[:, 0]) )[0] # x = xMax plane
    tri_surface_indices = tetmesh.get_surface_triangles_with_all_nodes_in_node_set(load_nodes)
    #tri_surface_indices =  tetmesh.get_surface_triangles_on_bounding_box(axis_dir = 0,min_plane = False)
 
    force = tetmesh.integrate_over_surface_triangles(heat_load, tri_surface_indices)
   
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=thermal_conductivity,mass_density=7850,specific_heat=500)
    initialTemperature = 23
    timeStep = 60
    totalTime = 6000
    def transientHeatFunction(timeIndex, timeStep, tetmesh):
        """Transient heat function for the problem."""
        # Example: Apply a time-dependent heat load
        time = timeIndex * timeStep
        heat_at_time = heat_load *time/totalTime  # Example: sinusoidal variation
        q = tetmesh.integrate_over_surface_triangles(heat_at_time, tri_surface_indices)
        return q
    
    
    ptsOfInterest = np.array([[100,25,10],[50,50,10],[100,50,10],[100,75,10],[100,100,10]])
    return tetmesh, mat_prop, bc,initialTemperature,totalTime,timeStep,transientHeatFunction,ptsOfInterest

