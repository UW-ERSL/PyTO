import numpy as np
import os
from hex_mesher import HexMesher
import mat_lib
import bound_cond
import enum
script_dir = os.path.dirname(os.path.abspath(__file__))

class HexTransientThermalExamples(enum.Enum):
	ThickPlate = enum.auto()

def getHexTransientThermalProblem(problem: HexTransientThermalExamples, **kwargs):
  """
  """
  if problem == HexTransientThermalExamples.ThickPlate:
    return createThickPlateTransientThermalProblemHex(**kwargs)
  else:
    raise ValueError("Invalid hex transient thermal example name.")


def createThickPlateTransientThermalProblemHex(nDOFDesired: int = 10000, heat_load = 100000):
    """
  
	"""
	# Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/ThickPlate/ThickPlate.STL')
    nElemsDesired = nDOFDesired	# estimate
    hexmesh = HexMesher()
	
    hexmesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    print(f"Number of elements: {hexmesh.num_elems}")
    print(f"Number of nodes: {hexmesh.num_nodes}")

    fixed_nodes = np.where(hexmesh.node_xyz[:, 0] == np.min(hexmesh.node_xyz[:, 0]) )[0] # x = xMin plane
    fixed_dofs = np.array([fixed_nodes]).flatten().astype(int)
    dirichlet_values = 23*np.ones_like(fixed_dofs, dtype = float)
    load_nodes = np.where(hexmesh.node_xyz[:, 0] == np.max(hexmesh.node_xyz[:, 0]) )[0] # x = xMax plane
    
    force = np.zeros_like(hexmesh.node_xyz[:, 0])
    force[load_nodes] = heat_load/len(load_nodes)
    bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel")
    initialTemperature = 23
    timeStep = 60
    totalTime = 6000
    def transientHeatFunction(timeIndex, timeStep, hexmesh):
        """Transient heat function for the problem."""
        # Example: Apply a time-dependent heat load
        currentTime = timeIndex * timeStep
        q = force*currentTime/totalTime
        return q
    
    
    ptsOfInterest = np.array([[100,25,10],[50,50,10],[100,50,10],[100,75,10],[100,100,10]])
    return hexmesh, mat_prop, bc,initialTemperature,totalTime,timeStep,transientHeatFunction,ptsOfInterest

