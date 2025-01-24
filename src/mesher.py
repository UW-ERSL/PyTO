"""Mesher module."""

import dataclasses
from typing import Optional
import numpy as np
import pyvista as pv # pip install pyvista

@dataclasses.dataclass
class Extent:
  min: float
  max: float


  @property
  def range(self)->float:
    return self.max - self.min


  @property
  def center(self)->float:
    return 0.5*(self.min + self.max)


@dataclasses.dataclass
class BoundingBox:
  x: Extent
  y: Extent
  z: Optional[Extent] = None


  @property
  def lx(self)->float:
    return np.abs(self.x.range)


  @property
  def ly(self)->float:
    return np.abs(self.y.range)


  @property
  def lz(self) -> float:
    return np.abs(self.z.range) if self.z else 0.0


  @property
  def diag_length(self)->float:
    return np.sqrt(self.lx**2 + self.ly**2 + self.lz**2)


class Mesher:
	def __init__(self):	
		self.num_nodes = 0
		self.num_elems = 0
		self.minVoxelsPerAxis = 2


	@property
	def dofs_per_node(self)->int:
		return 3


	def grid_mesh(self,
									 num_elems: tuple[int, int, int],
									 elem_size: tuple[float, float, float],):
		"""Generate a grid of elements and nodes.
		Args:
			num_elems: Tuple of integers of the number of elements in each direction 
				(nelx, nely, nelz).
			elem_size: Tuple of floats of the size of each element in each direction
				(dx, dy, dz).
		"""
		nelx, nely, nelz = num_elems
		self.bbox = BoundingBox(x=Extent(0.0, nelx*elem_size[0]),
														y=Extent(0.0, nely*elem_size[1]),
														z=Extent(0.0, nelz*elem_size[2]))

		self.num_elems = nelx * nely * nelz
		self.num_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
		self.grid = [nelx, nely, nelz]
		self.origin = [0,0,0]
		self.elem_size = elem_size
		self.node_array = np.zeros((self.num_nodes, 4), dtype = np.int32)
		self.elemArray = np.zeros((self.num_elems, 8), dtype = np.int32)
		self.elemPseudoDensity = np.ones(self.num_elems)

		# the elemNeighborsArray is needed for creating the filter
		self.elemNeighborsArray = np.zeros((self.num_elems, 27), dtype = np.int32)

		
		node = 0
		for iz in range(nelz+1):
			for iy in range(nely+1):
				for ix in range(nelx+1):
					self.node_array[node]  = [ix, iy, iz, 0]
					node = node+1

		elem = 0
		sx = nelx+1
		sy = nely+1
		for iz in range(nelz):
			for iy in range(nely):
				for ix in range(nelx):
					n0 = iz*sx*sy + iy*sx + ix # corner node number
					self.elemArray[elem]  = [n0, n0+1,
																	n0+1+sx, n0+sx,
																	n0+sx*sy, n0+1+sx*sy,
																	n0+1+sx+sx*sy ,n0+sx+sx*sy]
					
					# Calculate neighbors
					neighbors = []
					for dz in [-1, 0, 1]:
						for dy in [-1, 0, 1]:
							for dx in [-1, 0, 1]:
								neighbor_elem = elem + (dz * nelx * nely) + (dy * nelx) + dx
								neighbors.append(neighbor_elem)
					neighbors = np.array(neighbors)
					if (iz == 0):
						neighbors[0:9] = -1 
					if (iz == nelz-1):
						neighbors[18:27] = -1 
					if (iy == 0):
						neighbors[[0, 1, 2, 9, 10, 11, 18, 19, 20]] = -1
					if (iy == nely-1):
						neighbors[[6, 7, 8, 15, 16, 17, 24, 25, 26]] = -1
					if (ix == 0):
						neighbors[[0, 3, 6, 9, 12, 15, 18, 21, 24]] = -1
					if (ix == nelx-1):
						neighbors[[2, 5, 8, 11, 14, 17, 20, 23, 26]] = -1

					if (any(neighbors) > nelx*nely*nelz-1):
						print(ix,iy,iz)
						input('wait')
					neighbors [neighbors > nelx*nely*nelz-1] = nelx*nely*nelz-1
					self.elemNeighborsArray[elem] = neighbors		
					elem = elem+1

		self.elem_centers = np.zeros((self.num_elems, 3))
		node_array = self.node_array[:, :3]
		for elem in range(self.num_elems):
			self.elem_centers[elem, :] = np.array(np.sum(node_array[self.elemArray[elem]], 
																							axis = 0)/8.)

		self.createEdofMat()


	def read_pareto_mesh(self, fileName: str):
		"""Read a Pareto mesh from a binary file.

		The binary file should contain the following data:
		1. nelx nely nelz (#number of elements in x, y and z)
		2. X0 Y0 Z0 (origin)
		3. dx dy dz (deltax, deltay, deltaz)
		4. h_nNodes
		5. (i,j,k,l) of all nodes (i ranges from 0 to nelx, etc), l is the label of node
		6. h_nElems
		7. (n1,n2,...,n8) of all elements
		8. (m0 m1 m2 ....) for each element, the id (typically part #), default is 0
		9. (1 1 0.7 1 0 ....) pseudoDensity for each element
		"""
		with open(fileName, mode='rb') as file:

			self.grid = np.fromfile(file, dtype=np.uint32, count = 3)  
			self.origin = np.fromfile(file, dtype=np.double, count = 3)  
			self.elem_size = np.fromfile(file, dtype=np.double, count = 3)

			self.num_nodes = np.fromfile(file, dtype=np.uint32, count = 1)[0]
			self.node_array = np.fromfile(file, dtype=np.uint32,
													count = 4*self.num_nodes).reshape((self.num_nodes,4))
			self.num_elems = np.fromfile(file, dtype=np.uint32, count = 1)[0]

			self.elemArray = np.fromfile(file, dtype=np.uint32,
														count = 8*self.num_elems).reshape((self.num_elems,8))
			self.elemPartIndex = np.fromfile(file, dtype=np.uint32, count = self.num_elems)
			self.elemPseudoDensity = np.fromfile(file, dtype=np.double, count = self.num_elems)

			self.elemNeighborsfileName = fileName.replace("msh", "elneigh")
			self.elemNeighborsArray = np.loadtxt(self.elemNeighborsfileName, skiprows = 2)	

			print("#Nodes = ", self.num_nodes, "\n#Elems = ", self.num_elems)
			file.close()
			self.elem_centers = np.zeros((self.num_elems, 3))
			node_array = self.node_array[:, :3]
			for elem in range(self.num_elems):
				self.elem_centers[elem, :] = np.array(np.sum(node_array[self.elemArray[elem]],
																						axis = 0)/8)

			self.bbox = BoundingBox(
						x=Extent(np.min(self.node_array[:,0]), np.max(self.node_array[:,0])),
						y=Extent(np.min(self.node_array[:,1]), np.max(self.node_array[:,1])),
						z=Extent(np.min(self.node_array[:,2]), np.max(self.node_array[:,2])))

			self.createEdofMat()

	def createMeshFromSTLFile(self, stlFileName: str,nElemsDesired: int):
		self.stlMesh = pv.read(stlFileName)

		bounds = self.stlMesh.bounds
		Lx = bounds[1] - bounds[0]
		Ly = bounds[3] - bounds[2]
		Lz = bounds[5] - bounds[4]
		volume = self.stlMesh.volume
		alpha = (nElemsDesired/(Lx*Ly*Lz))**(1/3)
		
		nx = max(round(alpha*Lx), self.minVoxelsPerAxis)
		ny = max(round(alpha*Ly), self.minVoxelsPerAxis)
		nz = max(round(alpha*Lz), self.minVoxelsPerAxis)
		self.grid = [nx, ny, nz]
		self.elem_size= [Lx/nx, Ly/ny, Lz/nz]
		
		# Voxels near the boundary are being removed. So scale the mesh slightly
		scale = 1.001
		# Scale the mesh  about its center
		center = np.array(self.stlMesh.center)
		self.stlMesh.points = (self.stlMesh.points - center) * scale + center
		self.voxels = pv.voxelize(self.stlMesh, density=self.elem_size, check_surface=False)
		self.stlMesh.points = (self.stlMesh.points - center) / scale + center
		#extract the data
		self.num_elems = self.voxels.n_cells
		self.num_nodes = self.voxels.n_points 
		self.origin = [self.voxels.bounds[0], self.voxels.bounds[2], self.voxels.bounds[4]]

		self.node_array = np.zeros((self.num_nodes, 4), dtype = np.int32)

		# Node array is the index of the node, and the label of the node
		# Convert voxel points to integer indices by dividing by elem_size and rounding
		self.node_array[:, :3] = np.round((self.voxels.points - np.array(self.origin)) / np.array(self.elem_size))
		self.node_array[:, 3] = 0
		self.elemArray = self.voxels.cell_connectivity
		self.elemArray = self.elemArray.reshape((self.num_elems, 8))
		
		self.elemPartIndex = np.zeros(self.num_elems)
		self.elem_centers = np.zeros((self.num_elems, 3))
		node_array = self.node_array[:, :3]
		for elem in range(self.num_elems):
			self.elem_centers[elem, :] = np.array(np.sum(node_array[self.elemArray[elem]],
																						axis = 0)/8)
				
		self.elemPseudoDensity = np.ones(self.num_elems)
		# the elemNeighborsArray is needed for creating the filter
		self.elemNeighborsArray = np.zeros((self.num_elems, 27), dtype = np.int32)
		self.createEdofMat()
		self.bbox = BoundingBox(
						x=Extent(np.min(self.node_array[:,0]), np.max(self.node_array[:,0])),
						y=Extent(np.min(self.node_array[:,1]), np.max(self.node_array[:,1])),
						z=Extent(np.min(self.node_array[:,2]), np.max(self.node_array[:,2])))

	def createEdofMat(self):
		self.edofMat = np.zeros((self.num_elems, 24), dtype = int)
		elemArray= self.elemArray
		for el in range(self.num_elems):
			self.edofMat[el, :] = np.array([
				3*elemArray[el][0], 3*elemArray[el][0]+1, 3*elemArray[el][0]+2,
				3*elemArray[el][1], 3*elemArray[el][1]+1, 3*elemArray[el][1]+2,
				3*elemArray[el][2], 3*elemArray[el][2]+1, 3*elemArray[el][2]+2,
				3*elemArray[el][3], 3*elemArray[el][3]+1, 3*elemArray[el][3]+2,
				3*elemArray[el][4], 3*elemArray[el][4]+1, 3*elemArray[el][4]+2,
				3*elemArray[el][5], 3*elemArray[el][5]+1, 3*elemArray[el][5]+2,
				3*elemArray[el][6], 3*elemArray[el][6]+1, 3*elemArray[el][6]+2,
				3*elemArray[el][7], 3*elemArray[el][7]+1, 3*elemArray[el][7]+2]
			)


	def setPseudoDensity(self, rho):
		self.elemPseudoDensity = rho.copy()