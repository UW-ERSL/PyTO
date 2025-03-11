"""Mesher module."""

import dataclasses
from typing import Optional
import numpy as np
import pyvista as pv # pip install pyvista
from scipy.sparse import coo_matrix
import time

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
		startTime = time.time()
		print(f"Mesher: Grid size: {num_elems[0]} x {num_elems[1]} x {num_elems[2]}")
		nelx, nely, nelz = num_elems
		self.bbox = BoundingBox(x=Extent(0.0, nelx*elem_size[0]),
														y=Extent(0.0, nely*elem_size[1]),
														z=Extent(0.0, nelz*elem_size[2]))

		self.num_elems = nelx * nely * nelz
		self.num_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
		self.grid = [nelx, nely, nelz]
		self.origin = [0,0,0]
		self.elem_size = elem_size
		self.node_indices = np.zeros((self.num_nodes, 4), dtype = np.int32)

		self.elemArray = np.zeros((self.num_elems, 8), dtype = np.int32)
		self.elemPseudoDensity = np.ones(self.num_elems)

		# the elemNeighborsArray is needed for creating the filter
		self.elemNeighborsArray = np.zeros((self.num_elems, 27), dtype = np.int32)

		
		node = 0
		for iz in range(nelz+1):
			for iy in range(nely+1):
				for ix in range(nelx+1):
					self.node_indices[node]  = [ix, iy, iz, 0]
					node = node+1

		self.node_xyz = np.zeros((self.num_nodes, 3))
		for i in range(3):
			self.node_xyz[:,i] = self.origin[i] + self.elem_size[i]*self.node_indices[:,i]
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
		for elem in range(self.num_elems):
			self.elem_centers[elem, :] = np.array(np.sum(self.node_xyz[self.elemArray[elem]], 
																							axis = 0)/8.)
		self.createElementToNodeFieldMapping()

		# Report the time taken and the error in volume compared to box volume
		
		endTime = time.time()
		voxel_volume = self.num_elems * np.prod(self.elem_size)
		box_volume = self.bbox.lx * self.bbox.ly * self.bbox.lz
		volume_error = np.abs(voxel_volume - box_volume) / box_volume * 100
		endTime = time.time()
		print(f"Time taken to create mesh: {endTime - startTime:.2f} seconds")
		print(f"STL Volume: {box_volume:.2e}")
		print(f"Voxelized Mesh Volume: {voxel_volume:.2e}")
		print(f"Meshing Volume Error: {volume_error:.2f}%")
	
	def translate(self, dx: float, dy: float, dz: float):
		"""Translate mesh by specified amounts.
		
		Args:
			dx: Translation in x direction
			dy: Translation in y direction 
			dz: Translation in z direction
		"""
		# Update origin
		self.origin[0] += dx
		self.origin[1] += dy
		self.origin[2] += dz
		
		# Update node coordinates
		self.node_xyz[:,0] += dx
		self.node_xyz[:,1] += dy 
		self.node_xyz[:,2] += dz
		
		# Update element centers
		self.elem_centers[:,0] += dx
		self.elem_centers[:,1] += dy
		self.elem_centers[:,2] += dz
		
		# Update bounding box
		if self.bbox:
			self.bbox.x.min += dx
			self.bbox.x.max += dx
			self.bbox.y.min += dy
			self.bbox.y.max += dy
			self.bbox.z.min += dz
			self.bbox.z.max += dz
			
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
			self.node_indices = np.fromfile(file, dtype=np.uint32,
													count = 4*self.num_nodes).reshape((self.num_nodes,4))
			self.node_xyz = np.zeros((self.num_nodes, 3))
			for i in range(3):
				self.node_xyz[:,i] = self.origin[i] + self.elem_size[i]*self.node_indices[:,i]
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
			
			for elem in range(self.num_elems):
				self.elem_centers[elem, :] = np.array(np.sum(self.node_xyz[self.elemArray[elem]],
																						axis = 0)/8)

			self.bbox = BoundingBox(
						x=Extent(np.min(self.node_xyz[:,0]), np.max(self.node_xyz[:,0])),
						y=Extent(np.min(self.node_xyz[:,1]), np.max(self.node_xyz[:,1])),
						z=Extent(np.min(self.node_xyz[:,2]), np.max(self.node_xyz[:,2])))



	def createMeshFromSTLFile(self, stlFileName: str,nElemsDesired: int):
		startTime = time.time()
		self.stlMesh = pv.read(stlFileName)

		bounds = self.stlMesh.bounds
		Lx = bounds[1] - bounds[0]
		Ly = bounds[3] - bounds[2]
		Lz = bounds[5] - bounds[4]
		stlVolume = self.stlMesh.volume
		bBoxVolume = Lx*Ly*Lz
		# More voxels are needed inside the bounding box than the desired number of elements
		# Factor of 0.9 is arbitrary
		nElemsDesiredInsideBox = 0.9*nElemsDesired*(bBoxVolume/stlVolume)
		# assume nx = alpha*Lx, ny = alpha*Ly, nz = alpha*Lz
		alpha = (nElemsDesiredInsideBox/(Lx*Ly*Lz))**(1/3)
		nx = max(round(alpha*Lx), self.minVoxelsPerAxis)
		ny = max(round(alpha*Ly), self.minVoxelsPerAxis)
		nz = max(round(alpha*Lz), self.minVoxelsPerAxis)
		self.grid = [nx, ny, nz]
		self.elem_size= [Lx/nx, Ly/ny, Lz/nz]
		print(f"Mesher: Grid size: {nx} x {ny} x {nz}")
		print(f"Mesher: Element size: {self.elem_size[0]:.2e} x {self.elem_size[1]:.2e} x {self.elem_size[2]:.2e}")	
		# Voxels near the boundary are being removed. So scale the stl geometry slightly
		scale = 1.001
		# Scale the stl  about its center
		center = np.array(self.stlMesh.center)
		self.stlMesh.points = (self.stlMesh.points - center) * scale + center
		# Now voxelize
		self.voxels = pv.voxelize(self.stlMesh, density=self.elem_size, check_surface=False)
		# Unscale the stl back to its original size
		self.stlMesh.points = (self.stlMesh.points - center) / scale + center
		# Unscale the voxel points back to original size (same scaling as used for STL mesh)
		self.voxels.points = (self.voxels.points - center) / scale + center
		#extract the data
		self.num_elems = self.voxels.n_cells
		self.num_nodes = self.voxels.n_points 
		self.origin = [self.voxels.bounds[0], self.voxels.bounds[2], self.voxels.bounds[4]]

		self.node_indices = np.zeros((self.num_nodes, 4), dtype = np.int32)

		# Node array is the index of the node, and the label of the node
		# Convert voxel points to integer indices by dividing by elem_size and rounding
		self.node_indices[:, :3] = np.round((self.voxels.points - np.array(self.origin)) / np.array(self.elem_size))
		self.node_indices[:, 3] = 0
		self.node_xyz = np.zeros((self.num_nodes, 3))
		for i in range(3):
			self.node_xyz[:,i] = self.origin[i] + self.elem_size[i]*self.node_indices[:,i]
		self.elemArray = self.voxels.cell_connectivity
		self.elemArray = self.elemArray.reshape((self.num_elems, 8))
		
		self.elemPartIndex = np.zeros(self.num_elems)
		self.elem_centers = np.zeros((self.num_elems, 3))


		for elem in range(self.num_elems):
			self.elem_centers[elem, :] = np.array(np.sum(self.node_xyz[self.elemArray[elem]],
																						axis = 0)/8)

		self.elemPseudoDensity = np.ones(self.num_elems)
		# the elemNeighborsArray is needed for creating the filter
		self.elemNeighborsArray = np.zeros((self.num_elems, 27), dtype = np.int32)
		# Build a dictionary mapping each node to its associated elements
		node_to_elems = {}
		for elem_idx in range(self.num_elems):
			for node_idx in self.elemArray[elem_idx]:
				if node_idx not in node_to_elems:
					node_to_elems[node_idx] = []
				node_to_elems[node_idx].append(elem_idx)

		# For each element, find all neighboring elements by looking at shared nodes

		for elem in range(self.num_elems):
			neighbors = set()
			# Get all nodes of this element
			for node in self.elemArray[elem]:
				# Add all elements connected to this node
				neighbors.update(node_to_elems[node])
			# Convert to list 
			neighbor_list = list(neighbors)
			# Take first 27 neighbors (or pad with -1 if fewer exist)
			self.elemNeighborsArray[elem] = (neighbor_list[:27] + [-1] * 27)[:27]

		self.bbox = BoundingBox(
						x=Extent(bounds[0], bounds[1]),	
						y=Extent(bounds[2], bounds[3]),	
						z=Extent(bounds[4], bounds[5]))
		
		self.createElementToNodeFieldMapping()
		# Report the time taken and the error in volume compared to STL volume
		voxel_volume = self.num_elems * np.prod(self.elem_size)
		volume_error = np.abs(voxel_volume - stlVolume) / stlVolume * 100
		endTime = time.time()
		print(f"Time taken to create mesh: {endTime - startTime:.2f} seconds")
		print(f"STL Volume: {stlVolume:.2e}")
		print(f"Voxelized Mesh Volume: {voxel_volume:.2e}")
		print(f"Meshing Volume Error: {volume_error:.2f}%")

	def get_nodes_within_radius(self, pt: np.ndarray, r: float) -> np.ndarray:
		"""Find nodes within a given radius from a point.
		
		Args:
			pt: Array of shape (3,) containing x, y, z coordinates of the point
			r: Radius within which to find nodes
			
		Returns:
			np.ndarray: Indices of nodes within the given radius
		"""
		# Calculate squared distances from the point to all nodes
		distances_sq = np.sum((self.node_xyz - pt)**2, axis=1)
		
		# Find nodes within the radius (compare squared distances to squared radius)
		nodes_within_radius = np.where(distances_sq <= r**2)[0]
		return nodes_within_radius
	
	def get_nodes_within_annular_region(self, pt: np.ndarray, axis: np.ndarray, 
									  r_inner: float, r_outer: float) -> np.ndarray:
		"""Find nodes that lie within an annular region defined by two radii.
		
		Args:
			pt: Array of shape (3,) containing center point coordinates
			axis: Array of shape (3,) defining axis direction of cylinder
			r_inner: Inner radius of annular region
			r_outer: Outer radius of annular region
			
		Returns:
			np.ndarray: Indices of nodes within the annular region
		"""
		# Normalize axis vector
		axis = np.array(axis)
		axis = axis / np.linalg.norm(axis)
		
		# Vector from center to each node
		vectors = self.node_xyz - pt
		
		# Project vectors onto axis
		projections = np.dot(vectors, axis)[:, np.newaxis] * axis
		
		# Get perpendicular components
		perp_vectors = vectors - projections
		
		# Calculate radial distances
		radial_distances = np.linalg.norm(perp_vectors, axis=1)
		
		# Find nodes within annular region
		nodes_in_region = np.where((radial_distances >= r_inner) & 
								  (radial_distances <= r_outer))[0]
		
		return nodes_in_region
	def get_nodes_from_locations(self, locations: np.ndarray) -> np.ndarray:
		"""Find nodes closest to given x,y,z locations.
		
		Args:
			locations: Array of shape (n,3) containing x,y,z coordinates
			
		Returns:
			np.ndarray: Indices of closest nodes to each location
		"""
		# Convert locations to array if not already
		points = np.asarray(locations)
		
		# Calculate distances from each point to all nodes
		distances = np.sqrt(np.sum((self.node_xyz[:,np.newaxis,:] - points)**2, axis=2))
		
		# Get index of minimum distance for each point
		closest_nodes = np.argmin(distances, axis=0)
	
		return closest_nodes
	
	def get_element_containing_point(self, point: np.ndarray) -> int:
		"""Find element that contains given point based on closest element center.
		
		Args:
			point: Array of shape (3,) containing x,y,z coordinates
			
		Returns:
			int: Index of element containing point, or -1 if no element found
		"""
		# Calculate distances from point to all element centers
		distances = np.sqrt(np.sum((self.elem_centers - point)**2, axis=1))
		# Find element with minimum distance and the minimum distance
		closest_elem = np.argmin(distances) 
		min_distance = distances[closest_elem]
		# If the point is within 1.5 times the maximum element size, return the element
		if min_distance < 1.5*np.max(self.elem_size):
			# Calculate natural coordinates for the point within element
			# Get vertices of element
			vertices = self.node_xyz[self.elemArray[closest_elem]]
			# Transform point to element local coordinates (-1 to 1)
			center = self.elem_centers[closest_elem]
			xi =  (point - center) / (np.array(self.elem_size)/2)
			# Calculate shape functions at this point 
			# For hex8 element, shape functions are:
			# N = (1±xi)(1±eta)(1±zeta)/8
			N = np.array([
				(1-xi[0])*(1-xi[1])*(1-xi[2])/8,
				(1+xi[0])*(1-xi[1])*(1-xi[2])/8,
				(1+xi[0])*(1+xi[1])*(1-xi[2])/8,
				(1-xi[0])*(1+xi[1])*(1-xi[2])/8,
				(1-xi[0])*(1-xi[1])*(1+xi[2])/8,
				(1+xi[0])*(1-xi[1])*(1+xi[2])/8,
				(1+xi[0])*(1+xi[1])*(1+xi[2])/8,
				(1-xi[0])*(1+xi[1])*(1+xi[2])/8
			])
			return closest_elem, N
		return -1, None

	def getNodesOnBoundingBoxPlane(self, axis:int, minLimit:bool):
		"""Get the nodes on a bounding box plane.

		Args:
			axis: The axis of the bounding box plane (0 = x, 1 = y, 2 = z).
			minMax: The min or max of the bounding box plane (True or False).

		Returns:
			A list of node indices on the bounding box plane.
		"""
		if axis == 0:
			if minLimit:
				nodes_on_plane = np.where(self.node_indices[:, 0] == 0)[0]
			else: # max limit
				nodes_on_plane = np.where(self.node_indices[:, 0] == self.grid[0])[0]
		elif axis == 1:
			if minLimit:
				nodes_on_plane = np.where(self.node_indices[:, 1] == 0)[0]
			else:
				nodes_on_plane = np.where(self.node_indices[:, 1] == self.grid[1])[0]
		elif axis == 2:
			if minLimit:
				nodes_on_plane = np.where(self.node_indices[:, 2] == 0)[0]
			else:
				nodes_on_plane = np.where(self.node_indices[:, 2] == self.grid[2])[0]
		else:
			raise ValueError("Invalid axis. Must be 0, 1, or 2.")
		
		return nodes_on_plane

	def createEdofMatStructural(self):
		self.dofs_per_node = 3 # structural
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
		return
	
	def createElementToNodeFieldMapping(self):
		"""Create a sparse matrix mapping elements to nodes for field variables.
		This matrix is used to map element-based field variables (like density)		
		to node-based field variables.
		"""
		# Each element is associated with 8 nodes, and each node can belong to multiple elements
		# Initialize lists to store row indices, column indices, and data for the sparse matrix
		row_indices = []
		col_indices = []
		data = []

		# Loop over each element
		for elem in range(self.num_elems):
			# Loop over each node in the element
			for node in self.elemArray[elem]:
				row_indices.append(node)
				col_indices.append(elem)
				data.append(1/8)  # Each element value is divided equally among its 8 nodes

		# Create the sparse matrix
		self.elem_to_node_field_mapping = coo_matrix((data, (row_indices, col_indices)), shape=(self.num_nodes, self.num_elems))
		return
	
	def createEdofMatThermal(self):
		self.dofs_per_node = 1
		self.edofMat = np.zeros((self.num_elems, 8), dtype = int)
		elemArray= self.elemArray
		for el in range(self.num_elems):
			self.edofMat[el, :] = np.array([
				elemArray[el][0], 
				elemArray[el][1],
				elemArray[el][2], 
				elemArray[el][3], 
				elemArray[el][4],
				elemArray[el][5],
				elemArray[el][6],
				elemArray[el][7]]
			)
		return 
	
	def find_connected_components(self, threshold: float = 0.01) -> list[np.ndarray]:
		"""Find connected components of the mesh based on elemPseudoDensity.
		
		Args:
			threshold: Elements with pseudo-density below this value are considered "absent".
			
		Returns:
			A list of numpy arrays, where each array contains the indices of elements
			belonging to a connected component.
		"""
		
		# Create an adjacency matrix representing connections between elements
		adj_matrix = np.zeros((self.num_elems, self.num_elems), dtype=bool)
		
		# Iterate through each element and its neighbors
		for elem in range(self.num_elems):
			if self.elemPseudoDensity[elem] > threshold:
				neighbors = self.elemNeighborsArray[elem]
				# Set adjacency to True for valid neighbors (not -1)
				valid_neighbors = neighbors[neighbors != -1]
				adj_matrix[elem, valid_neighbors] = True
		
		# Convert adjacency matrix to a sparse format for efficiency
		adj_matrix_sparse = coo_matrix(adj_matrix)
		
		# Find connected components using depth-first search (DFS)
		visited = np.zeros(self.num_elems, dtype=bool)
		components = []
		
		
		# Iteratively find connected components using a stack
		for elem in range(self.num_elems):
			if self.elemPseudoDensity[elem] > threshold and not visited[elem]:
				current_component = []
				stack = [elem]  # Initialize stack with the starting element
				
				while stack:
					elem_idx = stack.pop()  # Get the last element from the stack
					
					if not visited[elem_idx]:
						visited[elem_idx] = True
						current_component.append(elem_idx)
						
						# Get neighbors of the current element
						neighbors = adj_matrix_sparse.row[adj_matrix_sparse.col == elem_idx]
						
						# Add unvisited neighbors to the stack
						for neighbor in neighbors:
							if self.elemPseudoDensity[neighbor] > threshold and not visited[neighbor]:
								stack.append(neighbor)
				
				components.append(np.array(current_component))
		
		return components
	def get_boundary_nodes(self) -> np.ndarray:
		"""Find nodes that lie on the boundary of the mesh.
		
		Returns:
			np.ndarray: Array of node indices that are on the boundary
		"""
		# For each node, count how many elements it belongs to
		node_elem_count = np.zeros(self.num_nodes, dtype=int)
		for elem in self.elemArray:
			np.add.at(node_elem_count, elem, 1)
		
		# In a fully interior hex mesh, each node should belong to 8 elements
		# (except at corners, edges, and faces where it will be less)
		boundary_nodes = np.where(node_elem_count < 8)[0]
		
		return boundary_nodes
	
	def setPseudoDensity(self, rho):
		self.elemPseudoDensity = rho.copy()

if __name__ == "__main__":
    import os
    import plots
    import time


    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/Knuckle/Knuckle.STL')
    mesh = Mesher()
    mesh.createMeshFromSTLFile(stlFileName,nElemsDesired=100000)
    plots.plotMesh(mesh,  title=f'Knuckle; nElems = {mesh.num_nodes}')
  