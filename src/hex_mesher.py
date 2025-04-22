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
		#print(f"Mesher: Grid size: {num_elems[0]} x {num_elems[1]} x {num_elems[2]}")
		#print(f"Mesher: Element size: {elem_size[0]:.2e} x {elem_size[1]:.2e} x {elem_size[2]:.2e}")
		nelx, nely, nelz = num_elems
		self.bbox = BoundingBox(x=Extent(0.0, nelx*elem_size[0]),
														y=Extent(0.0, nely*elem_size[1]),
														z=Extent(0.0, nelz*elem_size[2]))
		self.num_components = 1
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
		self.volume_error = volume_error
		self.meshing_time = endTime - startTime
		#print(f"Time taken to create mesh: {endTime - startTime:.2f} seconds")
		#print(f"Meshing Volume Error: {volume_error:.2f}%")
	
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



	def createMeshFromSTLFileSingleComponent(self, stlFileName: str,nElemsDesired: int):
		startTime = time.time()
		self.stlMesh = pv.read(stlFileName)

		# Extract connected components
		components = self.stlMesh.connectivity(label=True)

		# Get the number of components
		num_components = components["RegionId"].max() + 1
		self.num_components = num_components
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
		
		#print(f"Mesher: Grid size: {nx} x {ny} x {nz}")
		#print(f"Mesher: Element size: {self.elem_size[0]:.2e} x {self.elem_size[1]:.2e} x {self.elem_size[2]:.2e}")	
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
		#print(f"Mesher: #Elements: {self.num_elems}")
		#print(f"Mesher: #Nodes: {self.num_nodes}")
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
		self.meshing_time = endTime - startTime
		self.volume_error = volume_error
		#print(f"Time taken to create mesh: {endTime - startTime:.2f} seconds")
		#print(f"Meshing Volume Error: {volume_error:.2f}%")

	def createMeshFromSTLFile(self, stlFileName: str,nElemsDesired: int):
		#print("Creating mesh from STL file...")
		startTime = time.time()
		self.stlMesh = pv.read(stlFileName)

		# Extract connected components
		components = self.stlMesh.connectivity(label=True)

		# Get the number of components
		num_components = components["RegionId"].max() + 1
		#print(f"Number of connected components: {num_components}")
		self.num_components = num_components
		if (num_components == 1):
			self.createMeshFromSTLFileSingleComponent(stlFileName, nElemsDesired)
			return 
		# Define voxel spacing (adjust as needed)
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
		#print(f"Mesher: Grid size: {nx} x {ny} x {nz}")
		#print(f"Mesher: Element size: {self.elem_size[0]:.2e} x {self.elem_size[1]:.2e} x {self.elem_size[2]:.2e}")


		n_open_edges = self.stlMesh.n_open_edges
		if n_open_edges > 0:
			print("Model has open edges. Please fix and retry.")
			return
		# Create a single voxelized mesh for the entire assembly
	
		voxel_mesh = pv.voxelize(self.stlMesh, density=self.elem_size, check_surface=False)
		component_ids = np.zeros(voxel_mesh.n_cells, dtype=np.int32)
		
		# Get cell centers of the voxelized mesh
		cell_centers = voxel_mesh.cell_centers().points
		# Create a PolyData object from the cell centers for easier processing
		centers_polydata = pv.PolyData(cell_centers)

		for i in range(num_components):
			# Extract individual component
			component = components.threshold([i, i], scalars="RegionId")
			#component_polydata = pv.PolyData(component.points, component.cells) #can use either this or the next line
			component_polydata = component.extract_surface()

            # use select_enclosed_points to find cells inside the component
			try:
				selection = centers_polydata.select_enclosed_points(component_polydata, tolerance=0.0, check_surface=True)
			except Exception as e:
				print(f"Surface is not closed")
				selection = centers_polydata.select_enclosed_points(component_polydata, tolerance=0.0, check_surface=False)
            
			#selection.plot(show_edges=True)
            # Get the mask of points that are inside the component
			if 'SelectedPoints' in selection.point_data:
				mask = selection.point_data['SelectedPoints'].astype(bool)
			else:
			# Alternatively try the InsidePoints array
				mask = selection.point_data.get('InsidePoints', np.zeros(len(cell_centers))).astype(bool)
			
			# Assign component ID to the cells that are inside this component
			# Only assign if not already assigned (priority to lower component IDs)
			empty_cells = component_ids == 0
			component_ids[np.logical_and(mask, empty_cells)] = i + 1
			#print(f"  Added {np.sum(np.logical_and(mask, empty_cells))} cells for component {i+1} using select_enclosed_points")

		# Add component IDs to the voxel mesh
		voxel_mesh.cell_data["component_id"] = component_ids
		# Create an array to store component IDs for each element
		
		# Extract only cells that belong to a component (non-zero component_id)
		voxel_mesh_components = voxel_mesh.threshold(0.5, scalars="component_id")
		
		#extract the data
		self.voxels = voxel_mesh_components
		self.num_elems = self.voxels.n_cells
		self.num_nodes = self.voxels.n_points 
		
		#print(f"Mesher: #Elements: {self.num_elems}")
		#print(f"Mesher: #Nodes: {self.num_nodes}")
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
		
		# Create an array to store component IDs for each element
		self.elemComponentId = np.zeros(self.num_elems, dtype=np.int32)
		for elem_idx in range(self.num_elems):
			self.elemComponentId[elem_idx] = self.voxels.cell_data["component_id"][elem_idx]

		# Count the number of elements for each component
		component_counts = np.bincount(self.elemComponentId)
	
		# Remove components with zero elements and shift the IDs
		# Get counts per component excluding zero
		valid_counts = component_counts[component_counts > 0]
		self.num_components = len(valid_counts)

		# Create mapping from old to new component IDs 
		old_to_new = np.zeros(len(component_counts), dtype=np.int32)
		new_id = 1
		for i in range(len(component_counts)):
			if component_counts[i] > 0:
				old_to_new[i] = new_id
				new_id += 1

		# Update component IDs using the mapping
		self.elemComponentId = old_to_new[self.elemComponentId]
		
		# Recount components after remapping
		component_counts = valid_counts
	
		for component_id, count in enumerate(component_counts):
			print(f"Component {component_id}: {count} elements")
		
		# Create a mapping from component ID to part index

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
		voxel_volume = self.num_elems * np.prod(self.elem_size)
		volume_error = np.abs(voxel_volume - stlVolume) / stlVolume * 100
		self.createElementToNodeFieldMapping()
		endTime = time.time()
		#print(f"Time taken to create mesh: {endTime - startTime:.2f} seconds")
		#print(f"STL Volume: {stlVolume:.2e}")
		#print(f"Voxelized Mesh Volume: {voxel_volume:.2e}")
		#print(f"Meshing Volume Error: {volume_error:.2f}%")
		self.volume_error = volume_error
		self.meshing_time = endTime - startTime

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
	
	def get_elems_within_annular_region(self, pt: np.ndarray, axis: np.ndarray, 
									  r_inner: float, r_outer: float) -> np.ndarray:
		"""Find elements that lie within an annular region defined by two radii.
		
		Args:
			pt: Array of shape (3,) containing center point coordinates
			axis: Array of shape (3,) defining axis direction of cylinder
			r_inner: Inner radius of annular region
			r_outer: Outer radius of annular region
			
		Returns:
			np.ndarray: Indices of elements within the annular region
		"""
		# Normalize axis vector
		axis = np.array(axis)
		axis = axis / np.linalg.norm(axis)
		
		# Vector from center to each element center
		vectors = self.elem_centers - pt
		
		# Project vectors onto axis
		projections = np.dot(vectors, axis)[:, np.newaxis] * axis
		
		# Get perpendicular components
		perp_vectors = vectors - projections
		
		# Calculate radial distances
		radial_distances = np.linalg.norm(perp_vectors, axis=1)
		
		# Find elements within annular region
		elems_in_region = np.where((radial_distances >= r_inner) & 
								  (radial_distances <= r_outer))[0]
		
		return elems_in_region
		
	
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
	
	def create_spatial_hash_grid(self):
		"""Create a spatial hash grid for fast element lookups.
		The grid divides space into cells of size max(elem_size)."""
		
		# Use maximum element size for grid cell size
		self.grid_cell_size = np.max(self.elem_size)
		
		# Calculate grid dimensions
		bbox_min = [self.bbox.x.min, self.bbox.y.min, self.bbox.z.min]
		bbox_max = [self.bbox.x.max, self.bbox.y.max, self.bbox.z.max]
		grid_dims = np.ceil((np.array(bbox_max) - np.array(bbox_min)) / self.grid_cell_size).astype(int)
		
		# Initialize hash grid dictionary
		self.hash_grid = {}
		
		# Add each element center to grid
		for elem_idx in range(self.num_elems):
			# Get grid cell indices for this element
			cell_indices = tuple(np.floor((self.elem_centers[elem_idx] - bbox_min) / self.grid_cell_size).astype(int))
			
			# Add element to grid cell
			if cell_indices not in self.hash_grid:
				self.hash_grid[cell_indices] = []
			self.hash_grid[cell_indices].append(elem_idx)

	def get_elems_within_radius(self, pt: np.ndarray, r: float) -> np.ndarray:
		"""Find elements within a given radius from a point using spatial hash grid.
		
		Args:
			pt: Array of shape (3,) containing x, y, z coordinates of the point
			r: Radius within which to find elements
			
		Returns:
			np.ndarray: Indices of elements within the given radius
		"""
		# Ensure hash grid exists
		if not hasattr(self, 'hash_grid'):
			self.create_spatial_hash_grid()
		
		# Get grid cell containing the point
		bbox_min = [self.bbox.x.min, self.bbox.y.min, self.bbox.z.min]
		cell_indices = tuple(np.floor((pt - bbox_min) / self.grid_cell_size).astype(int))
		
		# Determine the range of cells to search based on the radius
		cell_radius = int(np.ceil(r / self.grid_cell_size))
	
		
		# Collect potential elements from neighboring cells
		potential_elems = set()
		for dx in range(-cell_radius, cell_radius + 1):
			for dy in range(-cell_radius, cell_radius + 1):
				for dz in range(-cell_radius, cell_radius + 1):
					neighbor_cell = (cell_indices[0] + dx, 
									 cell_indices[1] + dy, 
									 cell_indices[2] + dz)
					if neighbor_cell in self.hash_grid:
						potential_elems.update(self.hash_grid[neighbor_cell])
		
		# Filter elements within the radius
		potential_elems = list(potential_elems)
		distances_sq = np.sum((self.elem_centers[potential_elems] - pt)**2, axis=1)
		elems_within_radius = np.array(potential_elems)[distances_sq <= r**2]
		
		return elems_within_radius
	def get_element_near_point(self, point: np.ndarray) -> int:
		"""Find element closest to point using spatial hash grid.
		
		Args:
			point: Array of shape (3,) containing x,y,z coordinates
			
		Returns:
			int: Index of closest element, or -1 if no element found
		"""
		# Ensure hash grid exists
		if not hasattr(self, 'hash_grid'):
			self.create_spatial_hash_grid()
			
		# Get grid cell containing point
		bbox_min = [self.bbox.x.min, self.bbox.y.min, self.bbox.z.min]
		cell_indices = tuple(np.floor((point - bbox_min) / self.grid_cell_size).astype(int))
		
		# Search cell and immediate neighbors
		min_dist = float('inf')
		closest_elem = -1
		
		# Check neighboring cells in a 3x3x3 region
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				for dz in [-1, 0, 1]:
					neighbor_cell = (cell_indices[0] + dx, 
								   cell_indices[1] + dy,
								   cell_indices[2] + dz)
					
					if neighbor_cell in self.hash_grid:
						# Check elements in this cell
						for elem_idx in self.hash_grid[neighbor_cell]:
							dist = np.sum((self.elem_centers[elem_idx] - point)**2)
							if dist < min_dist:
								min_dist = dist
								closest_elem = elem_idx
								
		return closest_elem

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
	
	def get_boundary_elements(self) -> np.ndarray:
		"""Find elements that lie on the boundary of the mesh.
		
		Returns:
			np.ndarray: Array of element indices that are on the boundary
		"""
		# An element is on the boundary if it has less than 6 neighbors
		# (count valid neighbors in elemNeighborsArray)
		valid_neighbors = (self.elemNeighborsArray != -1).sum(axis=1)
		boundary_elements = np.where(valid_neighbors < 26)[0]
		
		# Alternative: element is on boundary if any of its nodes are boundary nodes
		boundary_nodes = self.get_boundary_nodes()
		boundary_elements_2 = np.unique(np.where(np.isin(self.elemArray, boundary_nodes))[0])
		
		# Combine both criteria to be conservative
		boundary_elements = np.unique(np.concatenate([boundary_elements, boundary_elements_2]))
		
		return boundary_elements
	
	def get_boundary_surface_quads(self) -> tuple[np.ndarray, np.ndarray]:
		"""Find quadrilateral faces on the boundary surface of the mesh.
		
		Returns:
			tuple[np.ndarray, np.ndarray]: 
				- Array of shape (N,4) containing node indices for each quad
				- Array of shape (N) containing element indices that each quad belongs to
		"""
		# Get boundary elements
		boundary_elems = self.get_boundary_elements()
		
		# For hex8 elements, faces are defined by these node patterns
		face_patterns = [
			[0,3,2,1],  # -z face
			[4,5,6,7],  # +z face 
			[0,1,5,4],  # -y face
			[3,7,6,2],  # +y face
			[0,4,7,3],  # -x face
			[1,2,6,5]   # +x face
		]
		
		quads = []
		quad_elem_indices = []
		
		# For each boundary element
		for elem_idx in boundary_elems:
			# Get global node indices for this element
			elem_nodes = self.elemArray[elem_idx]
			
			# Check each face
			for face_nodes in face_patterns:
				# Get nodes for this face
				quad = elem_nodes[face_nodes]
				
				# Check if this is a boundary face by looking for another element
				# that shares these nodes
				is_boundary = True
				for other_elem in self.elemNeighborsArray[elem_idx]:
					if other_elem != -1:
						other_nodes = set(self.elemArray[other_elem])
						if all(node in other_nodes for node in quad):
							is_boundary = False
							break
				
				if is_boundary:
					quads.append(quad)
					quad_elem_indices.append(elem_idx)
		
		return np.array(quads), np.array(quad_elem_indices)
	
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


	def plot(self,title="Voxelized Mesh"):
		plotter = pv.Plotter()	
		plotter.add_title(title)
		
		# Add voxelized mesh with component colors
		if self.num_components == 1:
			# Simple case - just plot the voxels
			plotter.add_mesh(self.voxels, show_edges=True)
		else:
			# Multi-component case - plot each component with different colors
			for i in range(self.num_components):
				color = [i/self.num_components, 1.0, 1 - i/self.num_components]
				component_cells = self.voxels.threshold(i + 1, scalars="component_id")
				if component_cells.n_cells > 0:
					plotter.add_mesh(component_cells, color=color, label=f'Component {i}', show_edges=True)

			plotter.add_legend()
		plotter.add_axes()
		plotter.show_grid()
		plotter.show()
	
	
if __name__ == "__main__":
    import os
    import time
    mesh = Mesher()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stlFileName = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
    stlFileName = os.path.join(script_dir, '../Models/FilletedBeam/FilletedBeam.STL')
    mesh.createMeshFromSTLFile(stlFileName, nElemsDesired=10000)
    mesh.plot()


