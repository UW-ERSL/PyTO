

import numpy as np
from scipy.sparse import coo_matrix
import mesher


def createSmoothingFilter(mesh: mesher.Mesher):
	## Prepare filter
	nfilter = int(27 * mesh.num_elems)
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
	cc = 0

	elemNeighborsArray = mesh.elemNeighborsArray
	for elem in range(mesh.num_elems):
		elemNeighbors = elemNeighborsArray[elem]
		for neighbor in elemNeighbors:
			if neighbor >= 0:
				r = np.linalg.norm(mesh.elem_centers[elem, :] -
											 		 mesh.elem_centers[int(neighbor), :])
				weight = np.exp(-1*r**2)
				iH[cc] = elem
				jH[cc] = neighbor
				sH[cc] = weight
				cc = cc + 1
	# Finalize assembly and convert to csc format
	H = coo_matrix((sH, (iH, jH)), shape = (mesh.num_elems, mesh.num_elems)).tocsc()
	Hs = np.array(H.sum(1)).squeeze()
	return H, Hs

def createXSymmetryFilter(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about X mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HX: Sparse matrix that when multiplied with density vector enforces X mid-plane symmetry
			HXs: Array of row sums of HX matrix
	"""
	num_elems = mesh.num_elems
	x_mid = (mesh.elem_centers[:, 0].max() + mesh.elem_centers[:, 0].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_x = 2 * x_mid - elemCenter[0]
		otherElemCenter = [mirror_x, elemCenter[1], elemCenter[2]]
		distances = np.linalg.norm(mesh.elem_centers - otherElemCenter, axis=1)
		mirror_idx = np.argmin(distances)
		if (mirror_idx == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idx)
			data.append(0.5)

	HX = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HX

def createYSymmetryFilter(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Y mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HY: Sparse matrix that when multiplied with density vector enforces Y mid-plane symmetry
			HYs: Array of row sums of HY matrix
	"""
	num_elems = mesh.num_elems
	y_mid = (mesh.elem_centers[:, 1].max() + mesh.elem_centers[:, 1].min()) / 2
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_y = 2 * y_mid - elemCenter[1]
		otherElemCenter = [elemCenter[0], mirror_y, elemCenter[2]]
		distances = np.linalg.norm(mesh.elem_centers - otherElemCenter, axis=1)
		mirror_idy = np.argmin(distances)
		if (mirror_idy == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idy)
			data.append(0.5)
	
	HY = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HY
	
def createZSymmetryFilter(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Z mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZ: Sparse matrix that when multiplied with density vector enforces Z mid-plane symmetry
			HZs: Array of row sums of HZ matrix
	"""
	num_elems = mesh.num_elems
	z_mid = (mesh.elem_centers[:, 2].max() + mesh.elem_centers[:, 2].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_z = 2 * z_mid - elemCenter[2]
		otherElemCenter = [elemCenter[0], elemCenter[1], mirror_z]
		distances = np.linalg.norm(mesh.elem_centers - otherElemCenter, axis=1)
		mirror_idz = np.argmin(distances)
		if (mirror_idz == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idz)
			data.append(0.5)

	HZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HZ

def createAngularSymmetryFilter(mesh: mesher.Mesher, n_fold: int) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix for n-fold angular symmetry about Z axis.
	
	Args:
		mesh: The mesh object.
		n_fold: Number of symmetric segments 
	
	Returns:
		tuple containing:
			HA: Sparse matrix that enforces angular symmetry
			HAs: Array of row sums of HA matrix
	"""

	num_elems = mesh.num_elems
	angle_step = 2 * np.pi / n_fold
	
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		r = np.sqrt(elemCenter[0]**2 + elemCenter[1]**2)
		theta = np.arctan2(elemCenter[1], elemCenter[0])
		
		weight = 1.0 / n_fold
		rows.append(i)
		cols.append(i)
		data.append(weight)
		
		# Vectorized computation for all k at once
		k_values = np.arange(1, n_fold)
		new_thetas = theta + k_values[:, None] * angle_step
		new_x = r * np.cos(new_thetas)
		new_y = r * np.sin(new_thetas)
		
		# Create array of other element centers for all k
		otherElemCenters = np.column_stack((new_x.flatten(), 
										   new_y.flatten(), 
										   np.full_like(new_x.flatten(), elemCenter[2])))
		
		# Compute all distances at once
		distances = np.linalg.norm(mesh.elem_centers[None, :, :] - otherElemCenters[:, None, :], axis=2)
		sym_indices = np.argmin(distances, axis=1)
		
		# Append to COO matrix components
		rows.extend([i] * (n_fold - 1))
		cols.extend(sym_indices)
		data.extend([weight] * (n_fold - 1))
	
	HAZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HAZ


def createXExtrudeFilter(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix for extruding elements all the way through in the X direction.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HXE: Sparse matrix that when multiplied with density vector enforces X extrusion
			HXEs: Array of row sums of HXE matrix
	"""
	num_elems = mesh.num_elems
	x_max = mesh.elem_centers[:, 0].max()
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		extrude_x = x_max + (x_max - elemCenter[0])
		otherElemCenter = [extrude_x, elemCenter[1], elemCenter[2]]
		distances = np.linalg.norm(mesh.elem_centers - otherElemCenter, axis=1)
		extrude_idx = np.argmin(distances)
		if (extrude_idx == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(extrude_idx)
			data.append(0.5)

	HXE = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	HXEs = np.array(HXE.sum(1)).squeeze()
	return HXE, HXEs

def createZBuildFilter(mesh: mesher.Mesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix to enforce z-direction build constraints for additive manufacturing.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZB: Sparse matrix that enforces material must be supported from below
			HZBs: Array of row sums of HZB matrix
	"""
	num_elems = mesh.num_elems
	z_min = mesh.elem_centers[:, 2].min()
	
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		elem_height = elemCenter[2] - z_min
		
		# Find all elements below current element
		mask = (mesh.elem_centers[:, 0] == elemCenter[0]) & \
				(mesh.elem_centers[:, 1] == elemCenter[1]) & \
				(mesh.elem_centers[:, 2] < elemCenter[2])
				
		if np.any(mask):
			# Connect to all elements below
			below_elems = np.where(mask)[0]
			weight = 1.0 / (len(below_elems) + 1)
			
			rows.extend([i] * (len(below_elems) + 1))
			cols.extend(list(below_elems) + [i])
			data.extend([weight] * (len(below_elems) + 1))
		else:
			# Element is at bottom or has no support
			rows.append(i)
			cols.append(i) 
			data.append(1.0)

	HZAM = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	HZAMs = np.array(HZAM.sum(1)).squeeze()
	return HZAM, HZAMs