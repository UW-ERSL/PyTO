

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