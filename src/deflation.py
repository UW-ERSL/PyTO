"""Deflation Solver Module.

This module implements deflation techniques for solving large-scale linear systems.
It provides methods for creating deflation groups and solving systems using 
deflated preconditioned conjugate gradient method.

The deflation approach helps improve convergence by handling multiple scales
in the problem domain efficiently.
"""

from typing import TypeAlias, Union
import numpy as np
import scipy.sparse as spy_sprs
import scipy.linalg as spy_linalg
from hex_mesher import HexMesher
# Mac does not support pypardiso, so we skip it for now
try:
  import pypardiso # pip install pypardiso
except ImportError:
  pypardiso = None
import scipy
import pyvista as pv
from scipy.sparse import csr_matrix

_Array: TypeAlias = Union[spy_sprs.coo_matrix,
													spy_sprs.csr_matrix,
													spy_sprs.csc_matrix,
													np.ndarray]


class DeflationSolver:
	"""Deflation-based solver for large linear systems.
	
	The solver uses domain decomposition and deflation to improve convergence
	of iterative solvers for problems with multiple scales or poor conditioning.

	Attributes:
		minNodesPerGroup (int): Minimum number of nodes allowed in a deflation group
		ws_nGroups (int): Number of deflation groups
		ws_nodeGroupNumber (ndarray): Group assignment for each node
		ws_groupCount (ndarray): Number of nodes in each group
		groupCenter (ndarray): Centroid coordinates of each group
		elemPseudoDensity (ndarray): Density values for topology optimization
	"""


	def __init__(self, minNodesPerGroup: int = 15):	
		"""Initialize the deflation solver with default parameters."""
		self.minNodesPerGroup = minNodesPerGroup
		self.maxGroups = 2000
		self.minGroups = 10
		self.dofPerGroup = 500


	def setPseudoDensity(self,rho):
		"""Set pseudo-density values for topology optimization.

		Args:
			rho (ndarray): Array of density values for each element
		"""
		self.elemPseudoDensity = rho.copy()


	def create_deflation_groups(self, meshData, nGroupsDesired: int):
		"""Create deflation groups using geometric partitioning.

		This method divides the domain into approximately equal-sized boxes
		and assigns nodes to groups based on their spatial location. It also
		handles cases where some groups have too few nodes by merging them
		with nearby groups.

		Args:
			meshData: Mesh data object containing node and element information
			nGroupsDesired (int): Target number of deflation groups
			
		Returns:
			bool: True if grouping was successful
			
		Algorithm Steps:
		1. Calculate domain dimensions and group sizes
		2. Create initial groups based on spatial coordinates
		3. Merge small groups with neighbors
		4. Compute final group centers and counts
		"""
		# Limit number of groups based on minimum nodes per group
		nGroupsDesired = min(nGroupsDesired, 
						   int(meshData.num_nodes/(1 + self.minNodesPerGroup)))
		
		xyz = meshData.node_xyz

		xMin = np.min(xyz[:,0])
		yMin = np.min(xyz[:,1])
		zMin = np.min(xyz[:,2])		
		xLength = np.max(xyz[:,0]) - xMin
		yLength = np.max(xyz[:,1]) - yMin
		zLength = np.max(xyz[:,2]) - zMin

		# Calculate group dimensions to achieve desired number of groups
		temp = xLength * yLength * zLength
		alpha = (nGroupsDesired / temp) ** (1.0 / 3)
		
		nX = max(round(alpha*xLength),1)
		nY = max(round(alpha*yLength),1)
		nZ = max(round(alpha*zLength),1)

		# Initialize group data structures
		nGroupsTentative = nX * nY * nZ
		#print("Group dimensions:", [nX, nY, nZ])
		
		# Calculate group sizes
		sizeX = xLength / nX
		sizeY = yLength / nY
		sizeZ = zLength / nZ
		
		# Initialize arrays for group assignments
		nodeGroupNumber = np.zeros(meshData.num_nodes, dtype=np.int32)
		groupCount = np.zeros(nGroupsTentative, dtype=np.int32)
		groupCenter = np.zeros((nGroupsTentative, 3))
		
		#print("Number of tentative groups:", nGroupsTentative)
		
		# Assign nodes to groups using vectorized operations
		rel_pos = xyz - np.array([xMin, yMin, zMin])
		indices = np.floor(rel_pos / np.array([sizeX, sizeY, sizeZ])).astype(np.int32)
		indices = np.minimum(indices, np.array([nX - 1, nY - 1, nZ - 1]))

		# Compute group IDs for all nodes at once
		nodeGroupNumber = (indices[:, 0] + 
						nX * indices[:, 1] + 
						nX * nY * indices[:, 2]).astype(np.int32)

		# Count nodes per group using numpy
		groupCount = np.bincount(nodeGroupNumber, minlength=nGroupsTentative)

		# Compute group centers using vectorized operations
		groupCenter = np.zeros((nGroupsTentative, 3))
		for i in range(3):
			np.add.at(groupCenter[:, i], nodeGroupNumber, xyz[:, i])

		for group in range(nGroupsTentative):
			if (groupCount[group] > 0):
				for i in range(3):
					groupCenter[group,i] /=  groupCount[group] # we will need this for reassignment	
		# Identify groups with very few nodes
		groupMapping = np.zeros(nGroupsTentative,dtype = np.int32)
		currentGroupNumber  = 0
		for group in range(nGroupsTentative):
			if (groupCount[group] < self.minNodesPerGroup):
				groupMapping[group] = -1
			else:
				groupMapping[group] = currentGroupNumber
				currentGroupNumber = currentGroupNumber+1
		self.ws_nGroups = currentGroupNumber
		#print("Number of new groups: ", self.ws_nGroups)
		if (any(groupMapping == -1)):
			# Find small groups that need reassignment
			small_groups = np.where((groupMapping == -1) & (groupCount > 0))[0]
			valid_groups = np.where(groupCount >= self.minNodesPerGroup)[0]

			if len(small_groups) > 0:
				# Compute distances between all small groups and valid groups at once
				small_centers = groupCenter[small_groups][:, np.newaxis, :]  # Shape: (n_small, 1, 3)
				valid_centers = groupCenter[valid_groups][np.newaxis, :, :]  # Shape: (1, n_valid, 3)
				distances = np.linalg.norm(small_centers - valid_centers, axis=2)  # Shape: (n_small, n_valid)
				
				# Find closest valid group for each small group
				closest_valid_indices = valid_groups[np.argmin(distances, axis=1)]
				
				# Update group mapping
				groupMapping[small_groups] = groupMapping[closest_valid_indices]

			# assign nodes
			groupCount = np.zeros(self.ws_nGroups,dtype = np.int32)
			groupCenter = np.zeros((self.ws_nGroups ,3))
			# Replace the node-by-node loop with vectorized operations
			valid_nodes = nodeGroupNumber != -1
			nodeGroupNumber[valid_nodes] = groupMapping[nodeGroupNumber[valid_nodes]]

			# Count nodes per group using numpy
			groupCount = np.bincount(nodeGroupNumber[valid_nodes], minlength=self.ws_nGroups)

			# Compute group centers using vectorized operations
			groupCenter = np.zeros((self.ws_nGroups, 3))
			for i in range(3):
				np.add.at(groupCenter[:, i], nodeGroupNumber[valid_nodes], xyz[valid_nodes, i])
			for group in range(self.ws_nGroups):
				for i in range(3):
					groupCenter[group,i] /=  groupCount[group] 
		
		# Finally copy the data structures
		self.ws_nodeGroupNumber = nodeGroupNumber
		self.ws_groupCount = groupCount
		self.ws_groupCenter = groupCenter


		if (any(self.ws_groupCount < self.minNodesPerGroup)):
			print('Warning: Groups with very few nodes... might lead to numerical issues')
			print('Smallest group size:', np.min(self.ws_groupCount))
			input('Continue?')

		if (np.sum(self.ws_groupCount) != meshData.num_nodes):
			print('Invalid assignment of nodes to groups. Technical bug.')
			return False

		#print("Number of deflation groups: ", self.ws_nGroups)
		return True

	def create_deflation_groups_adaptive(self, meshData, nGroupsDesired: int, nodalStrainEnergy: np.ndarray):
		"""Create deflation groups using geometric partitioning.

		
		"""
		"""Create deflation groups with adaptive sizing based on strain energy.

		This method divides the domain into variable-sized boxes where regions with higher
		strain energy get smaller groups for better resolution.

		Args:
			meshData: Mesh data object containing node and element information
			nGroupsDesired (int): Target number of deflation groups
			nodalStrainEnergy (ndarray): Nodal strain energy values
			
		Returns:
			bool: True if grouping was successful

		Algorithm Steps:
		1. Calculate domain dimensions and group sizes based on strain energy
		2. Create groups with variable sizes based on local strain energy
		3. Merge small groups with neighbors
		4. Compute final group centers and counts
		"""
		# Limit number of groups based on minimum nodes per group
		nGroupsDesired = min(nGroupsDesired, 
						   int(meshData.num_nodes/(1 + self.minNodesPerGroup)))

		xyz = meshData.node_xyz

		xMin = np.min(xyz[:,0])
		yMin = np.min(xyz[:,1])
		zMin = np.min(xyz[:,2])        
		xLength = np.max(xyz[:,0]) - xMin
		yLength = np.max(xyz[:,1]) - yMin
		zLength = np.max(xyz[:,2]) - zMin

		# Calculate num of divisions along each direction to achieve desired number of groups
		temp = xLength * yLength * zLength
		alpha = (nGroupsDesired / temp) ** (1.0 / 3)

		nX = max(round(alpha*xLength),1)
		nY = max(round(alpha*yLength),1)
		nZ = max(round(alpha*zLength),1)

		# Initialize group data structures
		nGroupsTentative = nX * nY * nZ

		print("Base group dimensions:", [nX, nY, nZ])
		print(nGroupsTentative, " tentative groups")

		xSegments = np.linspace(xMin, xMin + xLength, nX + 1)
		ySegments = np.linspace(yMin, yMin + yLength, nY + 1)		
		zSegments = np.linspace(zMin, zMin + zLength, nZ + 1)

		scaling = 2
		xMinSize = xLength / nX/scaling
		yMinSize = yLength / nY/scaling
		zMinSize = zLength / nZ/scaling

	
		# Sort nodes by x coordinate
		sorted_indices = np.argsort(xyz[:, 0])
		x_sorted = xyz[sorted_indices, 0]
		se_sorted = nodalStrainEnergy[sorted_indices]

		# Compute cumulative strain energy along x
		cum_se = np.cumsum(se_sorted)
		total_se = cum_se[-1]
		target_se_per_segment = total_se / nX

		# Find xSegments so that each segment has equal strain energy, but not smaller than xMinSize
		xSegments = [xMin]
		last_idx = 0
		for i in range(1, nX):
			target_se = i * target_se_per_segment
			# Find the index where cumulative strain energy exceeds the target
			idx = np.searchsorted(cum_se, target_se)
			# Ensure minimum segment size
			while idx < len(x_sorted) and (x_sorted[idx] - xSegments[-1]) < xMinSize:
				idx += 1
			if idx >= len(x_sorted):
				break
			xSegments.append(x_sorted[idx])
			last_idx = idx
		xSegments.append(xMin + xLength)
		xSegments = np.array(xSegments)

		# Repeat for y direction
		sorted_indices = np.argsort(xyz[:, 1])
		y_sorted = xyz[sorted_indices, 1]
		se_sorted = nodalStrainEnergy[sorted_indices]
		cum_se = np.cumsum(se_sorted)
		total_se = cum_se[-1]
		target_se_per_segment = total_se / nY

		ySegments = [yMin]
		last_idx = 0
		for i in range(1, nY):
			target_se = i * target_se_per_segment
			idx = np.searchsorted(cum_se, target_se)
			while idx < len(y_sorted) and (y_sorted[idx] - ySegments[-1]) < yMinSize:
				idx += 1
			if idx >= len(y_sorted):
				break
			ySegments.append(y_sorted[idx])
			last_idx = idx
		ySegments.append(yMin + yLength)
		ySegments = np.array(ySegments)
		
		# Repeat for z direction
		sorted_indices = np.argsort(xyz[:, 2])
		z_sorted = xyz[sorted_indices, 2]
		se_sorted = nodalStrainEnergy[sorted_indices]
		cum_se = np.cumsum(se_sorted)
		total_se = cum_se[-1]
		target_se_per_segment = total_se / nZ

		zSegments = [zMin]
		last_idx = 0
		for i in range(1, nZ):
			target_se = i * target_se_per_segment
			idx = np.searchsorted(cum_se, target_se)
			while idx < len(z_sorted) and (z_sorted[idx] - zSegments[-1]) < zMinSize:
				idx += 1
			if idx >= len(z_sorted):
				break
			zSegments.append(z_sorted[idx])
			last_idx = idx
		zSegments.append(zMin + zLength)
		zSegments = np.array(zSegments)
		
		# Assign nodes to groups using adaptive segments
		ix = np.searchsorted(xSegments, xyz[:, 0], side='right') - 1
		iy = np.searchsorted(ySegments, xyz[:, 1], side='right') - 1
		iz = np.searchsorted(zSegments, xyz[:, 2], side='right') - 1

		ix = np.clip(ix, 0, len(xSegments) - 2)
		iy = np.clip(iy, 0, len(ySegments) - 2)
		iz = np.clip(iz, 0, len(zSegments) - 2)

		nodeGroupNumber = (ix + nX * iy + nX * nY * iz).astype(np.int32)
		

	
		groupCount = np.zeros(nGroupsTentative, dtype=np.int32)
		groupCenter = np.zeros((nGroupsTentative, 3))

	

		# Count nodes per group using numpy
		groupCount = np.bincount(nodeGroupNumber, minlength=nGroupsTentative)

		# Compute group centers using vectorized operations
		groupCenter = np.zeros((nGroupsTentative, 3))
		for i in range(3):
			np.add.at(groupCenter[:, i], nodeGroupNumber, xyz[:, i])

		for group in range(nGroupsTentative):
			if (groupCount[group] > 0):
				for i in range(3):
					groupCenter[group,i] /= groupCount[group] # we will need this for reassignment    

		# Identify groups with very few nodes
		groupMapping = np.zeros(nGroupsTentative, dtype=np.int32)
		currentGroupNumber = 0
		for group in range(nGroupsTentative):
			if (groupCount[group] < self.minNodesPerGroup):
				groupMapping[group] = -1
			else:
				groupMapping[group] = currentGroupNumber
				currentGroupNumber = currentGroupNumber+1
		self.ws_nGroups = currentGroupNumber
		#print("Number of new groups: ", self.ws_nGroups)

		if (any(groupMapping == -1)):
			# Find small groups that need reassignment
			small_groups = np.where((groupMapping == -1) & (groupCount > 0))[0]
			valid_groups = np.where(groupCount >= self.minNodesPerGroup)[0]

			if len(small_groups) > 0:
				# Compute distances between all small groups and valid groups at once
				small_centers = groupCenter[small_groups][:, np.newaxis, :]  # Shape: (n_small, 1, 3)
				valid_centers = groupCenter[valid_groups][np.newaxis, :, :]  # Shape: (1, n_valid, 3)
				distances = np.linalg.norm(small_centers - valid_centers, axis=2)  # Shape: (n_small, n_valid)
				
				# Find closest valid group for each small group
				closest_valid_indices = valid_groups[np.argmin(distances, axis=1)]
				
				# Update group mapping
				groupMapping[small_groups] = groupMapping[closest_valid_indices]

			# assign nodes
			groupCount = np.zeros(self.ws_nGroups, dtype=np.int32)
			groupCenter = np.zeros((self.ws_nGroups, 3))
			# Replace the node-by-node loop with vectorized operations
			valid_nodes = nodeGroupNumber != -1
			nodeGroupNumber[valid_nodes] = groupMapping[nodeGroupNumber[valid_nodes]]

			# Count nodes per group using numpy
			groupCount = np.bincount(nodeGroupNumber[valid_nodes], minlength=self.ws_nGroups)

			# Compute group centers using vectorized operations
			groupCenter = np.zeros((self.ws_nGroups, 3))
			for i in range(3):
				np.add.at(groupCenter[:, i], nodeGroupNumber[valid_nodes], xyz[valid_nodes, i])
			for group in range(self.ws_nGroups):
				if groupCount[group] > 0:
					for i in range(3):
						groupCenter[group,i] /= groupCount[group]

		# Finally copy the data structures
		self.ws_nodeGroupNumber = nodeGroupNumber
		self.ws_groupCount = groupCount
		self.ws_groupCenter = groupCenter

		if (any(self.ws_groupCount < self.minNodesPerGroup)):
			print('Warning: Groups with very few nodes... might lead to numerical issues')
			print('Smallest group size:', np.min(self.ws_groupCount))
			input('Continue?')

		if (np.sum(self.ws_groupCount) != meshData.num_nodes):
			print('Invalid assignment of nodes to groups. Technical bug.')
			return False

		return True
	
	def create_deflation_matrix(self, meshData):
		# Implements eqn 1 from Yadav, P., Suresh, K., "Large Scale Finite Element Analysis via  ..."
		# The size of the sparse W matrix is (num_nodes, 6*nGroups)
		# For every node, there are 9 non-zero entries in W
		num_nodes = meshData.num_nodes
		total_entries = 9 * num_nodes  # Each dof has exactly 3 non-zero entries, and each node has 3 dof
		
		# Pre-allocate arrays with exact size
		iW = np.empty(total_entries, dtype=np.int32)
		jW = np.empty(total_entries, dtype=np.int32)
		sW = np.empty(total_entries, dtype=np.float64)
		
		# Get node coordinates and group centers as arrays
		xyz = meshData.node_xyz
		
		group_centers = self.ws_groupCenter[self.ws_nodeGroupNumber]
		# Calculate relative positions
		rel_pos = xyz - group_centers
		
		# Generate indices for all nodes at once
		node_indices = np.arange(num_nodes)
		group_indices = self.ws_nodeGroupNumber
	
		# Fill arrays in blocks
		# u dof entries
		base_idx = 0
		iW[base_idx:base_idx + 3*num_nodes:3] = 3 * node_indices
		iW[base_idx + 1:base_idx + 3*num_nodes:3] = 3 * node_indices
		iW[base_idx + 2:base_idx + 3*num_nodes:3] = 3 * node_indices

		jW[base_idx:base_idx + 3*num_nodes:3] = 6 * group_indices
		jW[base_idx + 1:base_idx + 3*num_nodes:3] = 6 * group_indices + 4
		jW[base_idx + 2:base_idx + 3*num_nodes:3] = 6 * group_indices + 5

		sW[base_idx:base_idx + 3*num_nodes:3] = 1
		sW[base_idx + 1:base_idx + 3*num_nodes:3] = rel_pos[:, 2]  # z
		sW[base_idx + 2:base_idx + 3*num_nodes:3] = -rel_pos[:, 1]  # -y
		
		# v dof entries
		base_idx = 3 * num_nodes
		iW[base_idx:base_idx + 3*num_nodes:3] = 3 * node_indices + 1
		iW[base_idx + 1:base_idx + 3*num_nodes:3] = 3 * node_indices + 1
		iW[base_idx + 2:base_idx + 3*num_nodes:3] = 3 * node_indices + 1

		jW[base_idx:base_idx + 3*num_nodes:3] = 6 * group_indices + 1
		jW[base_idx + 1:base_idx + 3*num_nodes:3] = 6 * group_indices + 3
		jW[base_idx + 2:base_idx + 3*num_nodes:3] = 6 * group_indices + 5

		sW[base_idx:base_idx + 3*num_nodes:3] = 1
		sW[base_idx + 1:base_idx + 3*num_nodes:3] = -rel_pos[:, 2]  # -z
		sW[base_idx + 2:base_idx + 3*num_nodes:3] = rel_pos[:, 0]   # x
		
		# w dof entries
		base_idx = 6 * num_nodes
		iW[base_idx:base_idx + 3*num_nodes:3] = 3 * node_indices + 2
		iW[base_idx + 1:base_idx + 3*num_nodes:3] = 3 * node_indices + 2
		iW[base_idx + 2:base_idx + 3*num_nodes:3] = 3 * node_indices + 2
		
		jW[base_idx:base_idx + 3*num_nodes:3] = 6 * group_indices + 2
		jW[base_idx + 1:base_idx + 3*num_nodes:3] = 6 * group_indices + 3
		jW[base_idx + 2:base_idx + 3*num_nodes:3] = 6 * group_indices + 4
		
		sW[base_idx:base_idx + 3*num_nodes:3] = 1
		sW[base_idx + 1:base_idx + 3*num_nodes:3] = rel_pos[:, 1]   # y
		sW[base_idx + 2:base_idx + 3*num_nodes:3] = -rel_pos[:, 0]  # -x
		
	
		self.W = spy_sprs.coo_matrix((sW, (iW, jW)),
															 shape=(3*meshData.num_nodes, 6*self.ws_nGroups)
															 ).tocsr()
	
		#print("Finished computing W Matrix")
		return
	def plot_deflation_groups(self,mesh: HexMesher):
		# Create vertices array
		vertices = mesh.node_xyz.copy()
		
		# Create PyVista point cloud
		points = pv.PolyData(vertices)

		# Add group numbers as scalar data
		points.point_data['groups'] = self.ws_nodeGroupNumber

		# Create plotter
		plotter = pv.Plotter()
		plotter.set_background('white')

		# Add points with group coloring
		plotter.add_mesh(points, scalars='groups', 
						point_size=7,
						render_points_as_spheres=True,
						cmap='rainbow')

		# Add group centers as black spheres
		group_centers = self.ws_groupCenter
		centers_cloud = pv.PolyData(group_centers)
		plotter.add_mesh(centers_cloud, color='black', point_size=15, render_points_as_spheres=True)

		plotter.add_axes()

		# Set camera for isometric view
		plotter.view_isometric()
		plotter.show()

	def deflatedPCG(self,
									K: _Array,
									f: np.ndarray,
									W: _Array,
									M: _Array,
									rtol=1e-8,
									maxIters=2000,
									verbose=False):
		"""Deflated Preconditioned Conjugate Gradient."""

		n = f.shape[0]
		WT = W.transpose(copy=True)
		
		# Pre-compute matrices
		KW = K @ W
		WKW = (WT @ KW).toarray()
		WKW = csr_matrix((WKW + WKW.T) / 2)
		# Add small value to diagonal for numerical stability
		WKW += scipy.sparse.eye(WKW.shape[0]) * 1e-10
		#L = spy_linalg.cho_factor(WKW, lower=True)

		# Pre-allocate vectors
		x = np.zeros(n)
		r = np.zeros(n)
		z = np.zeros(n)
		p = np.zeros(n)
		Kp = np.zeros(n)

		# Initial solution
		if (pypardiso is None):
			mu = spy_linalg.spsolve(WKW,WT @ f) 
		else:
			mu = pypardiso.spsolve(WKW,WT @ f)# see notes on pypardiso.spsolve
		
		x = W @ mu

		# Initial residual
		r = f - K @ x
		z = M @ r

		# Initial search direction
		Kz = K @ z
		
		if (pypardiso is None):
			mu = spy_linalg.spsolve(WKW,WT @ Kz)
		else:
			mu = pypardiso.spsolve(WKW,WT @ Kz)
		
		p = z - W @ mu

		# Initial residual norm
		rz = r.dot(z)
		rz0 = rz
		
		for iter_num in range(maxIters):
			Kp = K @ p
			pKp = p.dot(Kp)
			
			alpha = rz / pKp
			x += alpha * p
			r -= alpha * Kp
			
			z = M @ r
			rz_new = r.dot(z)
			if np.sqrt(rz_new/rz0) <= rtol:
				if verbose:
					print(f"Converged in {iter_num + 1} iterations")
				break
				
			beta = rz_new / rz
			Kz = K @ z
			if (pypardiso is None):
				mu = spy_linalg.spsolve(WKW,WT @ Kz)
			else:
				mu = pypardiso.spsolve(WKW,WT @ Kz)
			p = z + beta * p - W @ mu
			
			rz = rz_new
		print("Deflated PCG iterations:", iter_num + 1)
		if (iter_num == maxIters - 1):
			print("Warning: Maximum iterations reached in DPCG; relative residual:", np.sqrt(rz_new/rz0))
		return x