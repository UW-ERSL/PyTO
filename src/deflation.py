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
try:
	import cupy as cp
except ImportError:
  cp = None
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


	def __init__(self, minNodesPerGroup: int = 15,use_gpu: bool = False):	
		"""Initialize the deflation solver with default parameters."""
		self.minNodesPerGroup = minNodesPerGroup
		self.maxGroups = 2000
		self.minGroups = 10
		self.dofPerGroup = 500
		self.use_gpu = use_gpu
		if cp is None:
			use_gpu = False

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

	def create_deflation_groups_manual(self, meshData, nGroupsDesired: int):
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
		
		# Make group lengths slightly increase along X
		# We'll use a linear scaling: length_x = base_length * (1 + alpha * (i / nX))
		alpha = 1 # increase from left to right
		base_sizeX = xLength / (nX * (1 + 0.5 * alpha))  # normalize so total length matches

		sizeX_arr = np.array([base_sizeX * (1 + alpha * (i / max(nX-1,1))) for i in range(nX)])
		sizeX_cumsum = np.concatenate(([0], np.cumsum(sizeX_arr)))
		# Now, for each node, find which X group it belongs to
		x_rel = xyz[:, 0] - xMin
		x_group = np.searchsorted(sizeX_cumsum, x_rel, side='right') - 1
		x_group = np.clip(x_group, 0, nX-1)
		# Y and Z are uniform
		sizeY = yLength / nY
		sizeZ = zLength / nZ
		y_group = np.floor((xyz[:, 1] - yMin) / sizeY).astype(np.int32)
		y_group = np.clip(y_group, 0, nY-1)
		z_group = np.floor((xyz[:, 2] - zMin) / sizeZ).astype(np.int32)
		z_group = np.clip(z_group, 0, nZ-1)
		indices = np.stack([x_group, y_group, z_group], axis=1)
	
		
		# Initialize arrays for group assignments
		nodeGroupNumber = np.zeros(meshData.num_nodes, dtype=np.int32)
		groupCount = np.zeros(nGroupsTentative, dtype=np.int32)
		groupCenter = np.zeros((nGroupsTentative, 3))
		

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

		print("Number of deflation groups: ", self.ws_nGroups)
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
		if(False):
			plotter.add_mesh(points, scalars='groups', 
							point_size=7,
							render_points_as_spheres=True,
							cmap='rainbow')

		# Add group centers as black spheres
		group_centers = self.ws_groupCenter
		centers_cloud = pv.PolyData(group_centers)
		plotter.add_mesh(centers_cloud, color='black', point_size=6, render_points_as_spheres=True)

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
		if self.use_gpu:
			return self.deflatedPCG_GPU(K, f, W, M, rtol, maxIters, verbose)
		else:
			return self.deflatedPCG_CPU(K, f, W, M, rtol, maxIters, verbose)

	def deflatedPCG_CPU(self,
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
		#print("Deflated PCG iterations:", iter_num + 1)
		if (iter_num == maxIters - 1):
			print("Warning: Maximum iterations reached in DPCG; relative residual:", np.sqrt(rz_new/rz0))
		return x

	def deflatedPCG_GPU(self,
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
		
		# Convert arrays to GPU
		x_gpu = cp.array(x)
		r_gpu = cp.array(r)
		z_gpu = cp.array(z)
		p_gpu = cp.array(p)
		Kp_gpu = cp.array(Kp)
		
		K_gpu = cp.sparse.csr_matrix(K)
		W_gpu = cp.sparse.csr_matrix(W)
		WT_gpu = cp.sparse.csr_matrix(WT)
		M_gpu = cp.sparse.csr_matrix(M)

		for iter_num in range(maxIters):
			Kp_gpu = K_gpu @ p_gpu
			pKp = float(p_gpu.dot(Kp_gpu))
			
			alpha = rz / pKp
			x_gpu += alpha * p_gpu
			r_gpu -= alpha * Kp_gpu
			
			z_gpu = M_gpu @ r_gpu
			rz_new = float(r_gpu.dot(z_gpu))
			if np.sqrt(rz_new/rz0) <= rtol:
				if verbose:
					print(f"Converged in {iter_num + 1} iterations")
				break
			
			beta = rz_new / rz
			Kz_gpu = K_gpu @ z_gpu
			
			# Transfer back to CPU for direct solve
			Kz = cp.asnumpy(Kz_gpu)
			if (pypardiso is None):
				mu = spy_linalg.spsolve(WKW,WT @ Kz)
			else:
				mu = pypardiso.spsolve(WKW,WT @ Kz)
				mu_gpu = cp.array(mu)
			
			p_gpu = z_gpu + beta * p_gpu - W_gpu @ mu_gpu
			
			rz = rz_new

		# Transfer final result back to CPU
		x = cp.asnumpy(x_gpu)
		#print("Deflated PCG iterations:", iter_num + 1)
		if (iter_num == maxIters - 1):
			print("Warning: Maximum iterations reached in DPCG; relative residual:", np.sqrt(rz_new/rz0))
		return x
	def deflatedPCG_with_LagrangeMultipliers(self,
											K: _Array,
											f: np.ndarray,
											C: _Array,
											b: np.ndarray,
											W: _Array,
											M: _Array,
											rtol=1e-8,
											maxIters=500,
											verbose=True):
		"""Solve using null-space reduction method."""
		
		import scipy.sparse.linalg as spla
		from scipy.linalg import null_space
		
		n = f.shape[0]
		m = C.shape[0]
		
		if verbose:
			print(f"Null-space reduction: {n} DOFs, {m} constraints")
		
		CT = C.transpose(copy=True)
		
		# Step 1: Find particular solution d_p such that C*d_p = b
		# Solve: C^T * lambda_p = b using (C*C^T)*lambda_p = b
		CCT = (C @ CT).toarray()
		try:
			lambda_p = np.linalg.solve(CCT, b)
		except:
			lambda_p = np.linalg.lstsq(CCT, b, rcond=None)[0]
		
		d_p = (CT @ lambda_p).ravel()
		
		# Verify particular solution
		constraint_error = np.linalg.norm(C @ d_p - b)
		if verbose:
			print(f"Particular solution constraint error: {constraint_error:.3e}")
		
		# Step 2: Find null-space basis Z of C (C*Z = 0)
		if m >= n:
			raise ValueError("Over-constrained system: m >= n")
		
		if m < 5000:
			CT_dense = CT.toarray()
			Z = null_space(CT_dense.T)  # Columns of Z span null(C)
			if verbose:
				print(f"Null-space dimension: {Z.shape[1]} (expected {n-m})")
		else:
			# For very large m, this is expensive
			print("Warning: Computing null-space for large constraint matrix")
			# Use sparse SVD approximation
			from scipy.sparse.linalg import svds
			k = min(m, 100)
			try:
				U, s, Vt = svds(C, k=k, which='SM')
				tol = 1e-8 * s.max() if len(s) > 0 else 1e-8
				null_idx = s < tol
				if np.any(null_idx):
					Z = Vt[null_idx, :].T
				else:
					raise ValueError("No null space found in sparse SVD")
			except:
				# Fallback to dense computation
				C_dense = C.toarray()
				Z = null_space(C_dense)
		
		# Step 3: Build reduced system K_tilde = Z^T * K * Z
		ZT = Z.T
		KZ = K @ Z
		K_tilde = ZT @ KZ
		
		# Build reduced RHS f_tilde = Z^T * (f - K*d_p)
		f_tilde = ZT @ (f - K @ d_p)
		
		if verbose:
			print(f"Reduced system size: {K_tilde.shape[0]}")
		
		# Step 4: Solve reduced system K_tilde * y = f_tilde
		# This system is typically much better conditioned
		
		# Check if small enough for direct solve
		if K_tilde.shape[0] < 10000:
			# Direct solve
			y = np.linalg.solve(K_tilde, f_tilde)
			if verbose:
				print("Used direct solve on reduced system")
		else:
			# Iterative solve with preconditioner
			# Build preconditioner M_tilde = Z^T * M * Z
			MZ = M @ Z
			M_tilde = ZT @ MZ
			
			def matvec(v):
				return K_tilde @ v
			
			def precond(v):
				return np.linalg.solve(M_tilde, v)
			
			K_op = spla.LinearOperator(K_tilde.shape, matvec=matvec)
			M_op = spla.LinearOperator(K_tilde.shape, matvec=precond)
			
			y, info = spla.gmres(K_op, f_tilde, 
								 M=M_op,
								 rtol=rtol,
								 maxiter=maxIters)
			
			if info == 0:
				if verbose:
					print("GMRES converged on reduced system")
			else:
				print(f"GMRES info: {info}")
		
		# Step 5: Reconstruct full solution d = d_p + Z*y
		d = d_p + Z @ y
		
		# Verify solution
		r_prim = f - K @ d
		r_dual = b - C @ d
		
		# Compute Lagrange multipliers from optimality: K*d + C^T*mu = f
		# So: C^T*mu = f - K*d
		# Solve: (C*C^T)*mu = C*(f - K*d)
		rhs_mu = C @ r_prim
		try:
			mu = np.linalg.solve(CCT, rhs_mu)
		except:
			mu = np.linalg.lstsq(CCT, rhs_mu, rcond=None)[0]
		
		# Final residual with multipliers
		r_prim_full = f - K @ d - CT @ mu
		
		if verbose:
			print(f"Final residuals:")
			print(f"  ||f - K*d - C^T*mu|| = {np.linalg.norm(r_prim_full):.3e}")
			print(f"  ||b - C*d||          = {np.linalg.norm(r_dual):.3e}")
		
		return d