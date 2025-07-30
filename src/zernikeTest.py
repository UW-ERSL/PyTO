import numpy as np
from ZMPY3D import zernike # pip install ZMPY3D
from skimage import measure
import matplotlib.pyplot as plt

# Load voxel data (binary 0/1 or float in [0,1])
# Example: sphere for demo
def make_sphere(n=32):
    x, y, z = np.indices((n, n, n)) - n // 2
    r = np.sqrt(x**2 + y**2 + z**2)
    return (r < (n // 2 - 1)).astype(float)

voxel_grid = make_sphere(32)  # replace with your own bunny .npy if available

# Normalize to unit ball: scale indices
coords = np.linspace(-1, 1, voxel_grid.shape[0])
x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')

# Mask inside unit sphere
mask = x**2 + y**2 + z**2 <= 1
f = voxel_grid[mask]
xyz = np.stack([x[mask], y[mask], z[mask]], axis=-1)

# Compute Zernike moments
max_order = 10
moments = zernike.compute_moments(xyz, f, max_order)

# Reconstruct shape from moments
recon = np.zeros_like(voxel_grid)
coords_flat = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
recon_vals = zernike.evaluate_moments(moments, coords_flat)
recon = recon_vals.reshape(voxel_grid.shape)

# Plot central slices
plt.subplot(1, 2, 1)
plt.imshow(voxel_grid[:, :, voxel_grid.shape[2] // 2])
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(recon[:, :, recon.shape[2] // 2])
plt.title("Reconstructed")
plt.show()
