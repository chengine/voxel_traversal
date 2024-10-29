#%%
import numpy as np
import torch
from pathlib import Path
from traversal_utils import VoxelGrid
import open3d as o3d
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pcd_path = Path(
    "pcd_4050.pt"
)
bounds = torch.tensor([[-4, 4], [-4, 4], [-1.0, 2.0]], device=device)
pcd = torch.load(pcd_path, map_location=device)

# Filter Point Cloud
n_raw = pcd.shape[0]
maskx = (pcd[:, 0] > bounds[0][0]) & (pcd[:, 0] < bounds[0][1])
masky = (pcd[:, 1] > bounds[1][0]) & (pcd[:, 1] < bounds[1][1])
maskz = (pcd[:, 2] > bounds[2][0]) & (pcd[:, 2] < bounds[2][1])
mask = maskx & masky & maskz
pcd = pcd[mask]
n_filtered = pcd.shape[0]

xyzs = pcd[:, :3]
rgbs = pcd[:, 3:6]
origins = pcd[:, 7:10]
directions = xyzs - origins

directions_normalized = directions / torch.norm(directions, dim=-1, keepdim=True)

param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': bounds[:, 0],
    'upper_bound': bounds[:, 1],
    'voxel_grid_values': None
}

vgrid = VoxelGrid(param_dict, 3, device)
mesh = vgrid.create_mesh_from_points(xyzs, rgbs)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzs.cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())

o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([mesh])

# Now create the information metric from the viewing directions

tnow = time.time()
torch.cuda.synchronize()

voxel_index, in_bounds = vgrid.compute_voxel_index(xyzs)
voxel_index = voxel_index[in_bounds]
directions_normalized = directions_normalized[in_bounds]

# Create outer product from normalized directions
outer_product = torch.einsum("bi,bj->bij",directions_normalized, directions_normalized)

voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(vgrid.discretizations)), device=vgrid.device)

# Store the number of counts for each voxel
voxel_counts = torch.zeros(tuple(vgrid.discretizations), device=vgrid.device, dtype=voxel_index_flatten.dtype).flatten()
voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))
voxel_counts = voxel_counts.reshape(tuple(vgrid.discretizations))

# Store the covariance matrix for each voxel
voxel_covariance = torch.zeros(tuple(vgrid.discretizations) + (3, 3), device=vgrid.device, dtype=outer_product.dtype).reshape(-1, 3, 3)
voxel_covariance.scatter_add_(0, voxel_index_flatten.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3), outer_product)

eigvals, eigvecs = torch.linalg.eigh(voxel_covariance)

# Calculate the eccentricity of the covariances by taking the ratio of the largest to smallest eigenvalues
eccentricity = eigvals[:, 0] / (eigvals[:, 2] + 1e-6)

# Calculate the color of the voxel based on the eccentricity by passing it through a colormap

voxel_vals = (eccentricity - eccentricity.min()) / (eccentricity.max() - eccentricity.min())

torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)

# Create the colormap
cmap = plt.get_cmap('turbo')
voxel_colors = torch.tensor( cmap(voxel_vals.cpu().numpy())[:, :3], device=vgrid.device, dtype=torch.float32)

voxel_colors = voxel_colors.reshape(tuple(vgrid.discretizations) + (3,))

# Create the mesh
mask = voxel_counts > 0

grid_centers = vgrid.voxel_grid_centers[mask]
grid_centers = grid_centers.view(-1, 3).cpu().numpy()

grid_colors = voxel_colors[mask]
grid_colors = grid_colors.view(-1, 3).cpu().numpy()

grid_colors = grid_colors / voxel_counts[mask].unsqueeze(1).cpu().numpy()

scene = o3d.geometry.TriangleMesh()
for cell_color, cell_center in zip(grid_colors, grid_centers):
    box = o3d.geometry.TriangleMesh.create_box(width=vgrid.cell_sizes[0].cpu().numpy(), 
                                                height=vgrid.cell_sizes[1].cpu().numpy(), 
                                                depth=vgrid.cell_sizes[2].cpu().numpy())
    box = box.translate(cell_center, relative=False)
    box.paint_uniform_color(cell_color)
    scene += box
o3d.visualization.draw_geometries([scene])

#%%