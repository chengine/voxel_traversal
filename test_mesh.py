#%%
import numpy as np
import torch
import open3d as o3d
import time
import matplotlib
import imageio
import matplotlib.pyplot as plt

from traversal_utils import *
from camera_utils import look_at


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

#obj_path = 'stanford-bunny.obj'
obj_path = 'bond/source/Bond_Test.glb'

# Reads in mesh
mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()

# The stanford bunny is flipped on its side, so we need to rotate it to be upright.
mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=np.zeros(3))

# And make sure it's centered at the origin
mesh.translate(-mesh.get_center())
points = np.asarray(mesh.vertices)

# Compute the bounding box of the mesh to initialize voxel grid
bounding_box = mesh.get_axis_aligned_bounding_box()

# NOTE: Termination function dictates when rays stop marching. For example, if the ray hits
# an occupied cell, we stop marching and return the value of that cell.
def termination_fn(next_points, directions, next_indices, voxel_grid_binary, voxel_grid_values):

    # Possible for indices to be out of bounds
    discretization = torch.tensor(voxel_grid_binary.shape, device=device)
    out_of_bounds = torch.any( torch.logical_or(next_indices < 0, next_indices - discretization.unsqueeze(0) > -1) , dim=-1)

    out_values = torch.zeros_like(next_points).to(torch.float32)

    # Mask out of bounds points
    next_points = next_points[~out_of_bounds]
    directions = directions[~out_of_bounds]
    next_indices = next_indices[~out_of_bounds]

    mask = voxel_grid_binary[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
    values = voxel_grid_values[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]

    out_mask = ~out_of_bounds
    out_mask[~out_of_bounds] = mask

    out_values[~out_of_bounds] = values

    return out_mask, out_values

# Parameter dictionary for the voxel grid
param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
    'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'termination_fn': termination_fn
}

cmap = matplotlib.cm.get_cmap('viridis')
colors = cmap( (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]

# Initialize the voxel grid
vgrid = VoxelGrid(param_dict, 3, device)

# Converts point cloud to voxel grid with colors
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

# Optional: visualize the voxel grid
# scene = vgrid.create_mesh_from_points(torch.tensor(points, device=device), torch.tensor(np.asarray(mesh.vertex_colors), device=device, dtype=torch.float32))
# o3d.visualization.draw_geometries([scene])

#%%

# Camera intrinsics
K = torch.tensor([
    [1000., 0., 500.],
    [0., 1000., 500.],
    [0., 0., 1.]
], device=device)

t = torch.linspace(0., 2*np.pi, 100, device=device)
far_clip = 1.       # far field draw distance

depth_images = []
directions = []
origins = []
for i, t_ in enumerate(t):

    # Generate the camera extrinsics. Its a circle around the object.
    c2w = torch.eye(4, device=device)
    c2w[:3, 3] = torch.tensor([0.25*torch.cos(t_), 0.25*torch.sin(t_), 0.0], device=device)
    c2w[:3, :3] = look_at(c2w[:3, 3], torch.tensor([0., 0., 0.], device=device), torch.tensor([0., 0., 1.], device=device))

    # Do the voxel ray tracing and render images
    tnow = time.time()
    torch.cuda.synchronize()
    image, depth, output = vgrid.camera_voxel_intersection(K, c2w, far_clip)
    depth_normalized = depth / depth.max()
    
    depth_mask = (depth == 0).squeeze()
    torch.cuda.synchronize()
    print("Time taken: ", time.time() - tnow)

    depth_images.append(depth)
    directions.append(output['rays_d'])
    origins.append(output['rays_o'])

    # Optional: Save and visualize the images
    # combined image
    # combined_image = np.concatenate([image.cpu().numpy(), depth_normalized.cpu().numpy().repeat(3, axis=-1)], axis=1)
    # imageio.imwrite(f'output/{i}.png', (combined_image * 255).astype(np.uint8))

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(image.cpu().numpy())
    # ax[1].imshow(depth_normalized, cmap='gray')

### SEGMENT OUT THE RAYS THAT INTERSECTED THE VOXEL GRID ###
depths = np.stack(depth_images, axis=0).reshape(-1)
directions = torch.stack(directions, axis=0).cpu().numpy().reshape(-1, 3)
origins = torch.stack(origins, axis=0).cpu().numpy().reshape(-1, 3)

# For depths greater than 0, this means we hit the voxel grid
mask = depths > 0
depths = depths[mask]
directions = directions[mask]
origins = origins[mask]

# Calculate termination points
directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
xyzs = origins + depths[:, np.newaxis] * directions

xyzs = torch.tensor(xyzs, device=device)
directions = torch.tensor(directions, device=device)

# Find the associated voxel index for these termination points (so we know which ones to keep, in case they lie outside the grid for some reason)
voxel_index, in_bounds = vgrid.compute_voxel_index(xyzs)
voxel_index = voxel_index[in_bounds]
directions = directions[in_bounds]

# Create outer product from normalized directions. This forms the covariance matrix.
outer_product = torch.einsum("bi,bj->bij",directions, directions)

# Turn (i,j,k) tuple into a flattened index.
voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(vgrid.discretizations)), device=vgrid.device)

# Store the number of counts for each voxel.
voxel_counts = torch.zeros(tuple(vgrid.discretizations), device=vgrid.device, dtype=voxel_index_flatten.dtype).flatten()
voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))
voxel_counts = voxel_counts.reshape(tuple(vgrid.discretizations))

# Store the covariance matrix for each voxel. We add all covariances associated with a particular cell together.
voxel_covariance = torch.zeros(tuple(vgrid.discretizations) + (3, 3), device=vgrid.device, dtype=outer_product.dtype).reshape(-1, 3, 3)
voxel_covariance.scatter_add_(0, voxel_index_flatten.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3), outer_product)

# We take the mean covariance of each cell. 
voxel_covariance = voxel_covariance / (voxel_counts.reshape(-1).unsqueeze(1).unsqueeze(1) + 1e-6)

# Calculate the eigendecomposition of these covariances.
eigvals, eigvecs = torch.linalg.eigh(voxel_covariance)

# Reshape eigendecomposition into shape of the voxel grid
eigvals = eigvals.reshape(100, 100, 100, 3)
eigvecs = eigvecs.reshape(100, 100, 100, 3, 3)

# Calculate the eccentricity of the covariances by taking the ratio of the smallest to largest eigenvalues.
eccentricity = eigvals[:, 0] / (eigvals[:, 2] + 1e-6)

# Calculate the color of the voxel based on the eccentricity by passing it through a colormap. If you want to visualize
# this "static"  metric, just uncomment these two lines, and comment the following termination_fn function.
# voxel_vals = (eccentricity - eccentricity.min()) / (eccentricity.max() - eccentricity.min())
# vgrid.voxel_grid_values = voxel_vals.reshape(tuple(vgrid.discretizations))[..., None].expand(-1, -1, -1, 3)

# In order to render our view-dependent uncertainty metric, we need to rewrite our termination function, because
# although the occupancy grid may have stayed the same, the "values" of the grid have changed.
def termination_fn(next_points, directions, next_indices, voxel_grid_binary, voxel_grid_values):

    # Possible for indices to be out of bounds
    discretization = torch.tensor(voxel_grid_binary.shape, device=device)
    out_of_bounds = torch.any( torch.logical_or(next_indices < 0, next_indices - discretization.unsqueeze(0) > -1) , dim=-1)

    out_values = torch.zeros_like(next_points).to(torch.float32)

    # Mask out of bounds points
    next_points = next_points[~out_of_bounds]
    directions = directions[~out_of_bounds]
    next_indices = next_indices[~out_of_bounds]

    # Find which indices are occupied
    mask = voxel_grid_binary[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]

    # Find associated eigenvectors and eigenvalues
    vecs = eigvecs[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
    vals = eigvals[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]

    # calculates the Mahalanobis distance (sum_i (1 / lambda_i) * (v_i^T * d)^2), assuming d is normalized
    directions = directions.to(torch.float32)
    values = (torch.bmm(directions[..., None, :], vecs).squeeze() / torch.norm(directions, dim=-1, keepdim=True))**2
    values = values / (vals + 1e-8)
    values = torch.sum(values, dim=-1)

    # We then return the mask (which indices to terminate) and the values associated with those indices.
    out_mask = ~out_of_bounds
    out_mask[~out_of_bounds] = mask

    out_values[~out_of_bounds] = values[..., None].expand(-1, 3)

    return out_mask, out_values

# REMEMBER TO SET THE CLASS"S TERINATION FUNCTION #
vgrid.termination_fn = termination_fn
# %%
# We do the same thing as the training loop, but now that we've changed the termination function, we should see
# a different render.
for i, t_ in enumerate(t):

    c2w = torch.eye(4, device=device)
    c2w[:3, 3] = torch.tensor([0.25*torch.cos(3*t_), 0.25*torch.sin(3*t_), -0.25 + 0.5*i/len(t)], device=device)
    c2w[:3, :3] = look_at(c2w[:3, 3], torch.tensor([0., 0., 0.], device=device), torch.tensor([0., 0., 1.], device=device))

    tnow = time.time()
    torch.cuda.synchronize()
    image, depth, output = vgrid.camera_voxel_intersection(K, c2w, far_clip)
    depth = depth.cpu().numpy().reshape(1000, 1000, 1)
    depth_normalized = depth / depth.max()

    image = image / image.max()
    torch.cuda.synchronize()
    print("Time taken: ", time.time() - tnow)

    # combined image
    cmap = matplotlib.cm.get_cmap('turbo')
    combined_image = np.concatenate([image.cpu().numpy()[..., 0], depth_normalized.squeeze()], axis=1)
    combined_image = cmap(combined_image)[..., :3]
    imageio.imwrite(f'output/{i}.png', (combined_image * 255).astype(np.uint8))

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(image.cpu().numpy()[..., 0], cmap='turbo')
    # ax[1].imshow(depth_normalized, cmap='turbo')
# %%
