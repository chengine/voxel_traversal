#%%
import numpy as np
import torch
import open3d as o3d
import time
import matplotlib
import imageio
from traversal_utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

obj_path = 'stanford-bunny.obj'

mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
points = np.asarray(mesh.vertices)

bounding_box = mesh.get_axis_aligned_bounding_box()

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

vgrid = VoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

scene = vgrid.create_mesh_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

o3d.visualization.draw_geometries([scene])

#%%
import matplotlib.pyplot as plt
def look_at(location, target, up):
    z = (location - target)
    z /= torch.norm(z)
    x = torch.cross(up, z)
    x /= torch.norm(x)
    y = torch.cross(z, x)
    y /= torch.norm(y)

    R = torch.stack([x, y, z], dim=1)
    return R

K = torch.tensor([
    [1000., 0., 500.],
    [0., 1000., 500.],
    [0., 0., 1.]
], device=device)

t = torch.linspace(0., 2*np.pi, 100, device=device)
far_clip = 1.

for i, t_ in enumerate(t):
    c2w = torch.eye(4, device=device)
    c2w[:3, 3] = torch.tensor([0.25*torch.cos(t_), 0.25*torch.sin(t_), 0.25], device=device)

    c2w[:3, :3] = look_at(c2w[:3, 3], torch.tensor([0., 0., 0.], device=device), torch.tensor([0., 0., 1.], device=device))

    tnow = time.time()
    torch.cuda.synchronize()
    image, depth = vgrid.camera_voxel_intersection(K, c2w, far_clip)
    depth = depth.cpu().numpy().reshape(1000, 1000, 1)
    depth = depth / depth.max()
    torch.cuda.synchronize()
    print("Time taken: ", time.time() - tnow)

    # combined image
    combined_image = np.concatenate([image.cpu().numpy(), depth.repeat(3, axis=-1)], axis=1)

    imageio.imwrite(f'output/{i}.png', (combined_image * 255).astype(np.uint8))

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(image.cpu().numpy())
    # ax[1].imshow(depth, cmap='gray')
# %%
