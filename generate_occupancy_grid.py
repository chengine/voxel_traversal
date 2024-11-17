import numpy as np
import torch
import open3d as o3d
import time
import matplotlib
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch.optim as optim

from traversal_utils import *
from covariance_utils import angle_axis_to_rotation_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

obj_path = 'stanford-bunny.obj'

mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, 0)), center=np.zeros(3))
mesh.translate(-mesh.get_center())
points = np.asarray(mesh.vertices)

bounding_box = mesh.get_axis_aligned_bounding_box()
cmap = matplotlib.cm.get_cmap('turbo')

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

print("lower bound: ", bounding_box.get_min_bound())
print("upper bound: ", bounding_box.get_max_bound())

param_dict = {
    'discretizations': torch.tensor([50, 50, 50], device=device),
    'lower_bound': torch.tensor([-0.2, -0.2, -0.2], device=device),
    'upper_bound': torch.tensor([0.2, 0.2, 0.2], device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'termination_fn': termination_fn
}

colors = cmap( (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]

vgrid = VoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

print("voxel grid centers: ", vgrid.voxel_grid_centers.shape)
print("voxel grid binary: ", vgrid.voxel_grid_binary.shape)
print("voxel grid centers binary: ", vgrid.voxel_grid_centers[vgrid.voxel_grid_binary].shape)
print("points: ", points.shape )

occupied_cells = vgrid.voxel_grid_centers[vgrid.voxel_grid_binary].cpu().numpy()

ax1 = plt.figure().add_subplot(projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2])
plt.savefig("./plots/image_points.png")

ax2 = plt.figure().add_subplot(projection='3d')
ax2.scatter(occupied_cells[:,0], occupied_cells[:,1], occupied_cells[:,2])
plt.savefig("./plots/image_voxels.png")

import dijkstra3d
source = (10,30,0)
voxel_grid_3d = vgrid.voxel_grid_binary.cpu().numpy()
voxel_grid_2d = voxel_grid_3d[:,:,27,None] # 1 where occupied, 0 where unoccupied

dist_field = dijkstra3d.euclidean_distance_field(np.invert(voxel_grid_2d), source=source)
print("dist_field: ", dist_field[25,:,0])
parents = dijkstra3d.parental_field(dist_field, source=source)

ax3 = plt.figure().add_subplot(projection='3d')
ax3.voxels(voxel_grid_2d,alpha=1)

voxel_inds = np.indices(voxel_grid_2d.shape)
voxel_inds = np.vstack((voxel_inds[0].flatten(),voxel_inds[1].flatten(),voxel_inds[2].flatten()))
print("num trajectories: ", voxel_inds.shape[1])
for i in range(0,(voxel_inds.shape[1]),9):
    target = tuple(voxel_inds[:,i])
    if voxel_grid_2d[target[0], target[1], target[2]] == 1:
        print("occupied target!")
        continue
    path = dijkstra3d.path_from_parents(parents, target=tuple(target))
    cost = dist_field[target[0], target[1], target[2]]
    if cost == np.inf:
        print("infinite cost!")
        continue
    print("path: ", path.shape, " target: ", target, " cost: ", cost)
    # compute the information gain metric for the views here
    ax3.plot(path[:,0], path[:,1], path[:,2],c='g')
    ax3.scatter(target[0], target[1], target[2],c='r')
ax3.scatter(source[0], source[1], source[2], c='c')
plt.savefig("./plots/image_dijkstra.png")
ax3.view_init(azim=0, elev=90)
plt.savefig("./plots/image_dijkstra_top.png")

