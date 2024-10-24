#%%
from traversal_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(0)

ndim = 2
param_dict = {
    'discretizations': torch.tensor([10, 20], device=device),
    'lower_bound': torch.tensor([-0.5, -0.5], device=device),
    'upper_bound': torch.tensor([0.5, 0.5], device=device),
    'voxel_grid_values': None
}

# ndim = 3
# param_dict = {
#     'discretizations': torch.tensor([100, 50, 50], device=device),
#     'lower_bound': torch.tensor([-0.5, -0.5, -0.5], device=device),
#     'upper_bound': torch.tensor([0.5, 0.5, 0.5], device=device),
#     'voxel_grid_values': None
# }

vgrid = VoxelGrid(param_dict, ndim, device)


nrays = 50

points = 2*torch.rand(nrays, ndim, device=device)- 1.0
directions = torch.randn(nrays, ndim, device=device)

tnow = time.time()
torch.cuda.synchronize()
output = vgrid.project_points_into_voxel_grid(points, directions)
torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)

# Three classes of rays: Clearly in the voxel grid, outside of the voxel grid and don't intersect, and outside of the voxel grid but intersect.
# output = {
#     'in_bounds': in_bounds,
#     'not_intersecting': not_intersecting,       # mask of not intersecting data points
#     'intersecting_out_bounds': torch.logical_and(~not_intersecting, ~in_bounds), # mask of intersecting data points  (not clearly in bounds but intersecting gives you the points outside the grid that intersect the grid)
#     'in_bounds_voxel_index': in_bound_voxel_index,
#     'in_bounds_points': points[in_bounds],               # directions remain unchanged since we are already in the voxel grid
#     'in_bounds_directions': directions[in_bounds],
#     'out_bounds_points': out_bound_intersect_pts,           
#     'out_bounds_directions': out_bound_intersect_dirs,
#     'out_bounds_intersect_points': intersect_pts,           # project points outside of voxel grid that do intersect the grid into the edge of the grid
#     'out_bounds_intersect_direction': intersect_directions, # remaining vector after intersecting the grid
#     'out_bounds_intersect_voxel_index': intersect_voxel_index,  # voxel index of the intersecting point with the voxel grid
# }

in_bounds_points = output['in_bounds_points'].cpu().numpy()
in_bounds_directions = output['in_bounds_directions'].cpu().numpy()

out_bounds_points = output['out_bounds_points'].cpu().numpy()
out_bounds_directions = output['out_bounds_directions'].cpu().numpy()

out_bounds_intersect_points = output['out_bounds_intersect_points'].cpu().numpy()
out_bounds_intersect_directions = output['out_bounds_intersect_direction'].cpu().numpy()

not_intersecting_points = points[output['not_intersecting']].cpu().numpy()
not_intersecting_directions = directions[output['not_intersecting']].cpu().numpy()

#%%

# NOTE: THIS TESTS IF THE RAYS SHOULD BE CONSIDERED IN THE RAY TRACING ALGORITHM

# Create figure and axes
fig, ax = plt.subplots()

# Create a Rectangle patch
rect = patches.Rectangle((-0.5, -0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

for i in range(in_bounds_points.shape[0]):
    ax.arrow(in_bounds_points[i, 0], in_bounds_points[i, 1], in_bounds_directions[i, 0], in_bounds_directions[i, 1], head_width=0.05, head_length=0.1, fc='g', ec='g')

for i in range(out_bounds_points.shape[0]):
    ax.arrow(out_bounds_points[i, 0], out_bounds_points[i, 1], out_bounds_directions[i, 0], out_bounds_directions[i, 1], head_width=0.05, head_length=0.1, fc='orange', ec='orange', linestyle='dashed', alpha=0.4)

for i in range(out_bounds_intersect_points.shape[0]):
    ax.arrow(out_bounds_intersect_points[i, 0], out_bounds_intersect_points[i, 1], out_bounds_intersect_directions[i, 0], out_bounds_intersect_directions[i, 1], head_width=0.05, head_length=0.1, fc='blue', ec='blue')

for i in range(not_intersecting_points.shape[0]):
    ax.arrow(not_intersecting_points[i, 0], not_intersecting_points[i, 1], not_intersecting_directions[i, 0], not_intersecting_directions[i, 1], head_width=0., head_length=0., fc='red', ec='red')

plt.show()

#%%

# NOTE: THIS VISUALIZES THE TRACING WITHIN THE GRID
# Create figure and axes
fig, ax = plt.subplots()

bottom_corners = vgrid.grid_vertices[:-1, :-1].reshape(-1, ndim).cpu().numpy()
cell_size = vgrid.cell_sizes.cpu().numpy()

for corner in bottom_corners:
    # Create a Rectangle patch
    rect = patches.Rectangle((corner[0], corner[1]), cell_size[0], cell_size[1], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)
plt.show()

#%%