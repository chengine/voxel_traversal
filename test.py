#%%
from traversal_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

# ndim = 2
# param_dict = {
#     'discretizations': torch.tensor([10, 20], device=device),
#     'lower_bound': torch.tensor([-0.5, -0.5], device=device),
#     'upper_bound': torch.tensor([0.5, 0.5], device=device),
#     'voxel_grid_values': None
# }

ndim = 3
param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor([-0.5, -0.5, -0.5], device=device),
    'upper_bound': torch.tensor([0.5, 0.5, 0.5], device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'termination_fn': None
}

vgrid = VoxelGrid(param_dict, ndim, device)

nrays = 1000000

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
in_bound_voxel_index = output['in_bounds_voxel_index'].cpu().numpy()

out_bounds_points = output['out_bounds_points'].cpu().numpy()
out_bounds_directions = output['out_bounds_directions'].cpu().numpy()

out_bounds_intersect_points = output['out_bounds_intersect_points'].cpu().numpy()
out_bounds_intersect_directions = output['out_bounds_intersect_direction'].cpu().numpy()
out_bounds_intersect_voxel_index = output['out_bounds_intersect_voxel_index'].cpu().numpy()

not_intersecting_points = points[output['not_intersecting']].cpu().numpy()
not_intersecting_directions = directions[output['not_intersecting']].cpu().numpy()

#%%

#TODO: !!! THERE'S A HANGING ISSUE !!!#

tnow = time.time()
torch.cuda.synchronize()
vgrid_intersects = vgrid.compute_voxel_ray_intersection(points, directions)
torch.cuda.synchronize()
print("Time taken to traverse: ", time.time() - tnow)
#%%

# NOTE: THIS VISUALIZES THE TRACING WITHIN THE GRID

bottom_corners = vgrid.grid_vertices[:-1, :-1]
mask = torch.zeros(bottom_corners.shape[:-1], dtype=torch.bool)
voxel_intersections = vgrid_intersects["voxel_intersections"]
mask[voxel_intersections[:, 0], voxel_intersections[:, 1]] = True
# mask[vgrid_intersects[:, 0], vgrid_intersects[:, 1]] = True
# mask[in_bound_voxel_index[:, 0], in_bound_voxel_index[:, 1]] = True
# mask[out_bounds_intersect_voxel_index[:, 0], out_bounds_intersect_voxel_index[:, 1]] = True

bottom_corners = bottom_corners.reshape(-1, ndim).cpu().numpy()
bottom_corners_mask = mask.reshape(-1).cpu().numpy()
cell_size = vgrid.cell_sizes.cpu().numpy()

print("plotting")
#%%
# Create figure and axes
fig, ax = plt.subplots(dpi=500)
for mask, corner in zip(bottom_corners_mask, bottom_corners):
    if mask:
        facecolor = 'red'
        alpha = 0.4
    else:
        facecolor = 'none'
        alpha = 1.0
        
    # Create a Rectangle patch
    rect = patches.Rectangle((corner[0], corner[1]), cell_size[0], cell_size[1], linewidth=1, edgecolor='r', facecolor=facecolor, alpha=alpha)
    # Add the patch to the Axes
    ax.add_patch(rect)

# NOTE: THIS TESTS IF THE RAYS SHOULD BE CONSIDERED IN THE RAY TRACING ALGORITHM
# Create a Rectangle patch
# rect = patches.Rectangle((-0.5, -0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')

# # Add the patch to the Axes
# ax.add_patch(rect)

for i in range(in_bounds_points.shape[0]):
    ax.arrow(in_bounds_points[i, 0], in_bounds_points[i, 1], in_bounds_directions[i, 0], in_bounds_directions[i, 1], head_width=0.005, head_length=0.01, fc='g', ec='g')

for i in range(out_bounds_points.shape[0]):
    ax.arrow(out_bounds_points[i, 0], out_bounds_points[i, 1], out_bounds_directions[i, 0], out_bounds_directions[i, 1], head_width=0.005, head_length=0.01, fc='orange', ec='orange', linestyle='dashed', alpha=0.4)

for i in range(out_bounds_intersect_points.shape[0]):
    ax.arrow(out_bounds_intersect_points[i, 0], out_bounds_intersect_points[i, 1], out_bounds_intersect_directions[i, 0], out_bounds_intersect_directions[i, 1], head_width=0.005, head_length=0.01, fc='blue', ec='blue')

for i in range(not_intersecting_points.shape[0]):
    ax.arrow(not_intersecting_points[i, 0], not_intersecting_points[i, 1], not_intersecting_directions[i, 0], not_intersecting_directions[i, 1], head_width=0., head_length=0., fc='red', ec='red')

ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)
# plt.show()
plt.savefig("./plots/plot.png")

#%%