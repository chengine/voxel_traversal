#%%
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
mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=np.zeros(3))
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

param_dict = {
    'discretizations': torch.tensor([100, 100, 100], device=device),
    'lower_bound': torch.tensor(bounding_box.get_min_bound(), device=device),
    'upper_bound': torch.tensor(bounding_box.get_max_bound(), device=device),
    'voxel_grid_values': None,
    'voxel_grid_binary': None,
    'termination_fn': termination_fn,
    'termination_value_type': "colors"
}

colors = cmap( (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()))[:, :3]
# colors = np.asarray(mesh.vertex_colors)

vgrid = VoxelGrid(param_dict, 3, device)
vgrid.populate_voxel_grid_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

# scene = vgrid.create_mesh_from_points(torch.tensor(points, device=device), torch.tensor(colors, device=device, dtype=torch.float32))

# o3d.visualization.draw_geometries([scene])

#%%
def look_at(location, target, up):
    z = (location - target)
    z_ = z / torch.norm(z)
    x = torch.cross(up, z)
    x_ = x / torch.norm(x)
    y = torch.cross(z, x)
    y_ = y / torch.norm(y)

    R = torch.stack([x_, y_, z_], dim=-1)
    return R

K = torch.tensor([
    [100., 0., 100.],
    [0., 100., 100.],
    [0., 0., 1.]
], device=device)

N_cameras = 1000
t = torch.linspace(0., 2*np.pi, N_cameras, device=device)
far_clip = 1.

depth_images = []
directions = []
origins = []

c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
c2w[:, :3, 3] = torch.stack([0.25*torch.cos(t), 0.25*torch.sin(t), torch.zeros_like(t)], dim=-1)

target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)
c2w[:, :3, :3] = look_at(c2w[:, :3, 3], target, up)

# combined image
cmap = matplotlib.cm.get_cmap('turbo')

tnow = time.time()
torch.cuda.synchronize()
image, depth, output = vgrid.camera_voxel_intersection(K, c2w, far_clip)
torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)

input("press enter to continue")
#%%

### SEGMENT OUT THE RAYS THAT INTERSECTED THE VOXEL GRID ###
depths = depth.reshape(-1)
directions = output['rays_d'].reshape(-1, 3)
origins = output['rays_o'].reshape(-1, 3)

# For depths greater than 0
mask = depths > 0
depths = depths[mask]
directions = directions[mask]
origins = origins[mask]

# Termination points
directions = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)
xyzs = origins + depths[:, None] * directions

voxel_index, in_bounds = vgrid.compute_voxel_index(xyzs)
voxel_index = voxel_index[in_bounds]
directions = directions[in_bounds]

# Create outer product from normalized directions
outer_product = torch.einsum("bi,bj->bij",directions, directions)

voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(vgrid.discretizations)), device=vgrid.device)

# Store the number of counts for each voxel
voxel_counts = torch.zeros(tuple(vgrid.discretizations), device=vgrid.device, dtype=voxel_index_flatten.dtype).flatten()
voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))
voxel_counts = voxel_counts.reshape(tuple(vgrid.discretizations))

# Store the covariance matrix for each voxel
voxel_covariance = torch.zeros(tuple(vgrid.discretizations) + (3, 3), device=vgrid.device, dtype=outer_product.dtype).reshape(-1, 3, 3)
voxel_covariance.scatter_add_(0, voxel_index_flatten.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3), outer_product)

voxel_covariance = voxel_covariance / (voxel_counts.reshape(-1).unsqueeze(1).unsqueeze(1) + 1e-6)

eigvals, eigvecs = torch.linalg.eigh(voxel_covariance)

eigvals = eigvals.reshape(100, 100, 100, 3)
eigvecs = eigvecs.reshape(100, 100, 100, 3, 3)

# Calculate the eccentricity of the covariances by taking the ratio of the largest to smallest eigenvalues
eccentricity = eigvals[:, 0] / (eigvals[:, 2] + 1e-6)

# Calculate the color of the voxel based on the eccentricity by passing it through a colormap
voxel_vals = (eccentricity - eccentricity.min()) / (eccentricity.max() - eccentricity.min())

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

    vecs = eigvecs[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
    vals = torch.nn.functional.relu(eigvals[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]])

    directions = directions.to(torch.float32)
    values = (torch.bmm(directions[..., None, :], vecs).squeeze() / torch.norm(directions, dim=-1, keepdim=True))**2
    values = values / (vals + 1e-4)
    values = torch.sum(values, dim=-1)

    out_mask = ~out_of_bounds
    out_mask[~out_of_bounds] = mask

    out_values[~out_of_bounds] = values[..., None].expand(-1, 3)

    return out_mask, out_values

vgrid.termination_fn = termination_fn
# %%

c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
c2w[:, :3, 3] = torch.stack([0.25*torch.cos(30*t), 0.25*torch.sin(30*t), -0.25 + 0.5*torch.arange(len(t), device=device)/len(t)], dim=-1)

target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)
c2w[:, :3, :3] = look_at(c2w[:, :3, 3], target, up)

tnow = time.time()
torch.cuda.synchronize()

c2ws = c2w

c2w_list = torch.split(c2ws, 300)
images = []
depths = []
for c2w in c2w_list:
    image, depth, _ = vgrid.camera_voxel_intersection(K, c2w, far_clip)
    images.append(image)
    depths.append(depth)

images = torch.cat(images, dim=0)
depths = torch.cat(depths, dim=0)
torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)


# combined image
cmap = matplotlib.cm.get_cmap('turbo')
# for i, (img, dep) in enumerate(zip(images, depths)):
#     dep = dep / far_clip
#     img = img / img.max()
#     combined_image = torch.concatenate([img[..., 0], dep], axis=1)

#     mask = (combined_image == 0).cpu().numpy()
#     combined_image = cmap(combined_image.cpu().numpy())[..., :3]
#     combined_image[mask] = 0

#     imageio.imwrite(f'output/{i}.png', (combined_image * 255).astype(np.uint8))

#%%
images = images / (torch.amax(images, dim=(1, 2))[:, None, None] + 1e-6)

uncertainty_metric = torch.sum(images, dim=(1, 2))[..., 0]
# Choose the image with the highest uncertainty

#%%
import cv2

# Sort and visualize the images in descending uncertainty
sorted, indices = torch.sort(uncertainty_metric, descending=True)
sorted_images = images[indices]
sorted_depths = depths[indices]

for i, (img, dep) in enumerate(zip(sorted_images, sorted_depths)):
    dep = dep / far_clip
    img = img / (img.max() + 1e-6)
    combined_image = torch.concatenate([img[..., 0], dep], axis=1)

    mask = (combined_image == 0).cpu().numpy()
    combined_image = cmap(combined_image.cpu().numpy())[..., :3]
    combined_image[mask] = 0

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (25, 25)

    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2
    
    combined_image = (combined_image * 255).astype(np.uint8)
    # Using cv2.putText() method
    combined_image = cv2.putText(combined_image, f'Image {i}: {sorted[i]:.2f}', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    imageio.imwrite(f'view_select/{i}.png', combined_image)

#%%
# Perform gradient descent
def spherical_to_cartesian(spherical_input):
    phi, theta = spherical_input[..., 0], spherical_input[..., 1]
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)

r = 0.25

# target = torch.zeros(3, device=device)
# up = torch.tensor([0., 0., 1.], device=device)

# init_trans = torch.tensor([np.pi/1.9, np.pi/2], device=device, dtype=torch.float32, requires_grad=True)

# # init_rot = look_at(r*spherical_to_cartesian(init_trans), target, up)
# # init_rot = torch.tensor(R.from_matrix(init_rot.detach().cpu().numpy()).as_rotvec(), dtype=torch.float32, device=device, requires_grad=True)

# # print(init_rot, r*spherical_to_cartesian(init_trans))

# # Create optimizer
# optimizer = optim.Adam([init_trans], lr=1e-2)

# combined_images = []
# losses = []
# tnow = time.time()
# torch.cuda.synchronize()
# try:
#     for i in range(100):
#         optimizer.zero_grad()

#         translation = r*spherical_to_cartesian(init_trans)
#         c2w = torch.eye(4, device=device)
#         c2w[:3, 3] = translation
#         c2w[:3, :3] = look_at(translation, target, up)

#         image, depth, _ = vgrid.camera_voxel_intersection(K, c2w[None], far_clip)

#         normalized_image = image.squeeze() / (torch.amax(image) + 1e-6)

#         uncertainty_metric = torch.sum(normalized_image)

#         loss = -uncertainty_metric

#         if torch.isnan(loss):
#             raise Exception("Loss is NaN")
        
#         losses.append(loss.item())
#         combined_images.append((normalized_image.detach(), depth.squeeze().detach()))

#         loss.backward()
#         optimizer.step()

#         if i % 10 == 0:
#             print(f"Loss: {loss.item()}")

# except Exception as e:
#     print(e)

# torch.cuda.synchronize()
# print("Time taken: ", time.time() - tnow)

# #%%
# for i, (img, dep) in enumerate(combined_images):
#     dep = dep / far_clip
#     img = img / (img.max() + 1e-6)
#     combined_image = torch.concatenate([img[..., 0], dep], axis=1)

#     mask = (combined_image == 0).cpu().numpy()
#     combined_image = cmap(combined_image.cpu().numpy())[..., :3]
#     combined_image[mask] = 0

#     # font
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # org
#     org = (25, 25)

#     # fontScale
#     fontScale = 1
    
#     # Blue color in BGR
#     color = (255, 255, 255)

#     # Line thickness of 2 px
#     thickness = 2
    
#     combined_image = (combined_image * 255).astype(np.uint8)
#     # Using cv2.putText() method
#     combined_image = cv2.putText(combined_image, f'Image {i}: {-losses[i]:.2f}', org, font, 
#                     fontScale, color, thickness, cv2.LINE_AA)

#     imageio.imwrite(f'view_select_opt/{i}.png', combined_image)
# %%
# Batch gradient descent
N_cameras = 25
target = torch.zeros(3, device=device)[None].expand(N_cameras, -1)
up = torch.tensor([0., 0., 1.], device=device)[None].expand(N_cameras, -1)

init_trans = torch.randn(N_cameras, 2, device=device, dtype=torch.float32, requires_grad=True)

# init_rot = look_at(r*spherical_to_cartesian(init_trans), target, up)
# init_rot = torch.tensor(R.from_matrix(init_rot.detach().cpu().numpy()).as_rotvec(), dtype=torch.float32, device=device, requires_grad=True)

# print(init_rot, r*spherical_to_cartesian(init_trans))

# Create optimizer
optimizer = optim.Adam([init_trans], lr=1e-2)

combined_images = []
losses = []
tnow = time.time()
torch.cuda.synchronize()
for i in range(100):
    optimizer.zero_grad()

    translation = r*spherical_to_cartesian(init_trans)
    c2w = torch.eye(4, device=device)[None].expand(N_cameras, -1, -1).clone()
    c2w[:, :3, 3] = translation
    c2w[:, :3, :3] = look_at(translation, target, up)

    image, depth, _ = vgrid.camera_voxel_intersection(K, c2w, far_clip)

    normalized_image = image.squeeze() / (torch.amax(image, dim=(1, 2))[:, None, None] + 1e-6)

    uncertainty_metric = torch.sum(normalized_image)

    loss = -uncertainty_metric

    if torch.isnan(loss):
        raise Exception("Loss is NaN")
    
    losses.append(loss.item())
    combined_images.append((normalized_image.detach(), depth.squeeze().detach()))

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Loss: {loss.item()}")

torch.cuda.synchronize()
print("Time taken: ", time.time() - tnow)
# %%
