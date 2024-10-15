import torch

# All these functions are minor modifications to each other to make them fast for their use case
# They all hinge on the core function: one-step voxel ray intersection

# For generalizability, we will assume the voxel grid parameters are a dictionary as follows

# voxel_grid_params = {
#     'discretizations': Tensor[3],  number of voxels in each dimension
#     'lower_bound': Tensor[3],     upper most corner
#     'upper_bound': Tensor[3],     lower most corner
# }

def initialize_voxel_index(origins, )


# This function takes in the current point and voxel index, and returns the next point and voxel index
def one_step_voxel_ray_intersection():

# Computes intersection of ray with a voxel grid and returns a list of tensors of the voxel indices with their ray indices.
# def compute_voxel_ray_intersection

# Goes one step further and only return list of tensors of voxel indices where the voxel values are non-zero. 
# May be able to just pass an argument into the previous function to do this.
# def compute_voxel_ray_intersection_nonzero

# Renders the voxel grid, given the voxel grid, camera extrinsics, and intrinsics.
# Makes use of the nerfacc library for fast rendering.
# We can play with some of the approximations, or make it exact if need be.
def render_from_voxel_grid(voxel_grid, origins, directions, max_steps):
    # Initialize the voxel index
    voxel_index = initialize_voxel_index(origins, directions)
    # Initialize the output tensor
    output = torch.zeros_like(origins)
    # Loop over the max_steps
    for i in range(max_steps):
        # Compute the next voxel index
        voxel_index = one_step_voxel_ray_intersection(voxel_index)
        # Compute the voxel value at the voxel index
        voxel_value = voxel_grid[voxel_index]
        # Update the output tensor
        output = output + voxel_value
    return output