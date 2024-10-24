import torch

# All these functions are minor modifications to each other to make them fast for their use case
# They all hinge on the core function: one-step voxel ray intersection

# For generalizability, we will assume the voxel grid parameters are a dictionary as follows

# voxel_grid_params = {
#     'discretizations': Tensor[3],  number of voxels in each dimension
#     'lower_bound': Tensor[3],     upper most corner
#     'upper_bound': Tensor[3],     lower most corner
#     'voxel_grid_values': Tensor[discretizations[0], discretizations[1], discretizations[2]]  3D tensor of voxel values # TODO Can be 4D tensor potentially
# }

class VoxelGrid:
    def __init__(self, param_dict, ndim, device):
        self.discretizations = param_dict['discretizations']
        self.ndim = ndim

        if ndim is None:
            self.ndim = len(self.discretizations)

        assert self.ndim == 2 or self.ndim == 3, "Only 2D and 3D voxel grids are supported"
        assert len(self.discretizations) == self.ndim, "Discretizations must be of length equal to the number of dimensions"

        ### IMPORTANT !!! ###
        # Lower bound must be strictly less than upper bound #
        self.lower_bound = param_dict['lower_bound']
        self.upper_bound = param_dict['upper_bound']

        assert torch.all(self.lower_bound - self.upper_bound <= 0.), "Lower bound must be less than upper bound"
        assert len(self.lower_bound) == self.ndim, "Lower bound must be of length equal to the number of dimensions"
        assert len(self.upper_bound) == self.ndim, "Upper bound must be of length equal to the number of dimensions"

        ### IMPORTANT !!! ###

        self.voxel_grid_values = param_dict['voxel_grid_values']
        self.device = device

        self.initialize_voxel_grid()

    #TODO: May want to take in some pre-made voxel grid values
    def initialize_voxel_grid(self):
        if self.voxel_grid_values is None:
            self.voxel_grid_values = torch.zeros(tuple(self.discretizations))
        
        self.cell_sizes = (self.upper_bound - self.lower_bound) / self.discretizations

        grid_pts = torch.meshgrid([torch.linspace(self.lower_bound[i], self.upper_bound[i], self.discretizations[i]+1, device=self.device) for i in range(self.ndim)])
        self.grid_vertices = torch.stack(grid_pts, dim=-1)      # n1 x n2 x n3 x 3

        if self.ndim == 2:
            self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1] + self.grid_vertices[1:, 1:])
            self.lower_grid_center = self.voxel_grid_centers[0, 0]

        elif self.ndim == 3:
            self.voxel_grid_centers = 0.5 * (self.grid_vertices[:-1, :-1, :-1] + self.grid_vertices[1:, 1:, 1:])
            self.lower_grid_center = self.voxel_grid_centers[0, 0, 0]
        

    def compute_voxel_index(self, points, debug=False):
        # IMPORTANT !!! #
        # This function does NOT handle points outside the voxel grid. It is the user's responsibility to ensure that the points are within the voxel grid, because
        # there are many ways in which one could project outside points into the grid. 

        ### IMPORTANT !!! ###
        # We are going with the "round" convention for indexing.
        voxel_index = torch.round( (points - self.lower_grid_center) / self.cell_sizes )

        # Sanity check
        is_in_bounds = []
        for i in range(self.ndim):
            if debug:
                assert torch.all(voxel_index[:, i] >= 0) and torch.all(voxel_index[:, i] < self.discretizations[i]), "Points must be within the voxel grid"
     
            is_in_bounds.append( torch.logical_and( (voxel_index[:, i] >= 0), (voxel_index[:, i] < self.discretizations[i])) )

        is_in_bounds = torch.all( torch.stack(is_in_bounds, dim=-1), dim = -1 )

        return voxel_index, is_in_bounds
    
    def project_points_into_voxel_grid(self, points, directions):
        # Reference: https://tavianator.com/2022/ray_box_boundary.html#:~:text=Geometry&text=The%20slab%20method%20tests%20for,intersects%20the%20box%2C%20if%20any.&text=t%20%3D%20x%20%E2%88%92%20x%200%20x%20d%20.
        ### IMPORTANT !!! ###
        # The directions you pass DO NOT need to be normalized. They will be treated as line segments!

        # If the point is in the voxel grid, we don't do anything.
        # However, if the point is outside the voxel grid, we want to use 
        # directions such that the voxel we return is the first voxel that the ray intersects.
        voxel_index, in_bounds = self.compute_voxel_index(points)

        # indices are within the voxel grid
        in_bound_voxel_index = voxel_index[in_bounds]

        # indices are outside the voxel grid
        out_bound_directions = directions[~in_bounds] + torch.randn_like(directions[~in_bounds]) * 1e-6       # Add some noise to reduce degeneracy
        out_bound_points = points[~in_bounds]

        # Of these out of bound points, there are rays that ultimately intersect the voxel, and others that miss the voxel grid entirely.
        # The general test:

        t1 = (self.lower_bound[None] - out_bound_points) / out_bound_directions     # N x 3
        t2 = (self.upper_bound[None] - out_bound_points) / out_bound_directions     # N x 3

        t = torch.stack([t1, t2], dim=-1)       # N x 3 x 2

        # TODO: We can skip this first min/max step by looking at the signs of the directions directly!
        tmin = torch.min(t, dim=-1).values      # N x 3
        tmax = torch.max(t, dim=-1).values      # N x 3

        tmin = torch.max(tmin, dim=-1).values     # N
        tmax = torch.min(tmax, dim=-1).values     # N

        # If tmin > tmax for any dimension, the line misses the voxel grid
        misses = (tmin - tmax) > 0.     # N

        # TODO: CHECK THIS PROCEDURE!!!

        # Of the rays that we think intersect the voxel grid, we need to make sure that the line segment actually intersects the voxel grid (t < 1)
        # AND we need to make sure the t's are also positive!!!
        short = tmin > 1.     # N
        reversed = tmin < 0.  # N

        # # These points will NEVER intersect the voxel grid
        misses = torch.logical_or(misses, short)
        misses = torch.logical_or(misses, reversed)

        # These are the points that do intersect the voxel grid but are outside the voxel grid
        out_bound_intersect_dirs = out_bound_directions[~misses]
        out_bound_intersect_pts = out_bound_points[~misses]

        # and the progress along the line in order to hit the voxel grid
        t_progress = tmin[~misses]
        remaining_progress = 1. - t_progress

        # and the point where the ray intersects the voxel grid
        intersect_pts = out_bound_intersect_pts + out_bound_intersect_dirs * t_progress[:, None]
        intersect_directions = out_bound_intersect_dirs * remaining_progress[:, None]
        intersect_voxel_index, _ = self.compute_voxel_index(intersect_pts)

        # This mask never intersects
        not_intersecting = torch.zeros_like(in_bounds, dtype=torch.bool)
        not_intersecting[~in_bounds] = misses

        output = {
            'in_bounds': in_bounds,
            'not_intersecting': not_intersecting,       # mask of not intersecting data points
            'intersecting_out_bounds': torch.logical_and(~not_intersecting, ~in_bounds), # mask of intersecting data points  (not clearly in bounds but intersecting gives you the points outside the grid that intersect the grid)
            'in_bounds_voxel_index': in_bound_voxel_index,
            'in_bounds_points': points[in_bounds],               # directions remain unchanged since we are already in the voxel grid
            'in_bounds_directions': directions[in_bounds],
            'out_bounds_points': out_bound_intersect_pts,           
            'out_bounds_directions': out_bound_intersect_dirs,
            'out_bounds_intersect_points': intersect_pts,           # project points outside of voxel grid that do intersect the grid into the edge of the grid
            'out_bounds_intersect_direction': intersect_directions, # remaining vector after intersecting the grid
            'out_bounds_intersect_voxel_index': intersect_voxel_index,  # voxel index of the intersecting point with the voxel grid
        }

        # Returns the voxel index, the point, the truncated direction for points in the grid, and the indices of points that do intersect the grid
        # Also returns the indices of points that don't ever intersect the voxel
        return output

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