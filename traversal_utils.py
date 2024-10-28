import torch
import time

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
        voxel_index = torch.round( (points - self.lower_grid_center) / self.cell_sizes ).to(torch.int32)

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
        intersect_pts = out_bound_intersect_pts + out_bound_intersect_dirs * (t_progress[:, None])     # Add some noise to avoid numerical instability
        intersect_directions = out_bound_intersect_dirs * remaining_progress[:, None]
        intersect_voxel_index, _ = self.compute_voxel_index(intersect_pts)

        in_bound_voxel_index = torch.clamp(in_bound_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)
        intersect_voxel_index = torch.clamp(intersect_voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)

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
    # NOTE: !!! IT MAY BE FASTER TO DO ALL THIS IN VOXEL COORDINATES RATHER THAN IN XYZ COORDINATES !!!
    def one_step_voxel_ray_intersection(self, points, indices, directions, sign_directions, step_xyz):
        # NOTE: !!! IMPORTANT !!! directions is the vector from the termination point to the point defined by points

        B = len(points)
        arange = torch.arange(B, device=self.device)

        # tnow = time.time()
        # torch.cuda.synchronize()

        # At each point, find the progress t to the next voxel boundary
        if self.ndim == 2:
            xyz_max = self.voxel_grid_centers[indices[:, 0], indices[:, 1]] + step_xyz/2
        else:
            xyz_max = self.voxel_grid_centers[indices[:, 0], indices[:, 1], indices[:, 2]] + step_xyz/2

        # torch.cuda.synchronize()
        # print("Time taken for getting xyz_max: ", time.time() - tnow)

        # tnow = time.time()
        # torch.cuda.synchronize()

        t = (xyz_max - points) / directions

        min_t, min_idx = torch.min(t, dim=-1)       # N, idx tells us which voxel index dimension to move in (+- 1)

        # torch.cuda.synchronize()
        # print("Time taken for getting min: ", time.time() - tnow)

        # tnow = time.time()
        # torch.cuda.synchronize()

        # Update next voxel intersection point and voxel index
        points = points + min_t.unsqueeze(-1) * directions
        new_indices = indices.clone()
        new_indices[arange, min_idx] += sign_directions[arange, min_idx]

        # torch.cuda.synchronize()
        # print("Time taken for updating points: ", time.time() - tnow)

        return points, new_indices, min_t
    
    # Computes intersection of ray with a voxel grid and returns a list of tensors of the voxel indices with their ray indices.
    def compute_voxel_ray_intersection(self, points, directions, termination_fn=None):
        # We assume directions is such that points + t * directions, where t = 1, is the termination point

        # tnow = time.time()
        # torch.cuda.synchronize()

        # Project the points into the voxel grid
        output = self.project_points_into_voxel_grid(points, directions)

        # Segment out the rays that intersect the voxel grid
        in_bounds_points = output['in_bounds_points']
        in_bounds_directions = output['in_bounds_directions']
        in_bounds_voxel_index = output['in_bounds_voxel_index']

        # out_bounds_points = output['out_bounds_points']
        # out_bounds_directions = output['out_bounds_directions']

        # These are the Out-of-bounds rays, truncated so that they intersect the voxel grid
        out_bounds_intersect_points = output['out_bounds_intersect_points']
        out_bounds_intersect_directions = output['out_bounds_intersect_direction']
        out_bounds_intersect_voxel_index = output['out_bounds_intersect_voxel_index']

        # Concatenate the in-bounds and out-bounds intersecting rays
        points = torch.cat([in_bounds_points, out_bounds_intersect_points], dim=0)
        directions = torch.cat([in_bounds_directions, out_bounds_intersect_directions], dim=0)
        voxel_index = torch.cat([in_bounds_voxel_index, out_bounds_intersect_voxel_index], dim=0)

        # NOTE: It shouldn't be possible to get -1 or self.discretizations[None] in the voxel index
        voxel_index = torch.clamp(voxel_index, torch.zeros_like(self.discretizations).unsqueeze(0), self.discretizations.unsqueeze(0) - 1)

        # We completely ignore these rays
        # TODO: Might want to store these ray indices for later use
        # not_intersecting_points = points[output['not_intersecting']]
        # not_intersecting_directions = directions[output['not_intersecting']]

        #Store the voxel indices that have been passed through
        #NOTE: Might want to store the (voxel, ray) indices for later use
        voxel_intersections = [voxel_index]

        # torch.cuda.synchronize()
        # print("Time taken for initialization: ", time.time() - tnow)

        # NOTE: This could be computed just once outside of this function!
        sign_directions = torch.sign(directions + 1e-6*torch.randn_like(directions)).to(torch.int32)
        step_xyz = sign_directions * self.cell_sizes.unsqueeze(0)     # N x 3, we add some noise to avoid numerical instability

        # TODO: NOT TERMINATING!!!
        counter = 0
        while len(points) > 0:

            # tnow = time.time()
            # torch.cuda.synchronize()

            # Begin the incremental traversal procedure
            next_points, next_indices, min_t = self.one_step_voxel_ray_intersection(points, voxel_index, directions, sign_directions, step_xyz)

            # torch.cuda.synchronize()
            # print("Time taken for one step intersection: ", time.time() - tnow)

            # tnow = time.time()
            # torch.cuda.synchronize()

            # If the min_t is greater than 1, we have reached the end of the ray
            terminated = min_t >= 1.

            # We also want to terminate if the ray leaves the voxel grid (NOTE: !!! IMPORTANT !!! This is true only for a single voxel grid!!! We
            # just need to truncate and store the ray if we have multiple voxel grids)
            out_of_bounds = torch.any( torch.logical_or(next_indices < 0, next_indices - self.discretizations.unsqueeze(0) > -1) , dim=-1)     #N

            terminated = torch.logical_or(terminated, out_of_bounds)

            if termination_fn is not None:
                terminated = torch.logical_or(terminated, termination_fn(next_points, next_indices))

            terminated_points = next_points[terminated]

            # If the min_t is less than 1, we have not reached the end of the ray
            not_terminated = ~terminated
            points = next_points[not_terminated]
            voxel_index = next_indices[not_terminated]

            sign_directions = sign_directions[not_terminated]
            step_xyz = step_xyz[not_terminated]

            # We want to truncate the directions to account for traversing by min_t
            directions = (1. - min_t[not_terminated]).unsqueeze(-1) * directions[not_terminated]

            #print("Number of rays left: ", len(points))
            # print(min_t[not_terminated])
            # print(voxel_index)

            voxel_intersections.append(voxel_index)

            # torch.cuda.synchronize()
            # print("Time taken for cleanup: ", time.time() - tnow)
            # counter += 1
            # print('Counter: ', counter)

        return torch.cat(voxel_intersections, dim=0)
        #return voxel_index

    # Renders the voxel grid, given the voxel grid, camera extrinsics, and intrinsics.
    # Makes use of the nerfacc library for fast rendering.
    # We can play with some of the approximations, or make it exact if need be.
    def render_from_voxel_grid(self, voxel_grid, origins, directions, max_steps):
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