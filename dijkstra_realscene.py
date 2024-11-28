#%%
import numpy as np
import torch
import open3d as o3d
import time
import copy
import matplotlib
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch.optim as optim
import cv2
import dijkstra3d
from traversal_utils import *
from covariance_utils import angle_axis_to_rotation_matrix

class Planner():
    def __init__(self):
        # torch stuff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        # import mesh
        # self.obj_path = 'stanford-bunny.obj'
        # self.mesh = o3d.io.read_triangle_mesh(self.obj_path)
        # self.mesh.compute_vertex_normals()
        # self.mesh.rotate(self.mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=np.zeros(3))
        # self.mesh.translate(-self.mesh.get_center())
        # self.points = np.asarray(self.mesh.vertices)
        # self.bounding_box = self.mesh.get_axis_aligned_bounding_box()
        # print("bounding_box: ", self.bounding_box.get_min_bound())

        self.pcd = np.loadtxt("./pcds/pcd_4.txt")

        self.points = self.pcd[:,0:3]
        self.rgbs = self.pcd[: 3]
        self.sims = self.pcd[:, 4]
        self.origins = self.pcd[:, 5::]
        self.directions = self.points - self.origins
        unique_outs, unique_counts = np.unique(self.origins, axis = 0, return_counts=True)
        self.N_training_cameras = unique_outs.shape[0]
        print("N training cameras: ", self.N_training_cameras)
        print("num points: ", self.points.shape[0])
        print("unique counts: ", unique_counts)
        x_max = np.max(self.points[:,0])
        x_min = np.min(self.points[:,0])
        x_mean = np.mean(self.points[:,0])
        x_stdev = np.std(self.points[:,0])
        print("x: ", x_mean, x_stdev)

        y_max = np.max(self.points[:,1])
        y_min = np.min(self.points[:,1])
        y_mean = np.mean(self.points[:,1])
        y_stdev = np.std(self.points[:,1])
        print("y: ", y_mean, y_stdev)

        self.cmap = matplotlib.cm.get_cmap('turbo')
        # Parameter initializations of VoxelGrid
        self.param_dict = {
                'discretizations': torch.tensor([100, 100, 100], device=self.device),
                'lower_bound': torch.tensor([-50.0, -50.0, 0.0], device=self.device),
                'upper_bound': torch.tensor([50.0, 50.0, 2.0], device=self.device),
                'voxel_grid_values': None,
                'voxel_grid_binary': None,
                'termination_fn': self.termination_fn,
                'termination_value_type': "colors"
            }
        # Camera Information
        self.K = torch.tensor([
                            [100., 0., 100.],
                            [0., 100., 100.],
                            [0., 0., 1.]
                        ], device=self.device)
        self.far_clip = 10.
        self.N_training_cameras = self.pcd.shape[0]
        # Initialize the VoxelGrid
        self.vgrid = VoxelGrid(self.param_dict, 3, self.device)
        
    def termination_fn(self, next_points, directions, next_indices, voxel_grid_binary, voxel_grid_values, termination_value_type):

        # Possible for indices to be out of bounds
        discretization = torch.tensor(voxel_grid_binary.shape, device=self.device)
        out_of_bounds = torch.any( torch.logical_or(next_indices < 0, next_indices - discretization.unsqueeze(0) > -1) , dim=-1)

        out_values = torch.zeros_like(next_points).to(torch.float32)

        # Mask out of bounds points
        next_points = next_points[~out_of_bounds]
        directions = directions[~out_of_bounds]
        next_indices = next_indices[~out_of_bounds]

        mask = voxel_grid_binary[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
        
        if termination_value_type == "colors":
            values = voxel_grid_values[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]]
            out_mask = ~out_of_bounds
            out_mask[~out_of_bounds] = mask
            out_values[~out_of_bounds] = values
        elif termination_value_type == "uncertainty":
            eigvals, eigvecs = voxel_grid_values
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

    def generate_training_view_info(self):
        colors = self.cmap( (self.points[:, 2] - self.points[:, 2].min()) / (self.points[:, 2].max() - self.points[:, 2].min()))[:, :3]        
        self.vgrid.termination_value_type = "colors"
        self.vgrid.termination_fn = self.termination_fn
        self.vgrid.populate_voxel_grid_from_points(torch.tensor(self.points, device=self.device), torch.tensor(colors, device=self.device, dtype=torch.float32))

        # combined image
        cmap = matplotlib.cm.get_cmap('turbo')

        # W, H = int(self.K[0, 2] * 2), int(self.K[1, 2] * 2)
        # B = np.unique(self.origins, axis=0).shape[0]
        # print("B: ", B)
        # print("num points: ", self.origins.shape[0])

        # # Flatten origins and directions
        origins = torch.from_numpy(self.origins).reshape(-1, 3).to(self.device).to(torch.float32)
        directions = torch.from_numpy(self.directions).reshape(-1, 3).to(self.device).to(torch.float32)

        # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        # directions = directions * self.far_clip
        # output = self.vgrid.compute_voxel_ray_intersection(origins, directions)
        # image = torch.zeros((B, H, W, 3), device=self.device)
        # depth = torch.zeros((B, H, W), device=self.device)

        # # Compute the depth and image
        # image = image.reshape(-1, 3)
        # image[output['terminated_ray_index']] = output['terminated_voxel_values']
        # image = image.reshape(B, H, W, 3)

        # depth = depth.reshape(-1)
        # depth[output['terminated_ray_index']] = torch.linalg.norm(self.vgrid.voxel_grid_centers[output['terminated_voxel_index'][:, 0], 
        #                                                                 output['terminated_voxel_index'][:, 1], 
        #                                                                 output['terminated_voxel_index'][:, 2]] - origins[output['terminated_ray_index']], dim=-1)
        # depth = depth.reshape(B, H, W)

        # output['rays_o'] = origins
        # output['rays_d'] = directions

        # ### SEGMENT OUT THE RAYS THAT INTERSECTED THE VOXEL GRID ###
        # depths = depth.reshape(-1)
        # directions = output['rays_d'].reshape(-1, 3)
        # origins = output['rays_o'].reshape(-1, 3)

        # # For depths greater than 0
        # mask = depths > 0
        # depths = depths[mask]
        # directions = directions[mask]
        # origins = origins[mask]

        # Termination points
        directions = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)
        # xyzs = origins + depths[:, None] * directions
        xyzs =  torch.from_numpy(self.points).reshape(-1, 3).to(self.device).to(torch.float32)

        voxel_index, in_bounds = self.vgrid.compute_voxel_index(xyzs)
        voxel_index = voxel_index[in_bounds]
        print("directions total: ", directions.shape)

        directions = directions[in_bounds]
        print("directions in bounds: ", directions.shape)
        # Create outer product from normalized directions
        outer_product = torch.einsum("bi,bj->bij",directions, directions)

        voxel_index_flatten = torch.tensor(np.ravel_multi_index(voxel_index.T.cpu().numpy(), tuple(self.vgrid.discretizations)), device=self.vgrid.device)

        # Store the number of counts for each voxel
        voxel_counts = torch.zeros(tuple(self.vgrid.discretizations), device=self.vgrid.device, dtype=voxel_index_flatten.dtype).flatten()
        voxel_counts.scatter_add_(0, voxel_index_flatten, torch.ones_like(voxel_index_flatten))
        voxel_counts = voxel_counts.reshape(tuple(self.vgrid.discretizations))

        # Store the covariance matrix for each voxel
        voxel_covariance = torch.zeros(tuple(self.vgrid.discretizations) + (3, 3), device=self.vgrid.device, dtype=outer_product.dtype).reshape(-1, 3, 3)
        voxel_covariance.scatter_add_(0, voxel_index_flatten.unsqueeze(1).unsqueeze(1).expand(-1, 3, 3), outer_product)

        voxel_covariance = voxel_covariance / (voxel_counts.reshape(-1).unsqueeze(1).unsqueeze(1) + 1e-6)

        eigvals, eigvecs = torch.linalg.eigh(voxel_covariance)

        eigvals = eigvals.reshape(100, 100, 100, 3)
        eigvecs = eigvecs.reshape(100, 100, 100, 3, 3)
        return eigvals, eigvecs

    def look_at(self, location, target, up):
        z = (location - target)
        z_ = z / torch.norm(z)
        x = torch.cross(up, z)
        x_ = x / torch.norm(x)
        y = torch.cross(z, x)
        y_ = y / torch.norm(y)

        R = torch.stack([x_, y_, z_], dim=-1)
        return R

    def generate_info_gain(self, positions, eigvals, eigvecs):
        '''
        positions: N_samples x 3 pytorch tensor that lists the xyz positions to compute the information gain from
        '''
        self.vgrid.termination_value_type = "uncertainty"
        self.vgrid.voxel_grid_values = (eigvals, eigvecs)
        self.vgrid.termination_fn = self.termination_fn

        N_sampled_cameras = positions.shape[0]
        positions_cart = self.vgrid.voxel_grid_centers[positions[:,0], positions[:,1], positions[:,2]]
        # print("positions_cart: ", positions_cart)
        c2w = torch.eye(4, device=self.device)[None].expand(N_sampled_cameras, -1, -1).clone()
        c2w[:, :3, 3] = torch.stack([positions_cart[:,0], positions_cart[:,1], positions_cart[:,2]], dim=-1)

        target = torch.zeros(3, device=self.device)[None].expand(N_sampled_cameras, -1)
        up = torch.tensor([0., 0., 1.], device=self.device)[None].expand(N_sampled_cameras, -1)
        c2w[:, :3, :3] = self.look_at(c2w[:, :3, 3], target, up)

        tnow = time.time()
        torch.cuda.synchronize()

        c2ws = c2w

        c2w_list = torch.split(c2ws, 300)
        images = []
        depths = []
        for c2w in c2w_list:
            # print("c2w shape: ", c2w[0,:,:])

            image, depth, _ = self.vgrid.camera_voxel_intersection(self.K, c2w, self.far_clip)
            images.append(image)
            depths.append(depth)

        images = torch.cat(images, dim=0)
        depths = torch.cat(depths, dim=0)
        torch.cuda.synchronize()
        print("Time taken: ", time.time() - tnow)

        images = images / (torch.amax(images, dim=(1, 2))[:, None, None] + 1e-6)

        uncertainty_metric = torch.sum(images, dim=(1, 2))[..., 0]
        return uncertainty_metric, images.cpu().numpy()

    def generate_dijkstra_paths(self, source, plot_graph = True):
        voxel_grid_3d = self.vgrid.voxel_grid_binary.cpu().numpy()
        voxel_grid_2d = voxel_grid_3d[:,:,48:52] # 1 where occupied, 0 where unoccupied

        dist_field = dijkstra3d.euclidean_distance_field(np.invert(voxel_grid_2d), source=source)
        inf_indices = np.where(dist_field == np.inf)
        pixel_radius = 10
        for i in range(pixel_radius):
            dist_field[np.clip(inf_indices[0]+i, 0, 99), inf_indices[1],inf_indices[2]] = np.inf
            dist_field[np.clip(inf_indices[0]-i, 0, 99), inf_indices[1],inf_indices[2]] = np.inf
            dist_field[inf_indices[0], np.clip(inf_indices[1]+i, 0, 99),inf_indices[2]] = np.inf
            dist_field[inf_indices[0], np.clip(inf_indices[1]-i, 0, 99),inf_indices[2]] = np.inf
        print("dist_field: ", dist_field[25,:,0])
        parents = dijkstra3d.parental_field(dist_field, source=source, connectivity=26)

        ax3 = plt.figure().add_subplot(projection='3d')
        ax3.voxels(voxel_grid_2d,alpha=1)

        voxel_inds = np.indices(voxel_grid_2d.shape)
        voxel_inds = np.vstack((voxel_inds[0].flatten(),voxel_inds[1].flatten(),voxel_inds[2].flatten()))
        trajectories = []
        print("num trajectories: ", voxel_inds.shape[1])
        for i in range(0,voxel_inds.shape[1], 1000):
            target = tuple(voxel_inds[:,np.random.randint(voxel_inds.shape[1])])
            if voxel_grid_2d[target[0], target[1], target[2]] == 1:
                print("occupied target!")
                continue
            cost = dist_field[target[0], target[1], target[2]]
            if cost == np.inf:
                print("infinite cost!")
                continue
            path = dijkstra3d.path_from_parents(parents, target=tuple(target))
            if path.shape[0] <= 1:
                print("no path found!")
                continue
            # remap path to 3D voxel grid:
            path_3d = copy.deepcopy(path)
            path_3d[:,2] += 48
            print("path: ", path.shape, " target: ", target, " cost: ", cost)
            positions = []
            for position in path_3d:
                positions.append(list(position))
            ax3.plot(path[:,0], path[:,1], path[:,2],c='g')
            ax3.scatter(target[0], target[1], target[2],c='r')
            trajectories.append(np.asarray(positions))
        ax3.scatter(source[0], source[1], source[2], c='c')
        ax3.view_init(azim=0, elev=90)
        plt.savefig("./plots/image_dijkstra_top.png")
        return trajectories, ax3 
        
    def score_paths(self, trajectories, eigvals, eigvecs):
        trajectory_uncertainties = []
        trajectory_images = []
        for trajectory in trajectories:
            positions = np.asarray(trajectory, dtype=np.int32)
            traj_length = positions.shape[0]
            # if traj_length > 50:
            #     positions = positions[0:50, :]
            positions_tensor = torch.from_numpy(positions)
            trajectory_uncertainty_metric, images = self.generate_info_gain(positions_tensor, eigvals, eigvecs)
            total_trajectory_uncertainty_metric = torch.mean(trajectory_uncertainty_metric) # consider taking a mean over the trajectory
            trajectory_uncertainties.append(total_trajectory_uncertainty_metric.item())
            trajectory_images.append(images)
        return np.asarray(trajectory_uncertainties), trajectory_images

    def choose_best_path(self, trajectory_uncertainties):
        indices = np.argsort(trajectory_uncertainties)
        lowest_uncertainty = trajectory_uncertainties[indices[-1]]
        return indices[-1]

def main():
    planner_class = Planner()
    eigvals, eigvecs = planner_class.generate_training_view_info()
    print("eigvals: ", eigvals.shape)
    print("eigvecs: ", eigvecs.shape)
    trajectories, ax3 = planner_class.generate_dijkstra_paths(source=(0,0,0))
    uncertainties, images = planner_class.score_paths(trajectories, eigvals, eigvecs)
    print("uncertainties: ", uncertainties)
    best_idx = planner_class.choose_best_path(uncertainties)
    best_trajectory = trajectories[best_idx]
    best_images = images[best_idx]
    for i, image in enumerate(best_images):
        # image = image_tensor.cpu().numpy()
        image = image / (image.max() + 1e-6)
        image = image * 255
        cv2.imwrite("./traj_images/frame_"+str(i).zfill(3)+".png", image)
    ax3.plot(best_trajectory[:,0], best_trajectory[:,1], 0*best_trajectory[:,2], c='m')
    ax3.view_init(azim=0, elev=90)
    plt.savefig("./plots/image_dijkstra_top.png")
    
if __name__ == "__main__":
    main()