#%%
import torch
import numpy as np
from camera_utils import get_rays, get_rays_batch

device = torch.device('cuda:0')
H = 1000
W = 1000
K = torch.tensor([
    [1000., 0., 500.],
    [0., 1000., 500.],
    [0., 0., 1.]
])
c2w = torch.eye(4, device=device)[None].expand(100, -1, -1)

get_rays_batch(H, W, K, c2w, device)