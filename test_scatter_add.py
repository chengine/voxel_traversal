#%%
import torch

A = torch.tensor([[1., 2.], [3., 4.], [5., 3.], [2., 1.], [4., 3.]])
idx = torch.tensor([0, 0, 0, 1, 1]).unsqueeze(1).expand(5, 2)

print(torch.zeros(2, 2).scatter_add_(0, idx, A))

print(torch.zeros(2).scatter_add_(0, idx[:, 0], torch.ones(5)))
# %%
