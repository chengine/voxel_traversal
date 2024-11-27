import numpy as np

pcd = np.loadtxt("./pcds/pcd_4.txt")

xyzs = pcd[:,0:3]
rgbs = pcd[: 3]
sims = pcd[:, 4]
origins = pcd[:, 5::]

directions = xyzs - origins