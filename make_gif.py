#%%
import glob
import numpy as np
from PIL import Image, ImageDraw
import cv2

def make_gif(frame_folder):
    frames = [Image.open(f"{frame_folder}/{i}.png") for i in range(len(glob.glob(f"{frame_folder}/*.png")))]
    frame_one = frames[0]
    videodims = tuple(np.array(frame_one).shape[:-1][::-1])
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
    # video = cv2.VideoWriter(f"{frame_folder}/render.mp4",fourcc, 4,videodims)
    # for frame in frames:
    #     # draw frame specific stuff here.
    #     video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    # video.release()

    # create gif
    frames[0].save(f'{frame_folder}/render.gif', save_all=True, append_images=frames[1:], optimize=False, duration=10, loop=0)
    
make_gif("output")
# %%
