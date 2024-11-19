'''
Author: Yuchen Zhang
Date: 2024-07-01

Example script for sample new correspondence from the TartanAir dataset.
'''

# General imports.
import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import flow_vis

# Local imports.
sys.path.append('..')
from tartanair.flow_calculation import *
from tartanair.flow_utils import *

import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/hdd_data/tartanair_v2'
ta.init(tartanair_data_root)

# Create the requested camera models and their parameters.
cam_model_0 = {'name': 'pinhole', 
               'params': 
                        {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 320, 'width': 640, 'height': 640},
                }

# cam_model_1 = {'name': 'doublesphere',
#                 'params':
#                         {'fx': 250, 
#                         'fy':  250, 
#                         'cx': 500, 
#                         'cy': 500, 
#                         'width': 1000, 
#                         'height': 1000, 
#                         'alpha': 0.6, 
#                         'xi': -0.2, 
#                         'fov_degree': 195},
#                 }

cam_model_1 = {'name': 'pinhole', 
               'params': 
                        {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 320, 'width': 800, 'height': 640},
                }

device = "cuda" if torch.cuda.is_available() else 'cpu'

print("Using device:", device)

random_accessor = ta.get_random_accessor()

random_accessor.cache_tartanair_pose()

cam_model_0_obj = random_accessor.generate_camera_model_object_from_config(cam_model_0)
cam_model_1_obj = random_accessor.generate_camera_model_object_from_config(cam_model_1)

traj_index0 = {
    "env" : "SeasideTown",
    "difficulty": "hard",
    "id": "P000",
    "cam_side": "lcam"
}

frame_idx_0 = 10

traj_index1 = {
    "env" : "SeasideTown",
    "difficulty": "hard",
    "id": "P000",
    "cam_side": "lcam"
}

frame_idx_1 = 20

# get cubemap images. This includes RGB, depth for all 6 faces of the cubemap.
cubemap_images_0 = random_accessor.get_cubemap_images_parallel(
    traj_index0,
    frame_idx=frame_idx_0
)

cubemap_images_1 = random_accessor.get_cubemap_images_parallel(
    traj_index1,
    frame_idx=frame_idx_1
)

# get pose
pose0 = random_accessor.get_front_cam_NED_pose(traj_index0, frame_idx_0)
pose1 = random_accessor.get_front_cam_NED_pose(traj_index1, frame_idx_1)

# move the camera models and images to GPU
cam_model_0_obj.device = device # special setter will handle moving internal tensors to the device
cam_model_1_obj.device = device

def convert_tensor(rendered, device):
    
    for key, images in rendered.items():
        if key == 'image':
            for k, v in images.items():
                v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                rendered[key][k] = (torch.from_numpy(v).to(device).float() / 255.0).permute(2, 0, 1)
        elif key == 'depth':
            for k, v in images.items():
                rendered[key][k] = torch.from_numpy(v).to(device).unsqueeze(0).float()
        else:
            raise NotImplementedError("Unknown key: {}".format(key))
    
    return rendered
    
cubemap_images_0 = convert_tensor(cubemap_images_0, device=device)
cubemap_images_1 = convert_tensor(cubemap_images_1, device=device)

# render the images according to camera models. May select a rotation
rotation0 = torch.eye(3) # this can also be from camera model.
rotation1 = torch.eye(3)

rendered_0, mask_0 = render_images_from_cubemap(cubemap_images_0, cam_model_0_obj, rotation=rotation0, device=device)
rendered_1, mask_1 = render_images_from_cubemap(cubemap_images_1, cam_model_1_obj, rotation=rotation1, device=device)

# compute correspondence
depth_value_gt, depth_error, fov_mask, valid_pixels_0, valid_pixels_1, flow_image, world_T_0, world_T_1 = calculate_pairwise_flow(
    pose0, rotation0, rendered_0['depth'].to(device), mask_0.to(device), cam_model_0_obj, 
    pose1, rotation1, rendered_1['depth'].to(device), mask_1.to(device), cam_model_1_obj,
    device=device
)

# compute occlusion
non_occluded_prob = calculate_occlusion(
    rendered_0['depth'].to(device), rendered_1['depth'].to(device), 
    valid_pixels_0, valid_pixels_1,
    depth_value_gt, device=device,
    depth_start_threshold=0.04,
    depth_temperature=0.02,
    apply_relative_error=True,
    relative_error_tol=0.01
)

valid = torch.zeros(fov_mask.shape, device=fov_mask.device, dtype=torch.float32)
valid[fov_mask] = non_occluded_prob # probability of not occluded is less then 0.5

# visualize everything
fig, axs = plt.subplots(3, 3, figsize=(20, 10))

axs[0, 0].imshow(rendered_0['image'].cpu().numpy())
axs[0, 0].set_title("RGB 0")

axs[0, 1].imshow(rendered_1['image'].cpu().numpy())
axs[0, 1].set_title("RGB 1")

axs[1, 0].imshow(mask_0.cpu().numpy(), vmin=0, vmax=1)
axs[1, 0].set_title("Mask 0")

axs[1, 1].imshow(mask_1.cpu().numpy(), vmin=0, vmax=1)
axs[1, 1].set_title("Mask 1")

axs[2, 0].imshow(np.log(rendered_0['depth'].cpu().numpy()))
axs[2, 0].set_title("Depth 0")

axs[2, 1].imshow(np.log(rendered_1['depth'].cpu().numpy()))
axs[2, 1].set_title("Depth 1")

axs[0, 2].imshow(flow_vis.flow_to_color(flow_image.cpu().numpy()))
axs[0, 2].set_title("Flow without occlusion")

axs[1, 2].imshow(fov_mask.cpu().numpy(), vmin=0, vmax=1)
axs[1, 2].set_title("FOV mask")

axs[2, 2].imshow(valid.cpu().numpy(), vmin=0, vmax=1)
axs[2, 2].set_title("Not occluded")

fig.savefig("flow_sampling_example.png")

