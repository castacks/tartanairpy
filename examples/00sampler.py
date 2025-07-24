'''
Author: Yuchen Zhang
Date: 2023-02-05

Example script for synthesizing data in new camera-models from the TartanAir dataset.
'''

# General imports.
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

# Local imports.
sys.path.append('.')
import tartanair as ta

from tartanair.customizer import TartanAirMatchingSampler

# Create a TartanAir object.
tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'
ta.init(tartanair_data_root)

sampler = TartanAirMatchingSampler(tartanair_data_root)


trajectory_idx = {
    "env": "SeasideTownExposure",
    "difficulty": "Data_easy",
    "id": "P000"
}

# sampler.interval_sampling(10, None, trajectory_idx)

R_raw_new = torch.eye(3).float()

camera_model_config = {
'name': 'doublesphere',
'params':
    {   
        'fx': 250, 
        'fy':  250, 
        'cx': 500, 
        'cy': 500, 
        'width': 1000, 
        'height': 1000, 
        'alpha': 0.6, 
        'xi': -0.3, 
        'fov_degree': 195
    }
}


output = sampler.map_to_new_cam_model(trajectory_idx, 0, camera_model_config, R_raw_new, cam_side="lcam")


fig, axs = plt.subplots(1, 3, figsize=(30,10))

axs[0].imshow(output["image"])
axs[1].imshow(np.log(output["depth"] + 1e-4))
axs[2].imshow(output["mask"])

fig.savefig("/ocean/projects/cis220039p/yzhang25/tartanairpy/sample_test.png")

print(output["depth"].mean())

# # Create the requested camera models and their parameters.
# R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

# cam_model_0 = {'name': 'pinholeyuchen', 
#                 'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
#                'params': 
#                         {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 320, 'width': 640, 'height': 640},
#                 'R_raw_new': R_raw_new0}

# R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

# cam_model_1 = {'name': 'doublesphereyuchen',
#                 'raw_side': 'left',
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
#                 'R_raw_new': R_raw_new1}

# ta.customize(env = 'SeasideTownExposure', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image'], new_camera_models_params=[cam_model_1, cam_model_0], num_workers = 2, device='cuda') # Or cpu.
