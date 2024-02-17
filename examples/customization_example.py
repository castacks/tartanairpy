'''
Author: Yorai Shaoul
Date: 2023-02-05

Example script for synthesizing data in new camera-models from the TartanAir dataset.
'''

# General imports.
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Local imports.
sys.path.append('.')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'
ta.init(tartanair_data_root)

# Create the requested camera models and their parameters.
R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

cam_model_0 = {'name': 'pinholeyuchen', 
                'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
               'params': 
                        {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 320, 'width': 640, 'height': 640},
                'R_raw_new': R_raw_new0}

R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

cam_model_1 = {'name': 'doublesphereyuchen',
                'raw_side': 'left',
                'params':
                        {'fx': 250, 
                        'fy':  250, 
                        'cx': 250, 
                        'cy': 250, 
                        'width': 500, 
                        'height': 500, 
                        'alpha': 0.4, 
                        'xi': -0.2, 
                        'fov_degree': 195},
                'R_raw_new': R_raw_new1}

ta.customize(env = 'SeasideTownExposure', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image'], new_camera_models_params=[cam_model_1, cam_model_0], num_workers = 2, device='cuda') # Or cpu.
