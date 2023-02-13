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
sys.path.append('../src/')
from tartanair.tartanair import TartanAir

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
ta = TartanAir(tartanair_data_root)

# Create the requested camera models and their parameters.
R_raw_new0 = Rotation.from_euler('z', 45, degrees=True).as_matrix().tolist()

cam_model_0 = {'name': 'pinhole', 
                'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
               'params': 
                        {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480},
                'R_raw_new': R_raw_new0}

R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

cam_model_1 = {'name': 'doublesphere',
                'raw_side': 'left',
                'params':
                        {'fx': 300, 
                        'fy': 300, 
                        'cx': 500, 
                        'cy': 500, 
                        'width': 1000, 
                        'height': 1000, 
                        'alpha': 0.6, 
                        'xi': -0.2, 
                        'fov_degree': 195},
                'R_raw_new': R_raw_new1}

ta.customize(env = 'OldBrickHouse', difficulty = 'easy', trajectory_id = ['P003'], modality = ['image', 'depth'], new_camera_models_params=[cam_model_0, cam_model_1], num_workers = 2)
