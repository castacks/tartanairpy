'''
Author: Yorai Shaoul
Data: March 2023

A script to test general functionality of the TartanAir toolbox.
'''
import sys

from colorama import Fore, Style
sys.path.append("..")
import tartanair as ta

env = 'ArchVizTinyHouseDayExposure'
difficulty = 'hard'
traj_name = 'P001'

# Start with initialization.
# Create a TartanAir object.
tartanair_data_root = './data/tartanair-v2'
azure_token = "?sv=2021-10-04&st=2023-03-01T1sp=rl&sig=LojCTa60TcA9ApMiMofliedxuu5cxg%3D"
ta.init(tartanair_data_root, azure_token)

############################
# Download.
############################
# Download example. Downloading data from a very small environment to save time.
# ta.download(env = env, difficulty = ['hard'], trajectory_id = ["P001"],  modality = ['image'],  camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'])

# Verify that the files are where they are supposed to be.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Get the path to the downloaded data.
downloaded_data_dir_path = os.path.join(tartanair_data_root, env, 'Data_' + difficulty, traj_name)

# Check that the number of files is correct.
assert len(os.listdir(os.path.join(downloaded_data_dir_path, 'image_lcam_front'))) == len(os.listdir(os.path.join(downloaded_data_dir_path, 'image_lcam_left')))
print(Fore.GREEN + "Download OK." + Style.RESET_ALL)


############################
# Customization.
############################
from scipy.spatial.transform import Rotation
R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

cam_model_0 = {'name': 'pinhole', 
                'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
               'params': 
                        {'fx': 32, 'fy': 32, 'cx': 32, 'cy': 32, 'width': 64, 'height': 64},
                'R_raw_new': R_raw_new0}


ta.customize(env = env, difficulty = difficulty, trajectory_id = [traj_name], modality = ['image'], new_camera_models_params=[cam_model_0], num_workers = 2, device='cpu') 
assert len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_front', '*.png'))) == len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_custom0_pinhole', '*.png')))
print(Fore.GREEN + "Customization on CPU OK." + Style.RESET_ALL)

R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

cam_model_1 = {'name': 'doublesphere',
                'raw_side': 'left',
                'params':
                        {'fx': 250, 
                        'fy':  250, 
                        'cx': 500, 
                        'cy': 500, 
                        'width': 1000, 
                        'height': 1000, 
                        'alpha': 0.6, 
                        'xi': -0.2, 
                        'fov_degree': 195},
                'R_raw_new': R_raw_new1}

ta.customize(env = env, difficulty = difficulty, trajectory_id = [traj_name], modality = ['image'], new_camera_models_params=[cam_model_1], num_workers = 2, device='cuda') 
assert len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_front', '*.png'))) == len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_custom0_doublesphere', '*.png')))

print(Fore.GREEN + "Customization on GPU OK." + Style.RESET_ALL)
