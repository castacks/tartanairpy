'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys
import numpy as np

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
 
ta.init(tartanair_data_root)

# List available trajectories.
traj = ta.get_traj_np(env = 'ConstructionSite', difficulty = 'easy', trajectory_id = "P000",  camera_name = 'lcam_front')
print(traj.shape)
np.set_printoptions(precision=3, suppress=True)
print(traj[0:10, :])