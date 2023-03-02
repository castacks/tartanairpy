'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2_training_data'
 
ta.init(tartanair_data_root)

# List available trajectories.
ta.visualize('ConstructionSite', difficulty='easy', trajectory_id = 'P000', modality = ['image', 'depth', 'seg'], camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left'])