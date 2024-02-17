'''
Author: Yorai Shaoul
Date: 2023-03-01

Example script for iterating over the TartanAir dataset.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
 
ta.init(tartanair_data_root)

# List available trajectories.
ta_iterator = ta.iterator(env = ['PolarSciFiExposure'], difficulty = 'easy', trajectory_id = [], modality = 'image', camera_name = ['lcam_left'])

for i in range(100):
    print(next(ta_iterator))