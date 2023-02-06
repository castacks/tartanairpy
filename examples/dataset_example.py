'''
Author: Yorai Shaoul
Date: 2023-02-05

Example script for creating a Pytorch dataset using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('../src/')
from tartanair.tartanair import TartanAir

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
ta = TartanAir(tartanair_data_root)

# Download a trajectory.
dataset = ta.create_image_dataset(env = 'ConstructionSite', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image'], camera_name = ['lcam_fish', 'lcam_front'])

