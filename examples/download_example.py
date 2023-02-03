'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
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
ta.download(env = 'AmericanDinerExposure', difficulty = 'easy', trajectory_id = ['P000', 'P003'], modality = ['imu', 'image'], camera_name = ['lcam_fish'])
