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
tair = TartanAir(tartanair_data_root)

# Download a trajectory.
tair.download(env = 'ArchVizTinyHouseDayExposure', difficulty = 'hard', trajectory_id = ['P003'], modality = ['seg'], camera_name = ['rcam_fish'])
