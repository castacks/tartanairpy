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
tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'

ta.init(tartanair_data_root)

# List available trajectories.
ta.visualize('ArchVizTinyHouseDay', 
              difficulty='easy', 
              trajectory_id = 'P000', 
              modality = ['image', 'depth', 'seg'], 
              camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left'],
              show_seg_palette = True)