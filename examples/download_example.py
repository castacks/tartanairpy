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
tartanair_data_root = '/home/wenshan/tmp' #'/my/path/to/root/folder/for/tartanair-v2'

ta.init(tartanair_data_root)

# Download data from following environments.
env = [ "Prison",
        "Ruins",
        "UrbanConstruction",
]

ta.download(env = env, 
              difficulty = ['easy', 'hard'], 
              modality = ['image', 'depth'],  
              camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'], 
              unzip = True)

# # Can also download via a yaml config file.
# ta.download(config = 'download_config.yaml')
