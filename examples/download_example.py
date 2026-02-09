'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
sys.path.append('/home/nicholas/tartanairpy')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/home/nicholas/tartanair_v2'

ta.init(tartanair_data_root)

# Download data from following environments.
env = [ "House"]

ta.download(env = env, 
              difficulty = ['easy'], 
              modality = ['depth'],  
              camera_name = ['lcam_front'], 
              unzip = True,
              delete_zip = False,
              num_workers = 4)

# Can also download via a yaml config file.
# ta.download(config = 'download_config.yaml')
