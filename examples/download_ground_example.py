# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanground_data_root = '/my/path/to/root/folder/for/tartan-ground'

ta.init(tartanground_data_root)

# Download data from following environments.
env = [ "Prison",
        "Ruins",
        "UrbanConstruction",
]

ta.download_ground(env = env, 
              version = ['v1', 'v2', 'v3_anymal'], 
              modality = ['seg', 'lidar', 'imu'],  
              camera_name = ['lcam_front', 'lcam_bottom'], 
              unzip = True)

env = ['OldTownSummer', 'DesertGasStation']

ta.download_ground_multi_thread(env = env, 
              version = ['v1', 'v2', 'v3_anymal'], 
              modality = ['image', 'depth', 'seg', 'lidar', 'imu'],  
              camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'], 
              unzip = True, 
              num_workers = 8)
