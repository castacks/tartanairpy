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

# Download all the data using multiple threads
ta.download_ground_multi_thread(env = [], 
              version = [], 
              modality = [],  
              camera_name = [], 
              unzip = True, 
              num_workers = 8)
