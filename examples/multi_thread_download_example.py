# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'

ta.init(tartanair_data_root)

# Download data from following environments.
env = [ "Prison",
        "Ruins",
        "UrbanConstruction",
]

ta.download_multi_thread(env = env, 
              difficulty = ['easy', 'hard'], 
              modality = ['image', 'depth'],  
              camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'], 
              unzip = True,
              num_workers = 8)

# To download the entire dataset
alldata = ta.get_all_data() # this fill in all available data for env, difficulty, modality and camera_name
ta.download_multi_thread(**alldata, 
                            unzip = True,
                            num_workers = 8)
