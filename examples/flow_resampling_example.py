'''
Author: Yuchen Zhang
Date: 2024-11-19

Example script for sample new correspondence from the TartanAir dataset.
'''
import tartanair as ta

# For help with rotations.
from scipy.spatial.transform import Rotation

# Initialize TartanAir.
tartanair_data_root = 'your/path/to_tav2'
ta.init(tartanair_data_root)

# Create your camera model(s).
# the rotation matrix represents the rotation from the actual camera to the front camera
R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

cam_model_0 =  {'name': 'pinhole',
                'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
                'params':
                        {'fx': 320,
                         'fy': 320,
                         'cx': 320,
                         'cy': 320,
                         'width': 640,
                         'height': 640},
                'R_raw_new': R_raw_new0}

R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

cam_model_1 = {'name': 'doublesphere',
               'raw_side': 'left',
               'params':
                        {'fx': 300,
                        'fy': 300,
                        'cx': 500,
                        'cy': 500,
                        'width': 1000,
                        'height': 1000,
                        'alpha': 0.6,
                        'xi': -0.2,
                        'fov_degree': 195},
               'R_raw_new': R_raw_new1}

# Customize the dataset.
ta.customize_flow(env = 'ArchVizTinyHouseDay',
             difficulty = 'easy',
             trajectory_id = ['P000'],
             cam_sides=["left", "right"],
             num_workers = 4,
             frame_sep = 1,
             device = "cuda") # or cpu