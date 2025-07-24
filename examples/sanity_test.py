'''
Author: Yorai Shaoul
Data: March 2023

A script to test general functionality of the TartanAir toolbox.
'''
import sys
import numpy as np
import cv2

from colorama import Fore, Style
sys.path.append("..")
import tartanair as ta

env = 'ArchVizTinyHouseDayExposure'
difficulty = 'hard'
traj_name = 'P001'

# Start with initialization.
# Create a TartanAir object.
tartanair_data_root = './data/tartanair-v2'
azure_token = "?s5cxg%3D"
ta.init(tartanair_data_root, azure_token)

############################
# Download.
############################
# Download example. Downloading data from a very small environment to save time.
# Specify the environments, difficulties, and trajectory ids to load.
envs = ['ArchVizTinyHouseDayExposure']
difficulties = ['hard']
trajectory_ids = ['P000', 'P001']

# Specify the modalities to load.
modalities = ['image', 'depth', 'pose', 'imu_acc', 'flow']
camnames = ['lcam_front', 'lcam_left']

ta.download(env = envs, difficulty = difficulties, trajectory_id = trajectory_ids, modality = modalities, camera_name = camnames)

# Verify that the files are where they are supposed to be.
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Get the path to the downloaded data.
downloaded_data_dir_path = os.path.join(tartanair_data_root, env, 'Data_' + difficulty, traj_name)

# Check that the number of files is correct.
assert len(os.listdir(os.path.join(downloaded_data_dir_path, 'image_lcam_front'))) == len(os.listdir(os.path.join(downloaded_data_dir_path, 'image_lcam_left')))
print(Fore.GREEN + "Download OK." + Style.RESET_ALL)


############################
# Dataloader.
############################

new_image_shape_hw = [640, 640] # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
subset_framenum = 364 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
seq_length = {'image': 2, 'depth': 1, 'pose': 2, 'imu': 10} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
batch_size = 8 # This is the number of data-sequences in a mini-batch.
num_workers = 4 # This is the number of workers to use for loading the data.
shuffle = False # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)\

# Create a dataloader object.
dataloader = ta.dataloader(env = envs, 
            difficulty = difficulties, 
            trajectory_id = trajectory_ids, 
            modality = modalities, 
            camera_name = camnames, 
            new_image_shape_hw = new_image_shape_hw, 
            subset_framenum = subset_framenum, 
            seq_length = seq_length, 
            seq_stride = seq_stride, 
            frame_skip = frame_skip, 
            batch_size = batch_size, 
            num_workers = num_workers, 
            shuffle = shuffle,
            verbose = True)


# Iterate over the dataloader and visualize the output.

# Iterate over the batches.
for i in range(30):    
    # Get the next batch.
    batch = dataloader.load_sample()
    # Check if the batch is None.
    if batch is None:
        break
    print("Batch number: {}".format(i), "Loaded {} samples so far.".format(i * batch_size))
    for b in range(batch_size):
        # Visualize some images.
        # The shape of an image batch is (B, S, H, W, C), where B is the batch size, S is the sequence length, H is the height, W is the width, and C is the number of channels.
        img0 = batch['rgb_lcam_front'][b][0] 
        img1 = batch['rgb_lcam_front'][b][1]

        # Visualize the images.
        outimg = np.concatenate((img0, img1), axis = 1)
        cv2.imshow('outimg', outimg)
        cv2.waitKey(1)
        
dataloader.stop_cachers()


############################
# Customization.
############################
from scipy.spatial.transform import Rotation
R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

cam_model_0 = {'name': 'pinhole', 
                'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
               'params': 
                        {'fx': 32, 'fy': 32, 'cx': 32, 'cy': 32, 'width': 64, 'height': 64},
                'R_raw_new': R_raw_new0}


ta.customize(env = env, difficulty = difficulty, trajectory_id = [traj_name], modality = ['image'], new_camera_models_params=[cam_model_0], num_workers = 2, device='cpu') 
assert len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_front', '*.png'))) == len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_custom0_pinhole', '*.png')))
print(Fore.GREEN + "Customization on CPU OK." + Style.RESET_ALL)

R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

cam_model_1 = {'name': 'doublesphere',
                'raw_side': 'left',
                'params':
                        {'fx': 250, 
                        'fy':  250, 
                        'cx': 500, 
                        'cy': 500, 
                        'width': 1000, 
                        'height': 1000, 
                        'alpha': 0.6, 
                        'xi': -0.2, 
                        'fov_degree': 195},
                'R_raw_new': R_raw_new1}

ta.customize(env = env, difficulty = difficulty, trajectory_id = [traj_name], modality = ['image'], new_camera_models_params=[cam_model_1], num_workers = 2, device='cuda') 
assert len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_front', '*.png'))) == len(glob.glob(os.path.join(downloaded_data_dir_path, 'image_lcam_custom0_doublesphere', '*.png')))

print(Fore.GREEN + "Customization on GPU OK." + Style.RESET_ALL)
