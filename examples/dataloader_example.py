'''
An example for using the dataloader. Note that under the hood the dataloader employs a powerful parallelized systems that simulatanuously loads data and serves mini-batches. So training and loading can happen concurrently.

Author: Yorai Shaoul
Date: 2023-03-28
'''

# General imports.
import sys
import numpy as np
import cv2

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
# tartanair_data_root = './data/tartanair-v2'
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
ta.init(tartanair_data_root)


# Specify the environments, difficulties, and trajectory ids to load.
envs = [
"AbandonedCableExposure",
"AbandonedFactoryExposure",
"AbandonedSchoolExposure",
"AmericanDinerExposure",
"ArchVizTinyHouseDayExposure",
"CarweldingExposure",
"HQWesternSaloonExposure",
"JapaneseCityExposure",
"MiddleEastExposure",
"ModularNeighborhoodIntExt",
"OldScandinaviaExposure",
"ShoreCavesExposure",
]
difficulties = []
trajectory_ids = ['P000']

# Specify the modalities to load.
modalities = ['image', 'depth', 'pose']
camnames = ['lcam_front', 'lcam_left', 'lcam_right', 'lcam_back', 'lcam_top', 'lcam_bottom']

# Specify dataloader configuration. The dataloader operates in the following way:
# 1. It loads a subset of the dataset to RAM.
# 2. It serves mini-batches from this subset. The mini-batches are of data-sequences. So for example, if the sequence length is 2, then the mini-batch will have samples of 2 frames each. A batch size of 16 means that the mini-batch will contain 16 pairs of images, for the example of images. The samples do not have to be consecutive, and the 'skip' between the samples can be specified. The sequences also do not have to start from consecutive indices, and the stride between the sequences can be specified.
# 3. The dataloader will load a new subset of the dataset to RAM while the mini-batches are loaded from the first subset, and switch the subsets when the first subset is exhausted. If the first subset is exhausted before the mini-batches are loaded, then the dataloader will keep loading mini-batches from the first subset until the second subset is loaded.

new_image_shape_hw = [640, 640] # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
subset_framenum = 200 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
seq_length = {'image': 2, 'depth': 1, 'pose': 2} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
batch_size = 16 # This is the number of data-sequences in a mini-batch.
num_workers = 8 # This is the number of workers to use for loading the data.
shuffle = True # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)\

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
for i in range(1000):    
    # Get the next batch.
    batch = dataloader.load_sample()
    # Check if the batch is None.
    if batch is None:
        break
    # Visualize some images.
    # The shape of an image batch is (B, S, H, W, C), where B is the batch size, S is the sequence length, H is the height, W is the width, and C is the number of channels.
    images = []
    for b in range(batch['rgb_lcam_front'].shape[0]):
        images.append(batch['rgb_lcam_front'][b][0])
    # Visualize the images.
    outimg = np.concatenate(images, axis=1)
    outimg = cv2.resize(outimg, (outimg.shape[1]//4, outimg.shape[0]//4))
    cv2.imshow('outimg', outimg)
    cv2.waitKey(1)
    
    
        
dataloader.stop_cachers()