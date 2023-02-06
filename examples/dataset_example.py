'''
Author: Yorai Shaoul
Date: 2023-02-05

Example script for creating a Pytorch dataset using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('../src/')
from tartanair.tartanair import TartanAir

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
ta = TartanAir(tartanair_data_root)

# Download a trajectory.
dataset = ta.create_image_dataset(env = 'ConstructionSite', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image', 'depth'], camera_name = ['lcam_fish', 'lcam_front'])

# Print the dataset.
print(dataset)


# Create a torch dataloader.
import torch
from torch.utils.data import Dataset, DataLoader

dataloader = DataLoader(dataset, batch_size = 3, shuffle = True, num_workers = 0)

# Show a few images.
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['lcam_fish']['image_0'].size())

    # Show the batch side by side.
    import cv2
    import numpy as np
    img0_0 = sample_batched['lcam_fish']['image_0'][0].numpy()
    img1_0 = sample_batched['lcam_fish']['image_0'][1].numpy()
    img2_0 = sample_batched['lcam_fish']['image_0'][2].numpy()

    img0_1 = sample_batched['lcam_fish']['image_1'][0].numpy()
    img1_1 = sample_batched['lcam_fish']['image_1'][1].numpy()
    img2_1 = sample_batched['lcam_fish']['image_1'][2].numpy()
    
    img0_0 = cv2.resize(img0_0, (0, 0), fx = 0.5, fy = 0.5)
    img1_0 = cv2.resize(img1_0, (0, 0), fx = 0.5, fy = 0.5)
    img2_0 = cv2.resize(img2_0, (0, 0), fx = 0.5, fy = 0.5)

    img0_1 = cv2.resize(img0_1, (0, 0), fx = 0.5, fy = 0.5)
    img1_1 = cv2.resize(img1_1, (0, 0), fx = 0.5, fy = 0.5)
    img2_1 = cv2.resize(img2_1, (0, 0), fx = 0.5, fy = 0.5)


    img = np.concatenate((img0_0, img1_0, img2_0), axis = 1)
    img1 = np.concatenate((img0_1, img1_1, img2_1), axis = 1)
    img = np.concatenate((img, img1), axis = 0)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    if i_batch == 5:
        break




