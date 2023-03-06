'''
Author: Yorai Shaoul
Date: 2023-02-05

Example script for creating a Pytorch dataset using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2_training_data'
ta.init(tartanair_data_root)

#####################
# Using a dataloader #
#####################

# Set up the dataset.
dataset = ta.create_image_dataset(env = ['ConstructionSite', 'SupermarketExposure'], difficulty = [], trajectory_id = [], modality = ['image', 'depth'], camera_name = ['lcam_front', 'lcam_back', 'lcam_right', 'lcam_left', 'lcam_top', 'lcam_bottom'], transform = None, num_workers=10)

# Print the dataset.
print(dataset)


# Create a torch dataloader.
import torch
from torch.utils.data import Dataset, DataLoader

dataloader = DataLoader(dataset, batch_size = 3, shuffle = False, num_workers = 0)

# Show a few images.
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['lcam_front']['image_0'].size())

    # Show the batch side by side.
    import cv2
    import numpy as np
    imgs1 = sample_batched['lcam_front']['image_0'].numpy()
    imgs2 = sample_batched['lcam_front']['image_1'].numpy()


    img = np.concatenate((imgs1[0], imgs1[1], imgs1[2]), axis = 1)
    img = np.concatenate((img, np.concatenate((imgs2[0], imgs2[1], imgs2[2]), axis = 1)), axis = 0)
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    if i_batch == 5:
        break


####################
# Using transforms #
####################

# Do this again, this time also create and pass a simple transform.
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    
    # Resize the image.
    transforms.Resize((224, 224)),

    # Randomly flip the image.
    transforms.RandomHorizontalFlip(p = 1.0),

])

# Set up a dataset.
dataset = ta.create_image_dataset(env = [], difficulty = 'easy', trajectory_id = [], modality = ['image', 'depth'], camera_name = ['lcam_fish', 'lcam_front'], transform = transform)

dataloader = DataLoader(dataset, batch_size = 3, shuffle = True, num_workers = 0)

# Show a few images.

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['lcam_fish']['image_0'].size())    

    # Show the batch side by side.
    import cv2
    import numpy as np
    imgs1 = sample_batched['lcam_fish']['image_0'].numpy()
    imgs2 = sample_batched['lcam_fish']['image_1'].numpy()

    # Move the color channel to the end.
    imgs1 = np.transpose(imgs1, (0, 2, 3, 1))
    imgs2 = np.transpose(imgs2, (0, 2, 3, 1))


    img = np.concatenate((imgs1[0], imgs1[1], imgs1[2]), axis = 1)
    img = np.concatenate((img, np.concatenate((imgs2[0], imgs2[1], imgs2[2]), axis = 1)), axis = 0)
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('image transform', img)
    cv2.waitKey(0)

    if i_batch == 5:
        break

####################
# Ask for all the data from an environment.
####################
dataset = ta.create_image_dataset(env = 'ConstructionSite')

dataloader = DataLoader(dataset, batch_size = 3, shuffle = True, num_workers = 0)

# Show a few images.

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['lcam_fish']['image_0'].size())    

    # Show the batch side by side.
    import cv2
    import numpy as np
    imgs1 = sample_batched['lcam_fish']['image_0'].numpy()
    imgs2 = sample_batched['lcam_fish']['image_1'].numpy()

    # Move the color channel to the end.
    imgs1 = np.transpose(imgs1, (0, 2, 3, 1))
    imgs2 = np.transpose(imgs2, (0, 2, 3, 1))


    img = np.concatenate((imgs1[0], imgs1[1], imgs1[2]), axis = 1)
    img = np.concatenate((img, np.concatenate((imgs2[0], imgs2[1], imgs2[2]), axis = 1)), axis = 0)
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('image transform', img)
    cv2.waitKey(0)

    if i_batch == 5:
        break