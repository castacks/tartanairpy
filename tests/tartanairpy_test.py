'''
Author: Yorai Shaoul
Date: 2023-02-03

Test file for the TartanAir dataset toolbox.
'''

# General imports.
import os
import unittest
import sys

from colorama import Fore, Style
import cv2
import numpy as np

sys.path.append("..")
import tartanair as ta

class TestTartanAir(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTartanAir, self).__init__(*args, **kwargs)

        # Kick start the initialization.
        self.tartanair_data_root = './sample_tartanair_v2_data_root'
        self.azure_token = "?sv=2021-10-04&st=-31T14%3A42%3A13Z&se=2023-07-01T14%3A42%3A00Z&sr=c&sp=rl&sig=IoZEVe1B5kQuZI5WzDSdGqiW%2BC9w8QKvmiK7QuaBhaA%3D"
        

    def test_test(self):
        '''
        Dummy test to make sure that the testing framework is working.
        '''
        a = 'a'
        self.assertEqual(a, 'a')

    def test_init(self):
        '''
        Test the initialization of the TartanAir toolbox.
        '''
        print(Fore.GREEN + "Testing initialization." + Style.RESET_ALL)

        tartanair_data_root = './sample_tartanair_v2_data_root'
        success = ta.init(tartanair_data_root, self.azure_token)

        # Check that the initialization was successful.
        self.assertEqual(success, True)
        print(Fore.GREEN + "..Initialization OK." + Style.RESET_ALL)
        

    def test_download(self):
        '''
        Test the download of a single trajectory.
        '''
        print(Fore.GREEN + "Testing download." + Style.RESET_ALL)

        # Initialize tartanair.
        ta.init(self.tartanair_data_root, self.azure_token)

        # Request the download.
        envs = ["ArchVizTinyHouseDayExposure"]
        ta.download(env = envs, difficulty = ['hard'], trajectory_id = ["P000"],  modality = ['image', 'pose'],  camera_name = ['lcam_front'])

        # Check that the download was successful.
        self.assertEqual(os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_front')), True)

        print(Fore.GREEN + "..Download OK." + Style.RESET_ALL)

    def test_dataloader(self):
        '''
        Test the dataloader.
        '''
        print(Fore.GREEN + "Testing dataloader." + Style.RESET_ALL)

        # Initialize tartanair.
        ta.init(self.tartanair_data_root, self.azure_token)

        # Check that we have the data.
        if os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_front')):
            print(Fore.GREEN + "..Data found." + Style.RESET_ALL)
        else:
            print(Fore.RED + "..Data not found. Downloading." + Style.RESET_ALL)
            # Request the download.
            envs = ["ArchVizTinyHouseDayExposure"]
            ta.download(env = envs, difficulty = ['hard'], trajectory_id = ["P000"],  modality = ['image', 'pose'],  camera_name = ['lcam_front'])

        envs = ["ArchVizTinyHouseDayExposure"]
        camnames = ['lcam_front']
        difficulties = ['hard']
        trajectory_ids = ['P000']
        modalities = ['image', 'pose']

        new_image_shape_hw = [640, 640] # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
        subset_framenum = 364 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
        frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
        seq_length = {'image': 2, 'pose': 2,} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
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
        
        # Destroy windows.
        cv2.destroyAllWindows()

        print(Fore.GREEN + "..Dataloader OK." + Style.RESET_ALL)


if __name__ == '__main__':
    unittest.main()
