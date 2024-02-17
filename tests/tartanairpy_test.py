'''
Author: Yorai Shaoul
Date: 2023-02-03

Test file for the TartanAir dataset toolbox.
'''

# General imports.
import glob
import os
import unittest
import sys

from colorama import Fore, Style
import cv2
import numpy as np

sys.path.append("..")
import tartanair as ta

class TartanAirTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TartanAirTest, self).__init__(*args, **kwargs)

        # Kick start the initialization.
        self.tartanair_data_root = './sample_tartanair_v2_data_root'
        self.azure_token = "?sv=2BC9w8QKvmiK7QuaBhaA%3D"
        

    def test_test(self):
        '''
        Dummy test to make sure that the testing framework is working.
        '''
        a = 'a'
        self.assertEqual(a, 'a')

    ############################
    # Test the initialization. 
    ############################
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
        
    ############################
    # Test download.     
    ############################
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

    ############################
    # Test the dataloader.
    ############################
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

    ############################
    # Test customization.
    ############################
    def test_customization(self):

        # Initialize tartanair.
        ta.init(self.tartanair_data_root, self.azure_token)

        # Check that we have the data.
        if os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000',  'image_lcam_front')) and \
            os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_left')) and \
            os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_right')) and \
            os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_back')) and \
            os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_top')) and \
            os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000', 'image_lcam_bottom')):
            print(Fore.GREEN + "..Data found." + Style.RESET_ALL)
        else:
            print(Fore.RED + "..Data not found. Downloading." + Style.RESET_ALL)
            # Request the download.
            envs = ["ArchVizTinyHouseDayExposure"]

            ta.download(env = envs, difficulty = ['hard'], trajectory_id = ["P000"],  modality = ['image', 'pose'],  camera_name = ['lcam_front' ,'lcam_left' ,'lcam_right' ,'lcam_back' ,'lcam_top' ,'lcam_bottom'])

        env = 'ArchVizTinyHouseDayExposure'
        difficulty = ['hard']
        traj_name = 'P000'
        downloaded_data_dir_path = os.path.join(self.tartanair_data_root, env, 'Data_' + difficulty[0], traj_name)

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

        # Test on GPU as well, if it is available.
        import torch
        if torch.cuda.device_count() > 0:
                
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

    ############################
    # Test Data Listing.
    ############################

    def test_env_listing(self):
            
        # Initialize tartanair.
        ta.init(self.tartanair_data_root, self.azure_token)

        # Check that we have the data.
        if os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000',  'image_lcam_front')):
            print(Fore.GREEN + "..Data found." + Style.RESET_ALL)
        else:
            print(Fore.RED + "..Data not found. Downloading." + Style.RESET_ALL)
            # Request the download.
            envs = ["ArchVizTinyHouseDayExposure"]

            ta.download(env = envs, difficulty = ['hard'], trajectory_id = ["P000"],  modality = ['image', 'pose'],  camera_name = ['lcam_front'])

        # Get the data listing.
        available_envs = ta.list_envs()
        assert "ArchVizTinyHouseDayExposure" in available_envs['local']

        print(Fore.GREEN + "..Env listing OK." + Style.RESET_ALL)

    ############################
    # Test Evaluation.
    ############################
    def test_evaluation(self):

        # Initialize tartanair.
        ta.init(self.tartanair_data_root, self.azure_token)

        # Check that we have the data.
        if os.path.exists(os.path.join(self.tartanair_data_root, 'ArchVizTinyHouseDayExposure', 'Data_hard', 'P000',  'pose_lcam_front.txt')):
            print(Fore.GREEN + "..Data found." + Style.RESET_ALL)

        else:
            print(Fore.RED + "..Data not found. Downloading." + Style.RESET_ALL)
            # Request the download.
            envs = ["ArchVizTinyHouseDayExposure"]

            ta.download(env = envs, difficulty = ['hard'], trajectory_id = ["P000"],  modality = ['pose'],  camera_name = ['lcam_front'])


        # Create an example trajectory. This is a noisy version of the ground truth trajectory.
        env = 'ArchVizTinyHouseDayExposure'
        difficulty = 'hard'
        trajectory_id = 'P000'
        camera_name = 'lcam_front'
        gt_traj = ta.get_traj_np(env, difficulty, trajectory_id, camera_name)
        est_traj = np.zeros_like(gt_traj)
        est_traj[:, :3] = gt_traj[:, :3] + np.random.normal(0, 0.2, gt_traj[:, :3].shape)  
        est_traj[:, 3:] = gt_traj[:, 3:] + np.random.normal(0, 0.01, gt_traj[:, 3:].shape)

        # Get the evaluation results.
        plot_out_path = "evaluator_example.png"
        results0 = ta.evaluate_traj(est_traj, env = env, 
                                   difficulty = difficulty, 
                                   trajectory_id = trajectory_id, 
                                   camera_name = camera_name, 
                                   enforce_length = True, 
                                   plot = True, 
                                   plot_out_path = plot_out_path, 
                                   do_scale = True, 
                                   do_align = True)

        # Optionally pass the ground truth trajectory directly to the evaluation function.
        results1 = ta.evaluate_traj(est_traj, gt_traj = gt_traj, enforce_length = True, plot = True, plot_out_path = plot_out_path, do_scale = True, do_align = True)

        # Check that the results are the same.
        assert results0['ate'] == results1['ate']

        print(Fore.GREEN + "..Evaluation OK." + Style.RESET_ALL)

if __name__ == '__main__':
    unittest.main()
