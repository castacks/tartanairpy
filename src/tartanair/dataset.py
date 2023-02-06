'''
Author: Yorai Shaoul
Date: 2023-02-05
'''

# General imports.
import os
import sys
import time
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# Local imports.
from tartanair.tartanair_module import TartanAirModule
from tartanair.reader import TartanAirImageReader

class TartanAirDataset(TartanAirModule):
    '''
    The TartanAirDataset class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root):
        # Call the parent class constructor.
        super(TartanAirDataset, self).__init__(tartanair_data_root)

        self.modality_names = ['image', 'depth', 'seg', 'imu', 'lidar']

        self.env_names = ['DesertGasStation', 'Forest', 'OldIndustrialCity', 'ApocalypticCity']

        self.difficulty_names = ['easy', 'hard']

        # Load the token from a text file.
        # TODO(yoraish): this will be changed and hardcoded.
        with open('/home/yoraish/azure_token.txt', 'r') as f:
            self.azure_token = f.read()

        # Load the dataset info.
        self.dataset = None

    def create_image_dataset(self, env, difficulty, trajectory_id, modality, camera_name):
        '''
        Create a dataset object, reading data from the TartanAir dataset, and return it.
        '''

        # Convert all inputs to lists.
        if type(env) is not list:
            env = [env]
        if type(difficulty) is not list:
            difficulty = [difficulty]
        if type(trajectory_id) is not list:
            trajectory_id = [trajectory_id]
        if type(modality) is not list:
            modality = [modality]
        if type(camera_name) is not list:
            camera_name = [camera_name]

        # Create a dataset object.
        self.dataset = TartanAirImageDatasetObject(self.tartanair_data_root, env, difficulty, trajectory_id, modality, camera_name)

        # Return the dataset object.
        return self.dataset


class TartanAirImageDatasetObject(Dataset):

    def __init__(self, tartanair_data_root, 
                        envs, 
                        difficulties = ['easy', 'hard'], 
                        trajectory_ids = [], 
                        modalities = ['image'], 
                        camera_names = ['lcam_front']):

        '''
        The TartanAirDatasetObject class implements a PyTorch Dataset object, which can be used to read data from the TartanAir dataset.

        Args:
        
        tartanair_data_root(str): The root directory of the TartanAir dataset.
        envs(list): A list of the environments to use. 
        difficulties(list): A list of the difficulties to use. The allowed names are: 'easy', 'hard'.
        trajectory_id(list): A list of the trajectory ids to use. If empty, then all the trajectories will be used.
        modalities(list): A list of the modalities to use. The allowed names are: 'image', 'depth', 'seg', 'imu', 'lidar'.
        camera_name(list): A list of the camera names to use. If the modality list does not include a form of an image (e.g. 'image', 'depth', 'seg'), then this parameter is ignored. 
        '''

        # Call the parent class constructor.
        super(TartanAirImageDatasetObject, self).__init__()

        # Save the parameters.
        self.tartanair_data_root = tartanair_data_root
        self.envs = envs
        self.difficulties = difficulties
        self.trajectory_ids = trajectory_ids
        self.modalities = modalities
        self.camera_names = camera_names

        # Some data reading functions.
        # Reading depth.
        self.read_depth = TartanAirImageReader().read_depth

        # Reading image.
        self.read_image = TartanAirImageReader().read_bgr

        # If imu or lidar are in the modalities list, then note that we cannot load it in this dataset.
        if 'imu' in self.modalities or 'lidar' in self.modalities:
            raise ValueError('The imu and lidar modalities are not supported in the TartanAirImageDatasetObject class.')

        # Create a mapping between img_now_gfp, img_next_gfp, motion.
        # Get all the environment names.
        available_envs = os.listdir(tartanair_data_root)

        # Check that all the requested environments are available.
        for env in self.envs:
            assert env in available_envs, 'The environment {} is not available.'.format(env)

        # Get all the difficulty names.
        available_difficulties = os.listdir(os.path.join(tartanair_data_root, envs[0]))

        # Check that all the requested difficulties are available.
        for difficulty in self.difficulties:
            assert 'Data_' + difficulty in available_difficulties, 'The difficulty {} is not available.'.format(difficulty)

        # The object that will keep all of the data. We will later concatenate all of the data. It takes the form of 
        # [{camera0: 
        #           {data0: {modality0: fp, modality1: fp, ...}, 
        #            data1: {modality0: fp, modality1: fp, ...}, 
        #            motion: motion}, ...}
        # {camera1:
        #          {data0: {modality0: fp, modality1: fp, ...},
        #           data1: {modality0: fp, modality1: fp, ...},
        #           motion: motion}, ...}, 
        # ...]

        self.data = []

        # The trajectory names.
        for env in self.envs:
            
            # Iterate over difficulties.
            for difficulty in os.listdir(tartanair_data_root + '/' + env):
                diff_dir_gp = tartanair_data_root + '/' + env + '/' + difficulty

                # Iterate over trajectories.
                for traj_name in self.trajectory_ids:
                    traj_dir_gp = os.path.join(diff_dir_gp, traj_name)

                    # Get the trajectory poses. This is a map from a camera_name to a list of poses.
                    camera_name_to_poses = {}
                    for camera_name in self.camera_names:
                        # If the camera_name is one of fish or equirct, then use the front camera motions.
                        if 'fish' in camera_name or 'equirct' in camera_name:
                            cam_side = 'lcam' if 'lcam' in camera_name else 'rcam'
                            posefile = traj_dir_gp + '/pose_{}_front.txt'.format(cam_side)
                        else:
                            posefile = traj_dir_gp + f'/pose_{camera_name}.txt'
                        poselist = np.loadtxt(posefile).astype(np.float32)
                        
                        traj_poses   = self.pos_quats2SEs(poselist) # From xyz, xyzw format, to SE(3) format (camera in world).
                        traj_matrix  = self.pose2motion(traj_poses) # From SE(3) format, to flattened-tranformation-matrix (1x12) format.
                        traj_motions = self.SEs2ses(traj_matrix).astype(np.float32) # From flattened-tranformation-matrix (1x12) format, to relative motion (1x6) format.

                    # Iterate over available frames.
                    tmp_data_path = os.path.join(traj_dir_gp, self.modalities[0] + '_' + self.camera_names[0])
                    num_frames_in_traj = len(os.listdir(tmp_data_path))

                    for frame_id in range(num_frames_in_traj - 1): # We do not have a motion for the last frame as we do not have a next frame.
                            
                        # Create entry.
                        entry = {camera_name: {'data0' : {}, 'data1' : {}, 'motion' : None} for camera_name in self.camera_names}

                        # Iterate over camera names.
                        for camera_name in self.camera_names:

                            # Start by adding the motion to the entry.
                            entry[camera_name]['motion'] = traj_motions[frame_id]

                            # Iterate over modalities.
                            for modality in self.modalities:
                                
                                # The data folders global path. One directory global path for each camera.
                                camera_dir_gps = os.path.join(traj_dir_gp, modality + '_' + camera_name)

                                # The data files global paths. Sorted.
                                data_file_gps = os.listdir(camera_dir_gps)
                                data_file_gps.sort()
                                data0_file_gp = os.path.join(camera_dir_gps, data_file_gps[frame_id])
                                data1_file_gp = os.path.join(camera_dir_gps, data_file_gps[frame_id + 1])


                                # Check that the data files exists.
                                assert os.path.exists(data0_file_gp), 'The data file {} does not exist.'.format(data0_file_gp)
                                assert os.path.exists(data1_file_gp), 'The data file {} does not exist.'.format(data1_file_gp)

                                # Add the data file global path to the entry.
                                entry[camera_name]['data0'][modality] = data0_file_gp
                                entry[camera_name]['data1'][modality] = data1_file_gp

                        # Add the entry to the data.
                        self.data.append(entry)

        # The number of data entries.
        self.num_data_entries = len(self.data)

        # Image transformations.
        # self.transform = transform

        print('The dataset has {} entries.'.format(self.num_data_entries))

    def __len__(self):
        return self.num_data_entries

    def __getitem__(self, index):
        # Get the entry.
        entry = self.data[index]

        # Create the sample.
        sample = {}

        # Iterate over camera names.
        for camera_name in self.camera_names:
            # Create the camera sample.
            camera_sample = {}

            # Iterate over modalities.
            for modality in self.modalities:
                # Get the data0 and data1 global paths.
                data0_gp = entry[camera_name]['data0'][modality]
                data1_gp = entry[camera_name]['data1'][modality]

                # Read the data0 and data1.
                data0 = self.read_image(data0_gp)
                data1 = self.read_image(data1_gp)

                # Add the data0 and data1 to the camera sample.
                camera_sample[modality + '_0'] = data0
                camera_sample[modality + '_1'] = data1

            # Add the camera sample to the sample.
            sample[camera_name] = camera_sample

        # Add the motion to the sample.
        sample['motion'] = entry[camera_name]['motion']

        # Return the sample.
        return sample

    # def __getitem__(self, idx):
    #     # If initialized, don't. Else, do.
    #     if not self.initialized:
    #         self.initialize()
    #         self.initialized = True

    #     ####################
    #     # Get the sample. These are paths to images and depth images.
    #     ####################
    #     sample = self.data[idx]

    #     ####################
    #     # First, choose a rotation sampler for the fisheye image within the image cube.
    #     ####################
    #     fish_sampler = random.choice(self.fisheye_samplers)
        
    #     ####################
    #     # Create the image dicts.
    #     ####################
    #     img0_dict = {"front": self.read_image(sample['img0']['front']),
    #                 "back": self.read_image(sample['img0']['back']),
    #                 "right": self.read_image(sample['img0']['right']),
    #                 "left": self.read_image(sample['img0']['left']),
    #                 "top": self.read_image(sample['img0']['top']),
    #                 "bottom": self.read_image(sample['img0']['bottom'])
    #                 }
    #     img1_dict = {"front": self.read_image(sample['img1']['front']),
    #                 "back": self.read_image(sample['img1']['back']),
    #                 "right": self.read_image(sample['img1']['right']),
    #                 "left": self.read_image(sample['img1']['left']),
    #                 "top": self.read_image(sample['img1']['top']),
    #                 "bottom": self.read_image(sample['img1']['bottom'])
    #                 }

    #     ####################
    #     # Create the first and second fisheye images.
    #     ####################
    #     fish0, valid_mask_fish0 = fish_sampler(img0_dict, interpolation = 'linear') # (H, W, 3) uint8.
    #     fish1, valid_mask_fish1 = fish_sampler(img1_dict, interpolation = 'linear')

    #     # Create the depth dict.
    #     depth_dict = {"front": self.read_depth(sample['depth0']['front']),
    #                 "back": self.read_depth(sample['depth0']['back']),
    #                 "right": self.read_depth(sample['depth0']['right']),
    #                 "left": self.read_depth(sample['depth0']['left']),
    #                 "top": self.read_depth(sample['depth0']['top']),
    #                 "bottom": self.read_depth(sample['depth0']['bottom'])
    #                 }

    #     ####################
    #     # Create a depth image for the first fisheye image.
    #     ####################
    #     depth0, valid_mask_depth = fish_sampler.blend_interpolation(depth_dict, self.blend_func)

    #     ####################
    #     # Adapt the motion to the rotated fisheye image.
    #     # TODO(yoraish): this rotation is in the image frame. Meaning z is forward, x is right, y is down. We would like to convert it to NED, where z is down, x is forward, y is right in order to adapt the motion.
    #     ####################
    #     R_cube_fish = fish_sampler.R_raw_fisheye.cpu().numpy() # This is in RDF (OpenCV camera).
    #     X_cube_fish = np.eye(4)
    #     X_cube_fish[:3, :3] = R_cube_fish
    #     R_fish_cube = np.linalg.inv(R_cube_fish)
    #     X_fish_cube = np.eye(4)
    #     X_fish_cube[:3, :3] = R_fish_cube

    #     # The cube in the base (b), this transforms NED to RDF camera. 
    #     X_NED_cam = np.eye(4)
    #     X_NED_cam[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    #     # Its inverse.
    #     X_cam_NED = np.linalg.inv(X_NED_cam)

    #     # Convert the motion to SE(3).
    #     X_b0_b1 = se2SE(sample['motion'])

    #     # The transform between the base (b) and the rotated base (bb).
    #     X_b_bb = X_NED_cam @ X_cube_fish @ X_cam_NED

    #     # The transform between the rotated base (bb) and the base (b).
    #     X_bb_b = np.linalg.inv(X_b_bb)

    #     # The NED transformation between the rotated bases.
    #     X_bb0_bb1 = X_bb_b @ X_b0_b1 @ X_b_bb

    #     # Back to xyz, rotvec.
    #     fish_ned_motion = SE2se(X_bb0_bb1)

    #     ####################
    #     # Create the fisheye flow image.
    #     ####################
    #     flow_fish = flow_from_depth_motion(depth0, fish_ned_motion, self.fish_camera_model) # Output is np array (H, W, 2).

    #     # Show.
    #     if self.visualize:
    #         print(Fore.GREEN + 'Showing the fisheye image and flow.' + Style.RESET_ALL)
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         flow = self.visualizer.visflow(flow_fish, add_arrows=True)
    #         # Apply the mask.
    #         flow[valid_mask_fish0 == 0] *= 0
    #         plt.imshow(flow)
    #         plt.show()

    #     ####################
    #     # Form the sample.
    #     ####################
    #     res = {'fish0': fish0, 'fish1': fish1, 'gt_fish_flow': flow_fish, 'fish_valid_mask': valid_mask_fish0} 

    #     if idx > len(self.data):
    #         raise StopIteration 

    #     # Add the motion if it is available.
    #     if fish_ned_motion is None:
    #         return res
    #     else:
    #         res['motion'] = fish_ned_motion
    #         return res


    ########################
    # Utility geometry functions.
    ########################

    def pos_quats2SEs(self, quat_datas):
        data_len = quat_datas.shape[0]
        SEs = np.zeros((data_len,12))
        for i_data in range(0,data_len):
            SE = self.pos_quat2SE(quat_datas[i_data,:])
            SEs[i_data,:] = SE
        return SEs

    def pos_quat2SE(self, quat_data):
        SO = Rotation.from_quat(quat_data[3:7]).as_matrix()
        SE = np.matrix(np.eye(4))
        SE[0:3,0:3] = np.matrix(SO)
        SE[0:3,3]   = np.matrix(quat_data[0:3]).T
        SE = np.array(SE[0:3,:]).reshape(1,12)
        return SE

    def pose2motion(self, data, skip=0):
        """ Converts a sequence of poses to a sequence of motions.

        Args:
            data (list): list of transformation matrices in the form of a 4x4 numpy array. Those are X_cam_in_world.
            skip (int, optional): If to skip poses, then how many. Defaults to 0.

        Returns:
            list: each element is a 1x12 array representing the motion from the current pose to the next pose. These are the top three rows of the relative transformation matrix, flattened.
        """
        data_size = data.shape[0]
        all_motion = np.zeros((data_size-1,12))
        for i in range(0,data_size-1-skip):
            pose_curr = self.line2mat(data[i,:])
            pose_next = self.line2mat(data[i+1+skip,:])
            motion = pose_curr.I*pose_next
            motion_line = np.array(motion[0:3,:]).reshape(1,12)
            all_motion[i,:] = motion_line
        return all_motion

    def line2mat(self, line_data):
        mat = np.eye(4)
        mat[0:3,:] = line_data.reshape(3,4)
        return np.matrix(mat)

    def SE2se(self, SE_data):
        result = np.zeros((6))
        result[0:3] = np.array(SE_data[0:3,3].T)
        result[3:6] = self.SO2so(SE_data[0:3,0:3]).T
        return result

    def SO2so(self, SO_data):
        return Rotation.from_matrix(SO_data).as_rotvec()
    
    def SEs2ses(self, motion_data):
        data_size = motion_data.shape[0]
        ses = np.zeros((data_size,6))
        for i in range(0,data_size):
            SE = np.matrix(np.eye(4))
            SE[0:3,:] = motion_data[i,:].reshape(3,4)
            ses[i,:] = self.SE2se(SE)
        return ses