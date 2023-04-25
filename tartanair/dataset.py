'''
Author: Yorai Shaoul
Date: 2023-02-05
'''

# General imports.
from multiprocessing import Pool
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from colorama import Fore, Back, Style

# Local imports.
from .tartanair_module import TartanAirModule
from .reader import TartanAirImageReader




################
# New below.
################

'''
Author: Yorai Shaoul
Date: 2023-04-10
'''

class TartanAirSlowLoader():
    # We mimic the data cacher syntax here. We do this by wrapping a torch dataloader in a DataCacherCostume class which creates the following two methods:
    # 1. load_sample() - returns one sample (batch) from the dataset.
    # 2. stop_cachers() - does nothing.
    def __init__(self, dataloader) -> None:

        # Super.
        super(TartanAirSlowLoader, self).__init__()
        
        # Save the dataloader.
        self.dataloader = dataloader

        # Iterator for the dataloader.
        self.dataloader_iter = iter(self.dataloader)

        # Create a dummy stop_cachers method.
        self.stop_cachers = lambda: None

    def load_sample(self):
        return next(self.dataloader_iter)
    

class TartanAirSlowLoaderCreator(TartanAirModule):
    '''
    The TartanAirDataset classb contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class. 
    '''
    def __init__(self, tartanair_data_root):
        # Call the parent class constructor.
        super(TartanAirSlowLoaderCreator, self).__init__(tartanair_data_root)

        # Load the dataset info.
        self.dataset = None

    def create_image_slowloader(self, 
                            env, 
                            difficulty = None, 
                            trajectory_id = None, 
                            modality = None, 
                            camera_name = None,
                            batch_size = 1,
                            shuffle = True,
                            num_workers=1):
        '''
        Create a dataset object, reading data from the TartanAir dataset, and return it.

        Args:

        env(str or list): The environment(s) to use.
        difficulty(str or list): The difficulty(s) to use. The allowed names are: 'easy', 'hard'.
        trajectory_id(str or list): The trajectory id(s) to use. If empty, then all the trajectories will be used.
        modality(str or list): The modality(ies) to use.
        camera_name(str or list): The camera name(s) to use. If the modality list does not include a form of an image (e.g. 'image', 'depth', 'seg'), then this parameter is ignored.
        '''

        # Add default values to empty inputs.
        if difficulty is None:
            difficulty = [] # Takes whatever is available.
        if trajectory_id is None:
            trajectory_id = [] # Empty list will default to all trajs down the line.
        if modality is None:
            modality = ['image', 'depth', 'seg']
        if camera_name is None:
            camera_name = ['lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_top', 'lcam_bottom', 'lcam_fish', 'lcam_equirect', 'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_top', 'rcam_bottom', 'rcam_fish', 'rcam_equirect']

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
        self.dataset = TartanAirImageDatasetForSlowLoaderObject(self.tartanair_data_root, env, difficulty, trajectory_id, modality, camera_name, num_workers)

        # We mimic the data cacher syntax here. We do this by wrapping a torch dataloader in a DataCacherCostume class which creates the following two methods:
        # 1. load_sample() - returns one sample (batch) from the dataset.
        # 2. stop_cachers() - does nothing.
        return TartanAirSlowLoader(torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))


class TartanAirImageDatasetForSlowLoaderObject(Dataset):

    def __init__(self, tartanair_data_root, 
                        envs = [], 
                        difficulties = [], 
                        trajectory_ids = [], 
                        modalities = ['image'], 
                        camera_names = ['lcam_front'],
                        num_workers = 4):

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
        super(TartanAirImageDatasetForSlowLoaderObject, self).__init__()

        # Save the parameters.
        self.tartanair_data_root = tartanair_data_root
        self.envs = envs
        self.difficulties = difficulties
        self.trajectory_ids = trajectory_ids
        self.modalities = modalities
        self.camera_names = camera_names

        # Keep track of motion requirements.
        self.motion_required = 'motion' in self.modalities
        self.pose_required = 'pose' in self.modalities

        # Remove motion and pose from the modalities list.
        if self.motion_required:
            self.modalities.remove('motion')
        if self.pose_required:
            self.modalities.remove('pose')

        # Some data reading functions.
        # Reading depth.
        self.tair_reader = TartanAirImageReader()
        self.read_depth = self.tair_reader.read_depth

        # Reading image.
        self.read_image = self.tair_reader.read_rgb

        # Reading segmentation.
        self.read_seg = self.tair_reader.read_seg

        # Reading distance images. Those store the distance to the closest object in the scene in each pixel along its ray.
        self.read_dist = self.tair_reader.read_dist

        # If imu or lidar are in the modalities list, then note that we cannot load it in this dataset.
        if 'imu' in self.modalities or 'lidar' in self.modalities:
            raise ValueError('The imu and lidar modalities are not supported in the TartanAirImageDatasetObject class.')

        # Create a mapping between img_now_gfp, img_next_gfp, motion.
        # Get all the environment names. Those are only folders.
        available_envs = os.listdir(tartanair_data_root)
        available_envs = [env for env in available_envs if os.path.isdir(os.path.join(tartanair_data_root, env))]

        # If no envs were passed, then use all of them.
        print('envs', envs)
        if not self.envs:
            self.envs = available_envs

            # Print.
            print('No environments were passed. Using all of them:')
            print("        " + "\n        ".join(self.envs))

        # Otherwise check that all the requested environments are available.
        else:
            for env in self.envs:
                assert env in available_envs, 'The environment {} is not available.'.format(env)
        
        # Check that all the requested difficulties are available in all the requested environments.
        for env in self.envs:
            # Get all the difficulty names.
            available_difficulties = os.listdir(os.path.join(tartanair_data_root, env))
            available_difficulties = [difficulty for difficulty in available_difficulties if os.path.isdir(os.path.join(tartanair_data_root, env, difficulty))]

            # Check that all the requested difficulties are available.
            for difficulty in self.difficulties:
                assert 'Data_' + difficulty in available_difficulties, 'The difficulty {} is not available in the environment {}.'.format(difficulty, env)
         

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

        pool = Pool(processes=num_workers)

        per_env_entries = pool.map(self.create_env_entries, self.envs)

        # Iterate over the environments.
        for env_entries in per_env_entries:
            self.data += env_entries

        # The number of data entries.
        self.num_data_entries = len(self.data)

        print('The dataset has {} entries.'.format(self.num_data_entries))

    def create_env_entries(self, env):
        env_entries = []

        # Iterate over difficulties.
        if self.difficulties:
            available_difficulties = [diff for diff in os.listdir(self.tartanair_data_root + '/' + env) if ('Data_' in diff) and (diff in ['Data_' + d for d in self.difficulties])]
        else:
            available_difficulties = [diff for diff in os.listdir(self.tartanair_data_root + '/' + env) if ('Data_' in diff)]

        for difficulty in available_difficulties: 
            diff_dir_gp = self.tartanair_data_root + '/' + env + '/' + difficulty

            # Check that this is a directory.
            if not os.path.isdir(diff_dir_gp):
                continue

            # Get the available trajectory ids.
            available_trajectory_ids = os.listdir(diff_dir_gp)
            available_trajectory_ids = [traj for traj in available_trajectory_ids if "P" in traj]
            
            # If no trajectory ids were passed, then use all of them.
            if len(self.trajectory_ids) == 0:
                trajectory_ids_for_env = available_trajectory_ids
            else:
                trajectory_ids_for_env = self.trajectory_ids

            # Iterate over trajectories.
            for traj_name in trajectory_ids_for_env:
                traj_dir_gp = os.path.join(diff_dir_gp, traj_name)

                # Check it it exists.
                if not os.path.isdir(traj_dir_gp):
                    print(Fore.RED + 'The trajectory {} does not exist. Skipping it.'.format(traj_dir_gp) + Style.RESET_ALL)
                    continue
                
                # Get the trajectory poses. This is a map from a camera_name to a list of poses.
                camera_name_to_motions = {}
                camera_name_to_poses = {}
                for camera_name in self.camera_names:
                    # If the camera_name is one of fish or equirct, then use the front camera motions.
                    if 'fish' in camera_name or 'equirct' in camera_name:
                        cam_side = 'lcam' if 'lcam' in camera_name else 'rcam'
                        posefile = traj_dir_gp + '/pose_{}_front.txt'.format(cam_side)

                    else:
                        posefile = traj_dir_gp + f'/pose_{camera_name}.txt'
                        
                    poselist = np.loadtxt(posefile).astype(np.float32)
                    camera_name_to_poses[camera_name] = poselist

                    if self.motion_required:
                        traj_poses   = self.pos_quats2SEs(poselist) # From xyz, xyzw format, to SE(3) format (camera in world).
                        traj_matrix  = self.pose2motion(traj_poses) # From SE(3) format, to flattened-tranformation-matrix (1x12) format.
                        traj_motions = self.SEs2ses(traj_matrix).astype(np.float32) # From flattened-tranformation-matrix (1x12) format, to relative motion (1x6) format.
                        camera_name_to_motions[camera_name] = traj_motions

                # Iterate over available frames.
                tmp_data_path = os.path.join(traj_dir_gp, self.modalities[0] + '_' + self.camera_names[0])
                num_frames_in_traj = len(os.listdir(tmp_data_path))

                ########################################
                # Memoization of some directory data.
                ########################################
                memoized_dir_data = {}
                # Iterate over camera names.
                for camera_name in self.camera_names:
                    # Iterate over modalities.
                    for modality in self.modalities:
                        # The data files global paths. Sorted.
                        if (camera_name, modality) not in memoized_dir_data:
                            # The data folders global path. One directory global path for each camera.
                            camera_dir_gps = os.path.join(traj_dir_gp, modality + '_' + camera_name)
                            # The data files global paths. Sorted.
                            data_file_gps = os.listdir(camera_dir_gps)
                            data_file_gps.sort()
                            memoized_dir_data[(camera_name, modality)] = data_file_gps

                print('         Creating entries for environment {} and difficulty {} and trajectory {}.'.format(env, difficulty, traj_name))
                for frame_id in range(num_frames_in_traj - 1): # We do not have a motion for the last frame as we do not have a next frame.
                        
                    # Create entry.
                    entry = {camera_name: {'data0' : {}, 'data1' : {}, 'motion' : None, 'poses': []} for camera_name in self.camera_names}
                    # Iterate over camera names.
                    for camera_name in self.camera_names:

                        # Add the poses to the entry.
                        entry[camera_name]['poses'].append(camera_name_to_poses[camera_name][frame_id])
                        entry[camera_name]['poses'].append(camera_name_to_poses[camera_name][frame_id + 1])

                        if self.motion_required:
                            # Start by adding the motion to the entry.
                            entry[camera_name]['motion'] = camera_name_to_motions[camera_name][frame_id]

                        # Iterate over modalities.
                        for modality in self.modalities:

                            # The data files global paths. Sorted.
                            if (camera_name, modality) not in memoized_dir_data:
                                print('Uhh, this should not happen. Could not find the data files for camera {} and modality {} in the memoized data.'.format(camera_name, modality))

                            data_file_gps = memoized_dir_data[(camera_name, modality)]
                            data0_file_gp = os.path.join(traj_dir_gp, modality + '_' + camera_name, data_file_gps[frame_id])
                            data1_file_gp = os.path.join(traj_dir_gp, modality + '_' + camera_name, data_file_gps[frame_id + 1])

                            # Check that the data files exists.
                            assert os.path.exists(data0_file_gp), 'The data file {} does not exist.'.format(data0_file_gp)
                            assert os.path.exists(data1_file_gp), 'The data file {} does not exist.'.format(data1_file_gp)

                            # Add the data file global path to the entry.
                            entry[camera_name]['data0'][modality] = data0_file_gp
                            entry[camera_name]['data1'][modality] = data1_file_gp

                    # Add the entry to the data.
                    env_entries.append(entry)

        return env_entries



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
            for modality in entry[camera_name]['data0'].keys():
                # Get the data0 and data1 global paths.
                data0_gp = entry[camera_name]['data0'][modality]
                data1_gp = entry[camera_name]['data1'][modality]

                # Read the data0 and data1.
                if 'image' in modality:
                    data0 = self.read_image(data0_gp)
                    data1 = self.read_image(data1_gp)

                    # Create the images tensor of shape (2, H, W, C).
                    rgb_key = 'rgb_' + camera_name
                    sample[rgb_key] = torch.from_numpy(np.stack([data0, data1], axis=0))

                elif 'depth' in modality:  
                    data0 = self.read_depth(data0_gp)
                    data1 = self.read_depth(data1_gp)

                    # Create the depth tensor of shape (2, H, W).
                    depth_key = 'depth_' + camera_name
                    sample[depth_key] = torch.from_numpy(np.stack([data0, data1], axis=0))

                elif 'dist' in modality:
                    data0 = self.read_dist(data0_gp)
                    data1 = self.read_dist(data1_gp)

                    # Create the distance-image tensor of shape (2, H, W).
                    dist_key = 'dist_' + camera_name
                    sample[dist_key] = torch.from_numpy(np.stack([data0, data1], axis=0))

                elif 'seg' in modality:
                    data0 = self.read_seg(data0_gp)
                    data1 = self.read_seg(data1_gp)

                    # Create the segmentation tensor of shape (2, H, W).
                    seg_key = 'seg_' + camera_name
                    sample[seg_key] = torch.from_numpy(np.stack([data0, data1], axis=0))

            if self.pose_required:
                pose_key = 'pose_' + camera_name
                sample[pose_key] = torch.from_numpy(np.stack(entry[camera_name]['poses'], axis=0))

            if self.motion_required:
                motion_key = 'motion_' + camera_name
                sample[motion_key] = torch.from_numpy(entry[camera_name]['motion'])

        # Return the sample.
        return sample
    

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


####################
# Stop new.
####################


class TartanAirDataset(TartanAirModule):
    '''
    The TartanAirDataset class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root):
        # Call the parent class constructor.
        super(TartanAirDataset, self).__init__(tartanair_data_root)

        # Load the dataset info.
        self.dataset = None

    def create_image_dataset(self, 
                             env, 
                             difficulty = None, 
                             trajectory_id = None, 
                             modality = None, 
                             camera_name = None, 
                             transform = transforms.Compose([]),
                             num_workers=1):
        '''
        Create a dataset object, reading data from the TartanAir dataset, and return it.

        Args:

        env(str or list): The environment(s) to use.
        difficulty(str or list): The difficulty(s) to use. The allowed names are: 'easy', 'hard'.
        trajectory_id(str or list): The trajectory id(s) to use. If empty, then all the trajectories will be used.
        modality(str or list): The modality(ies) to use.
        camera_name(str or list): The camera name(s) to use. If the modality list does not include a form of an image (e.g. 'image', 'depth', 'seg'), then this parameter is ignored.
        transform(torchvision.transforms): A torchvision transform object, which will be applied to the data. If out_to_tensor is True, then the transform will be applied before the data is converted to a tensor.
        '''

        # Add default values to empty inputs.
        if difficulty is None:
            difficulty = [] # Takes whatever is available.
        if trajectory_id is None:
            trajectory_id = [] # Empty list will default to all trajs down the line.
        if modality is None:
            modality = ['image', 'depth', 'seg']
        if camera_name is None:
            camera_name = ['lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_top', 'lcam_bottom', 'lcam_fish', 'lcam_equirect', 'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_top', 'rcam_bottom', 'rcam_fish', 'rcam_equirect']

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

        # If the transform is none, then create an empty transform.
        if transform is None:
            transform = transforms.Compose([])

        # Create a dataset object.
        self.dataset = TartanAirImageDatasetObject(self.tartanair_data_root, env, difficulty, trajectory_id, modality, camera_name, transform, num_workers)

        # Return the dataset object.
        return self.dataset


class TartanAirImageDatasetObject(Dataset):

    def __init__(self, tartanair_data_root, 
                        envs = [], 
                        difficulties = [], 
                        trajectory_ids = [], 
                        modalities = ['image'], 
                        camera_names = ['lcam_front'],
                        transform = None,
                        num_workers = 4):

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
        self.transform = transform

        # Alert the user that transforms may only make sense for RGB image modalities.
        if self.transform is not None and ('seg' in self.modalities or 'depth' in self.modalities):
            print('Warning: The transform parameter may only be relevant for RGB image modalities and will be applied to that only.')

        # Some data reading functions.
        # Reading depth.
        self.tair_reader = TartanAirImageReader()
        self.read_depth = self.tair_reader.read_depth

        # Reading image.
        self.read_image = self.tair_reader.read_rgb

        # Reading segmentation.
        self.read_seg = self.tair_reader.read_seg

        # Reading distance images. Those store the distance to the closest object in the scene in each pixel along its ray.
        self.read_dist = self.tair_reader.read_dist

        # If imu or lidar are in the modalities list, then note that we cannot load it in this dataset.
        if 'imu' in self.modalities or 'lidar' in self.modalities:
            raise ValueError('The imu and lidar modalities are not supported in the TartanAirImageDatasetObject class.')

        # Create a mapping between img_now_gfp, img_next_gfp, motion.
        # Get all the environment names. Those are only folders.
        available_envs = os.listdir(tartanair_data_root)
        available_envs = [env for env in available_envs if os.path.isdir(os.path.join(tartanair_data_root, env))]

        # If no envs were passed, then use all of them.
        print('envs', envs)
        if not self.envs:
            self.envs = available_envs

            # Print.
            print('No environments were passed. Using all of them:')
            print("        " + "\n        ".join(self.envs))

        # Otherwise check that all the requested environments are available.
        else:
            for env in self.envs:
                assert env in available_envs, 'The environment {} is not available.'.format(env)
        
        # Check that all the requested difficulties are available in all the requested environments.
        for env in self.envs:
            # Get all the difficulty names.
            available_difficulties = os.listdir(os.path.join(tartanair_data_root, env))
            available_difficulties = [difficulty for difficulty in available_difficulties if os.path.isdir(os.path.join(tartanair_data_root, env, difficulty))]

            # Check that all the requested difficulties are available.
            for difficulty in self.difficulties:
                assert 'Data_' + difficulty in available_difficulties, 'The difficulty {} is not available in the environment {}.'.format(difficulty, env)
         

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

        pool = Pool(processes=num_workers)

        per_env_entries = pool.map(self.create_env_entries, self.envs)

        # Iterate over the environments.
        for env_entries in per_env_entries:
            self.data += env_entries

        # The number of data entries.
        self.num_data_entries = len(self.data)

        # Image transformations.
        # self.transform = transform

        print('The dataset has {} entries.'.format(self.num_data_entries))

    def create_env_entries(self, env):
        env_entries = []

        # Iterate over difficulties.
        if self.difficulties:
            available_difficulties = [diff for diff in os.listdir(self.tartanair_data_root + '/' + env) if ('Data_' in diff) and (diff in ['Data_' + d for d in self.difficulties])]
        else:
            available_difficulties = [diff for diff in os.listdir(self.tartanair_data_root + '/' + env) if ('Data_' in diff)]

        for difficulty in available_difficulties: 
            diff_dir_gp = self.tartanair_data_root + '/' + env + '/' + difficulty

            # Check that this is a directory.
            if not os.path.isdir(diff_dir_gp):
                continue

            # Get the available trajectory ids.
            available_trajectory_ids = os.listdir(diff_dir_gp)
            available_trajectory_ids = [traj for traj in available_trajectory_ids if "P" in traj]
            
            # If no trajectory ids were passed, then use all of them.
            if len(self.trajectory_ids) == 0:
                trajectory_ids_for_env = available_trajectory_ids
            else:
                trajectory_ids_for_env = self.trajectory_ids

            # Iterate over trajectories.
            for traj_name in trajectory_ids_for_env:
                traj_dir_gp = os.path.join(diff_dir_gp, traj_name)
                
                # Get the trajectory poses. This is a map from a camera_name to a list of poses.
                camera_name_to_motions = {}
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
                    camera_name_to_motions[camera_name] = traj_motions

                # Iterate over available frames.
                tmp_data_path = os.path.join(traj_dir_gp, self.modalities[0] + '_' + self.camera_names[0])
                num_frames_in_traj = len(os.listdir(tmp_data_path))

                ########################################
                # Memoization of some directory data.
                ########################################
                memoized_dir_data = {}
                # Iterate over camera names.
                for camera_name in self.camera_names:
                    # Iterate over modalities.
                    for modality in self.modalities:
                        # The data files global paths. Sorted.
                        if (camera_name, modality) not in memoized_dir_data:
                            # The data folders global path. One directory global path for each camera.
                            camera_dir_gps = os.path.join(traj_dir_gp, modality + '_' + camera_name)
                            # The data files global paths. Sorted.
                            data_file_gps = os.listdir(camera_dir_gps)
                            data_file_gps.sort()
                            memoized_dir_data[(camera_name, modality)] = data_file_gps

                print('         Creating entries for environment {} and difficulty {} and trajectory {}.'.format(env, difficulty, traj_name))
                for frame_id in range(num_frames_in_traj - 1): # We do not have a motion for the last frame as we do not have a next frame.
                        
                    # Create entry.
                    entry = {camera_name: {'data0' : {}, 'data1' : {}, 'motion' : None} for camera_name in self.camera_names}
                    # Iterate over camera names.
                    for camera_name in self.camera_names:

                        # Start by adding the motion to the entry.
                        entry[camera_name]['motion'] = camera_name_to_motions[camera_name][frame_id]

                        # Iterate over modalities.
                        for modality in self.modalities:

                            # The data files global paths. Sorted.
                            if (camera_name, modality) not in memoized_dir_data:
                                print('Uhh, this should not happen. Could not find the data files for camera {} and modality {} in the memoized data.'.format(camera_name, modality))

                            data_file_gps = memoized_dir_data[(camera_name, modality)]
                            data0_file_gp = os.path.join(traj_dir_gp, modality + '_' + camera_name, data_file_gps[frame_id])
                            data1_file_gp = os.path.join(traj_dir_gp, modality + '_' + camera_name, data_file_gps[frame_id + 1])

                            # Check that the data files exists.
                            assert os.path.exists(data0_file_gp), 'The data file {} does not exist.'.format(data0_file_gp)
                            assert os.path.exists(data1_file_gp), 'The data file {} does not exist.'.format(data1_file_gp)

                            # Add the data file global path to the entry.
                            entry[camera_name]['data0'][modality] = data0_file_gp
                            entry[camera_name]['data1'][modality] = data1_file_gp

                    # Add the entry to the data.
                    env_entries.append(entry)

        return env_entries



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
            for modality in entry[camera_name]['data0'].keys():
                # Get the data0 and data1 global paths.
                data0_gp = entry[camera_name]['data0'][modality]
                data1_gp = entry[camera_name]['data1'][modality]

                # Read the data0 and data1.
                if 'image' in modality:
                    data0 = self.read_image(data0_gp)
                    data1 = self.read_image(data1_gp)

                    # Transform the data0 and data1.
                    if self.transform is not None:
                        data0 = self.transform(data0)
                        data1 = self.transform(data1)

                elif 'depth' in modality:  
                    data0 = self.read_depth(data0_gp)
                    data1 = self.read_depth(data1_gp)

                elif 'dist' in modality:
                    data0 = self.read_dist(data0_gp)
                    data1 = self.read_dist(data1_gp)

                elif 'seg' in modality:
                    data0 = self.read_seg(data0_gp)
                    data1 = self.read_seg(data1_gp)

                # Add the data0 and data1 to the camera sample.
                camera_sample[modality + '_0'] = data0
                camera_sample[modality + '_1'] = data1

            # Add the camera sample to the sample.
            sample[camera_name] = camera_sample

            # Add the motion to the sample.
            sample[camera_name]['motion'] = entry[camera_name]['motion']

        # Return the sample.
        return sample

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