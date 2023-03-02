'''
Author: Yorai Shaoul
Date: 2023-03-01

This file contains the iterator class, which provides an iterator over the TartanAir dataset. A set of environments, difficulties, and trajectories can be specified, and the iterator will iterate over the specified data. Otherwise, the iterator will iterate over all available data in the TartanAir data root.
'''
# General imports.
import os
from scipy.spatial.transform import Rotation
from colorama import Fore, Style
import numpy as np

# Local imports.
from .tartanair_module import TartanAirModule
from .reader import TartanAirImageReader

class TartanAirIterator(TartanAirModule):
    def __init__(self, tartanair_data_root):
        '''
        Iterate over the TartanAir dataset. This is only valid for images at this point. So imu and lidar are not supported.
        '''
        super().__init__(tartanair_data_root)
        self.tartanair_data_root = tartanair_data_root

    def get_iterator(self, env = None, difficulty = None, trajectory_id = None, modality = None, camera_name = None):
        ###############################
        # Process the inputs.
        ###############################
        # Add default values to empty inputs.
        if difficulty is None:
            difficulty = ['easy', 'hard']
            print(Fore.RED + 'Warning: Difficulty is not specified. Defaulting to difficulty = ' +  ', '.join(difficulty) + Style.RESET_ALL)
        if trajectory_id is None:
            trajectory_id = [] # Empty list will default to all trajs down the line.
            print(Fore.RED + 'Warning: Trajectory id is not specified. Defaulting to all available trajectories.' + Style.RESET_ALL)
        if modality is None:
            modality = ['image', 'depth', 'seg']
            print(Fore.RED + 'Warning: Modality is not specified. Defaulting to modality = ' +  ', '.join(modality) + Style.RESET_ALL)
        if camera_name is None:
            camera_name = ['lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_top', 'lcam_bottom', 'lcam_fish', 'lcam_equirect', 'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_top', 'rcam_bottom', 'rcam_fish', 'rcam_equirect']
            print(Fore.RED + 'Warning: Camera name is not specified. Defaulting to camera_name = ' +  ', '.join(camera_name) + Style.RESET_ALL)

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

        # Convert difficulties to format.
        difficulty = ["Data_{}".format(difficulty) for difficulty in difficulty]

        # Save the parameters.
        self.envs = env
        self.difficulties = difficulty
        self.trajectory_ids = trajectory_id
        self.modalities = modality
        self.camera_names = camera_name

        ###############################
        # Some data reading functions.
        ###############################
        # Reading depth.
        self.tair_reader = TartanAirImageReader()
        self.read_depth = self.tair_reader.read_depth

        # Reading image.
        self.read_image = self.tair_reader.read_rgb

        # Reading segmentation.
        self.read_seg = self.tair_reader.read_seg

        # Reading distance images. Those store the distance to the closest object in the scene in each pixel along its ray.
        self.read_dist = self.tair_reader.read_dist

        # If imu is in the modalities list, then note that we cannot load it in this dataset.
        if 'imu' in self.modalities or 'lidar' in self.modalities:
            raise ValueError('The imu and lidar modalities are not supported in this iterator yet.')

        ###############################
        # Create a mapping between images and motions.
        ###############################
        # Get all the environment names. Those are only folders.
        available_envs = os.listdir(self.tartanair_data_root)
        available_envs = [env for env in available_envs if os.path.isdir(os.path.join(self.tartanair_data_root, env))]

        # If no envs were passed, then use all of them.
        print('envs', self.envs)
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
            available_difficulties = os.listdir(os.path.join( self.tartanair_data_root, env))
            available_difficulties = [difficulty for difficulty in available_difficulties if os.path.isdir(os.path.join( self.tartanair_data_root, env, difficulty))]

            # Check that all the requested difficulties are available.
            for difficulty in self.difficulties:
                assert difficulty in available_difficulties, 'The difficulty {} is not available in the environment {}.'.format(difficulty, env)
         

        # The object that will keep all of the data. We will later concatenate all of the data. It takes the form of 
        # [{camera0: 
        #           {data: {modality0: fp, modality1: fp, ...}, 
        #            motion: motion}, ...}
        # {camera1:
        #          {data: {modality0: fp, modality1: fp, ...},
        #           motion: motion}, ...}, 
        # ...]

        self.data = []

        ###############################
        # Iterate over environments.
        ###############################
        for env in self.envs:
            
            ###############################
            # Iterate over difficulties.
            ###############################
            for difficulty in os.listdir(self.tartanair_data_root + '/' + env):
                if difficulty not in self.difficulties:
                    continue
                diff_dir_gp = self.tartanair_data_root + '/' + env + '/' + difficulty

                # Check that this is a directory.
                if not os.path.isdir(diff_dir_gp):
                    continue

                # Get the available trajectory ids.
                available_trajectory_ids = os.listdir(diff_dir_gp)

                # If no trajectory ids were passed, then use all of them.
                if len(self.trajectory_ids) == 0:
                    trajectory_ids_for_env = available_trajectory_ids
                else:
                    trajectory_ids_for_env = self.trajectory_ids

                ###############################
                # Iterate over trajectories.
                ###############################
                for traj_name in trajectory_ids_for_env:
                    traj_dir_gp = os.path.join(diff_dir_gp, traj_name)


                    ###############################
                    # Iterate over cameras.
                    ###############################
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

                    # Memoization of some directory data.
                    memoized_dir_data = {}

                    ###############################
                    # Iterate over frames.
                    ###############################
                    for frame_id in range(num_frames_in_traj - 1): # We do not have a motion for the last frame as we do not have a next frame.
                        
                        ###############################
                        # Create entry.
                        ###############################
                        entry = {camera_name: {'data' : {}, 'motion' : None} for camera_name in self.camera_names}

                        ###############################
                        # Iterate over camera names.
                        ###############################
                        for camera_name in self.camera_names:
                            
                            ###############################
                            # Start by adding the motion to the entry.
                            ###############################
                            entry[camera_name]['motion'] = camera_name_to_motions[camera_name][frame_id]

                            ###############################
                            # Iterate over modalities.
                            ###############################
                            for modality in self.modalities:

                                # Handle lidar separately.
                                # TODO(yoraish).

                                # The data files global paths. Sorted.
                                if (camera_name, modality) not in memoized_dir_data:

                                    # The data folders global path. One directory global path for each camera.
                                    camera_dir_gps = os.path.join(traj_dir_gp, modality + '_' + camera_name)
                                    
                                    # The data files global paths. Sorted.
                                    data_file_gps = os.listdir(camera_dir_gps)
                                    data_file_gps.sort()
                                    memoized_dir_data[(camera_name, modality)] = data_file_gps

                                data_file_gps = memoized_dir_data[(camera_name, modality)]
                                data0_file_gp = os.path.join(traj_dir_gp, modality + '_' + camera_name, data_file_gps[frame_id])

                                # Check that the data files exists.
                                assert os.path.exists(data0_file_gp), 'The data file {} does not exist.'.format(data0_file_gp)

                                # Add the data file global path to the entry.
                                entry[camera_name]['data'][modality] = data0_file_gp

                        # Add the entry to the data.
                        self.data.append(entry)

        # The number of data entries.
        self.num_data_entries = len(self.data)

        print('The iterator has {} entries.'.format(self.num_data_entries))

        self.num_data_entries

        ###############################
        # Iterate over data entries.
        ###############################
        index = 0
        for index in range(self.num_data_entries):

            # Get the entry.
            entry = self.data[index]

            # Create the sample.
            sample = {}

            # Iterate over camera names.
            for camera_name in self.camera_names:
                # Create the camera sample.
                camera_sample = {}

                # Iterate over modalities.
                for modality in entry[camera_name]['data'].keys():
                    # Get the data global paths.
                    data0_gp = entry[camera_name]['data'][modality]

                    # Read the data0 and data1.
                    if 'image' in modality:
                        data0 = self.read_image(data0_gp)

                    elif 'depth' in modality:  
                        data0 = self.read_depth(data0_gp)

                    elif 'dist' in modality:
                        data0 = self.read_dist(data0_gp)

                    elif 'seg' in modality:
                        data0 = self.read_seg(data0_gp)

                    # Add the data0 and data1 to the camera sample.
                    camera_sample[modality] = data0

                # Add the camera sample to the sample.
                sample[camera_name] = camera_sample

                # Add the motion to the sample.
                sample[camera_name]['motion'] = entry[camera_name]['motion']

            # Return the sample.
            yield sample


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