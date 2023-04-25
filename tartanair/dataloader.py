'''
Author: Yorai Shaoul, Wenshan Wang
Date: 2023-03-27

This class provides an interface to the Data-Cacher module. The data cacher is an efficient two-stage loader that allows for continuous training: parallelizing loading data and serving mini-batches.

Output batches will be of the form:

'''

# TODO(yoraish): there is a notation discrepancy between 'rgb' and 'image'. Should fix this and probably stick with 'image' as this is the name used in the dataset. 
# TODO(yoraish): support naming between 'imu_acc' and 'imu_gyro' etc. Currently only 'imu' is supported and is mapped to 'imu_acc'.

# General imports.
import os
from colorama import Fore, Back, Style


# Local imports.
from .tartanair_module import TartanAirModule
from .data_cacher.MultiDatasets import MultiDatasets

class TartanAirDataLoader(TartanAirModule):
    '''
    The TartanAirDataset class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root):
        # Call the parent class constructor.
        super(TartanAirDataLoader, self).__init__(tartanair_data_root)

        # Remap some modality names to support naming inconsistencies.
        self.modality_name_remaps = {
            'rgb': 'rgb',
            'image': 'rgb',
            'imu': 'imu_acc',
            'imu_acc': 'imu_acc',
            'depth': 'depth',
            'lidar': 'lidar',
            'pose': 'pose',
            'seg': 'seg',
            'flow': 'flow'
        }


        self.modality_default_seq_length = {
            'rgb': 1,
            'depth': 1,
            'seg': 1,
            'flow': 1,
            'pose': 1,
            'imu_acc': 10,
            'lidar': 1,
        }

        self.modality_default_cacher_size = {
            'rgb': [640, 640],
            'depth': [640, 640],
            'seg': [640, 640],
            'flow': [640, 640],
            'pose': [7],
            'imu_acc': [3],
            'lidar': [3],
        }

    def get_data_cacher(self, 
                        env, 
                        difficulty = None, 
                        trajectory_id = None, 
                        modality = None, 
                        camera_name = None, 
                        new_image_shape_hw = [640, 640],
                        subset_framenum = 56, # <--- Note in the docs that this is an upper bound on the batch size.
                        seq_length = 1, # This can also be a dictionary, mapping each modality name to a sequence length.
                        seq_stride = 1,
                        frame_skip = 0,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False, 
                        verbose=False,):

        '''
        Create a dataloader object, reading data from the TartanAir dataset, and return it. Note that under the hood a very powerful data cacher is used, which allows for efficient data loading and mini-batch serving. This method effectively creates a config file to this data cacher, and returns an iterator object that can be used to load data. This method effectively populates a config dictionary and uses it to create a dataloader object.
        '''

        # Add default values to empty inputs.
        if not difficulty:
            difficulty = []
        if not trajectory_id:
            trajectory_id = [] # Empty list will default to all trajs down the line.
        if not modality:
            print("WARNING: No modality was specified. Defaulting to _some_ modalities: ('image', 'depth', 'seg')")
            modality = ['image', 'depth', 'seg']
        if not camera_name:
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

        # Allow to take in a dictionary for the modality sequence length.
        # If an integer is given, then use it for all modalities.
        if type(seq_length) is int:
            seq_length = {mod: seq_length for mod in modality}
        elif type(seq_length) is dict:
            # Remap modality names.
            seq_length = {self.modality_name_remaps[mod]: seq_length[mod] for mod in seq_length}
            for mod in modality:
                mod = self.modality_name_remaps[mod]
                if mod not in seq_length:
                    print("WARNING: No sequence length was given for modality {}. Defaulting to {}.".format(mod, self.modality_default_seq_length[mod]))
                    seq_length[mod] = self.modality_default_seq_length[mod]
        else:
            raise ValueError("seq_length must be an integer or a dictionary.")

        config = {}
        config['task'] = 'tartanair'
        config['transform_data_augment'] = True # Unused.
        config['transform_flow_norm_factor'] = 1.0 # Unused.
        config['transform_uncertainty'] = True # Unused.
        config['transform_input_size'] = new_image_shape_hw # Unused.
        config['dataset_frame_skip'] = frame_skip
        config['dataset_seq_stride'] = seq_stride
        config['data'] = {}

        # Create a composed data specification text file for this data cacher. This is saved with a common name, ta_data_spec.txt, and is stored in the root directory of the TartanAir dataset. The contents are a composition of all the data specifications for each environment, difficulty, and trajectory.
        data_spec_fpath = 'ta_data_spec.txt'
        if os.path.isfile(data_spec_fpath):
            os.remove(data_spec_fpath)

        # Iterate over all environments and trajectories, and build an entry for each one.
        for env_name in env:
            print("Building data cacher for env {}...".format(env_name))
            # If no difficulty was given, then use all difficulties.
            if not difficulty:
                available_diffs = [diff for diff in ['easy', 'hard'] if os.path.isdir(os.path.join(self.tartanair_data_root, env_name, 'Data_' + diff))]
                print([os.path.join(self.tartanair_data_root, env_name, 'Data_' + diff) for diff in ['easy', 'hard']])
                print(Fore.GREEN + "WARNING: No difficulty was specified for env {}. Defaulting to all available difficulties: {}".format(env_name, available_diffs),  Style.RESET_ALL)

            else:
                available_diffs = difficulty

            for diff in available_diffs:

                # If no trajectory id was given, then use all trajectories.
                available_traj_ids = self.get_available_trajectory_ids(env_name, diff)
                if not trajectory_id:
                    print(Fore.GREEN + "WARNING: No trajectory id was specified for env {} and difficulty {}. Defaulting to all available trajectories: {}".format(env_name, diff, available_traj_ids),  Style.RESET_ALL)
                else:
                    for traj_id in trajectory_id:
                        if traj_id not in available_traj_ids:
                            print(Fore.RED + "WARNING: Trajectory id {} was specified for env {} and difficulty {}, but it is not available. It is skipped.".format(traj_id, env_name, diff),  Style.RESET_ALL)
                    available_traj_ids = [traj_id for traj_id in available_traj_ids if traj_id in trajectory_id]
                
                for traj_id in available_traj_ids:
                    # Build the data entry.         

                    # Read the data specification file and concatenate it to the data spec file.
                    with open(data_spec_fpath, 'a') as f:
                        env_diff_traj_data_spec_fpath = os.path.join(self.tartanair_data_root, env_name, "analyze", 'data_' + env_name + '_Data_' + diff + '_' + traj_id + '.txt')

                        # Read the data spec file.
                        # with open(env_diff_traj_data_spec_fpath, 'r') as f2:
                        # Create a new data spec file for this trajectory, where it is broken down into smaller pseudo-trajectories.
                        # This is done to allow for the data cacher to cache the data in small trajectory portions, such that the data is shuffled also when the RAM memory allocation is small.
                        pseudo_traj_length = max(max(seq_length.values()) * seq_stride * (frame_skip + 1) * 2, 20)
                        pseudo_traj_spec_list = self.traj_spec_file_to_pseudo_traj_spec_list(env_diff_traj_data_spec_fpath, pseudo_traj_length)
                        # Write all but the last line.
                        # for line in f2.readlines():
                        for line in pseudo_traj_spec_list:
                            f.write(line)

        # Build the data entry. It is shared for the entire dataset.
        data_entry = {}
        data_entry['modality'] = {}

        self.add_modality_entries(data_entry, modality, camera_name, new_image_shape_hw, seq_length, subset_framenum, num_workers)

        # Build the cacher entry.
        data_entry['cacher'] = {}
        data_entry['cacher']['data_root_key'] = 'tartanairv2' # TODO(yoraish): this is disregarded currently and overriden by the data_root_path_override.
        data_entry['cacher']['subset_framenum'] = subset_framenum
        data_entry['cacher']['worker_num'] = num_workers
        # Add the tartanair data root to the config.
        data_entry['cacher']['data_root_path_override'] = self.tartanair_data_root
        # Build the transform entry.
        data_entry['transform'] = {}
        data_entry['transform']['resize_factor'] = 1.0
        # Build the dataset entry.
        data_entry['dataset'] =  None

        # Add the data entry to the config with the key being the path to the frame-enumeration file.
        config['data'][data_spec_fpath] = data_entry

        print(config)
        # Create the data loader from the config.
        trainDataloader = MultiDatasets(config, 
                        'local', 
                        batch= batch_size, 
                        workernum= num_workers,
                        shuffle= shuffle,
                        verbose= verbose)

        return trainDataloader

    def traj_spec_file_to_pseudo_traj_spec_list(self, traj_spec_fpath, pseudo_traj_length):
        '''
        Read the trajectory specification file and break it down into smaller pseudo-trajectories. This is done to allow for the data cacher to cache the data in small trajectory portions, such that the data is shuffled also when the RAM memory allocation is small.
        '''
        pseudo_traj_spec_list = []
        with open(traj_spec_fpath, 'r') as f:
            lines = f.readlines()

            # Get the current traj header.
            traj_header = lines[0]
            lines = lines[1:]

            # The only thing that we'll change is the number of files, which is the last number in the string.
            numless_header = traj_header[:traj_header.rfind(' ') + 1]

            for i in range(0, len(lines)):
                if i % pseudo_traj_length == 0:
                    # Add a new line. If we have more than traj_len lines left, then add the full traj_len. Otherwise, add the remaining lines.
                    if i + pseudo_traj_length < len(lines):
                        pseudo_traj_spec_list.append(numless_header + str(pseudo_traj_length) + '\n')

                    else:
                        pseudo_traj_spec_list.append(numless_header + str(len(lines) - i) + '\n')
                
                # Append the line.
                pseudo_traj_spec_list.append(lines[i])

        return pseudo_traj_spec_list

    def add_modality_entries(self, data_entry, modality, camera_name, new_image_shape_hw, seq_length, subset_framenum, num_workers):

        '''
        Mutate the data_entry dictionary to add modality entries for all types in the combined modality and camera_name lists.
        Convert a modality list and camera name list to a list of modcamalities. Those are the modality names that remain after combining the modality and camera name lists. For example, if 'image' and 'lidar' is in modality, and 'lcam_front' is in camera_name, then 'rgb_lcam_front', 'lidar' will be in the returned list, as 'lidar' is not a modality that requires a camera name.
        '''

        # Create the list of types. Each type is a modality_camera name, if the modality requires a camera name, otherwise it is just the modality.
        types = []
        for mod in modality:
            if mod in ['image', 'depth', 'seg', 'flow', 'pose']:
                for cam in camera_name:
                    mod = self.modality_name_remaps[mod]
                    types.append(mod + '_' + cam)
            else:
                types.append(mod)

        for type in types:
            # Build the modality entry.
            modality_entry = {}
            mod = type.split('_')[0]

            # Account for modality names that have `_` in them.
            if mod == 'imu':
                mod = '_'.join(type.split('_')[:2])

            if mod in ['rgb', 'depth', 'seg', 'flow']:
                modality_entry['cacher_size'] = new_image_shape_hw
            else: # 'lidar', 'imu', 'pose', etc.
                modality_entry['cacher_size'] = self.modality_default_cacher_size[mod]
            
            modality_entry['length'] = seq_length[mod]
            modality_entry['subset_framenum'] = subset_framenum

            # Add the modality entry to the data entry.
            data_entry['modality'][type] = modality_entry
            data_entry['modality'][type]['type'] = type    


    def get_available_trajectory_ids(self, env_name, diff):
            '''
            Return a list of trajectory ids for the given environment and difficulty.
            '''

            # Get the path to the env directory.
            env_dir_path = os.path.join(self.tartanair_data_root, env_name, "Data_" + diff)
            # Check what's there of the form 'P0..'.
            traj_ids = []
            for f in os.listdir(env_dir_path):
                if f.startswith('P'):
                    traj_ids.append(f)

            return traj_ids




