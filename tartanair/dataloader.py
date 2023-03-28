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
            'imu': 10,
            'lidar': 1,
        }

        self.modality_default_cacher_size = {
            'rgb': [640, 640],
            'depth': [640, 640],
            'seg': [640, 640],
            'flow': [640, 640],
            'pose': [7],
            'imu': [3],
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
                        frame_skip = 0,
                        seq_length = 2,
                        seq_stride = 1,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False):

        '''
        Create a dataloader object, reading data from the TartanAir dataset, and return it. Note that under the hood a very powerful data cacher is used, which allows for efficient data loading and mini-batch serving. This method effectively creates a config file to this data cacher, and returns an iterator object that can be used to load data. This method effectively populates a config dictionary and uses it to create a dataloader object.

        Args:

        '''

        # Add default values to empty inputs.
        if not difficulty:
            difficulty = ['easy', 'hard']
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

        # Iterate over all environments and trajectories, and build an entry for each one.
        for env_name in env:
            for diff in difficulty:

                # If no trajectory id was given, then use all trajectories.
                available_traj_ids = trajectory_id
                if not trajectory_id:
                    print("WARNING: No trajectory id was specified. Defaulting to all trajectories.")
                    available_traj_ids = self.get_available_trajectory_ids(env_name, diff)
                    print("Available trajectories for env {} and difficulty {}: {}".format(env_name, diff, available_traj_ids))
                for traj_id in available_traj_ids:
                    # Build the data entry.                    
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
                    config['data'][os.path.join(self.tartanair_data_root, env_name, "analyze", 'data_' + env_name + '_Data_' + diff + '_' + traj_id + '.txt')] = data_entry

        print("Config: {}".format(config))




        # Create the data loader from the config.
        trainDataloader = MultiDatasets(config, 
                        'local', 
                        batch= batch_size, 
                        workernum= num_workers,
                        shuffle= shuffle)

        return trainDataloader

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
            if mod in ['rgb', 'depth', 'seg', 'flow']:
                modality_entry['cacher_size'] = new_image_shape_hw
            else: # 'lidar', 'imu', 'pose', etc.
                modality_entry['cacher_size'] = self.modality_default_cacher_size[mod]
            
            modality_entry['length'] = seq_length[mod]
            modality_entry['subset_framenum'] = subset_framenum

            # Add the modality entry to the data entry.
            data_entry['modality'][type] = modality_entry
            data_entry['modality'][type]['type'] = type

            print(modality_entry)
    


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






# {
#     'task': 'flowvo', 
#     'transform_data_augment': True, 
#     'transform_flow_norm_factor': 0.05, 
#     'transform_uncertainty': True, 
#     'transform_input_size': [160, 160], 
#     'dataset_frame_skip': 0, 
#     'dataset_seq_stride': 1, 
#     'data': {
#         '/media/yoraish/overflow/data/tartanair-v2/HQWesternSaloonExposure/analyze/data_HQWesternSaloonExposure_Data_easy_P000.txt': 
#         {
#         'modality': {
#             'img0': {
#             'type': 'rgb_lcam_front', 'cacher_size': [160, 160], 'length': 3}, 
#             'lcam_left': {
#             'type': 'rgb_lcam_left', 'cacher_size': [160, 160], 'length': 3}, 
#             'lcam_back': {
#             'type': 'rgb_lcam_back', 'cacher_size': [160, 160], 'length': 3}, 
#             'lcam_right': {
#             'type': 'rgb_lcam_right', 'cacher_size': [160, 160], 'length': 3}, 
#             'lcam_top': {
#             'type': 'rgb_lcam_top', 'cacher_size': [160, 160], 'length': 3}, 
#             'lcam_bottom': {
#             'type': 'rgb_lcam_bottom', 'cacher_size': [160, 160], 'length': 3}, 
#             'depth0': {
#             'type': 'depth_lcam_bottom', 'cacher_size': [160, 160], 'length': 1}}, 
#         'cacher': {
#             'data_root_key': 'tartan2', 
#             'subset_framenum': 120, 
#             'worker_num': 2}, 
#         'transform': {
#             'resize_factor': 2.5}, 
#          'dataset': None}}}




