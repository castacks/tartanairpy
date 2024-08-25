'''
Author: Yorai Shaoul, Wenshan Wang
Date: 2023-03-27

This class provides an interface to the Data-Cacher module. The data cacher is an efficient two-stage loader that allows for continuous training: parallelizing loading data and serving mini-batches.

Output batches will be of the form:

'''

# TODO(yoraish): there is a notation discrepancy between 'rgb' and 'image'. Should fix this and probably stick with 'image' as this is the name used in the dataset. 

# General imports.
import os
from os.path import join, isfile

# Local imports.
from .tartanair_module import TartanAirModule, print_error, print_highlight, print_warn
from .data_cacher.MultiDatasets import MultiDatasets
from .data_cacher.datafile_editor import generate_datafile, enumerate_frames, breakdown_trajectories

class TartanAirDataLoader(TartanAirModule):
    '''
    The TartanAirDataset class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root):
        # Call the parent class constructor.
        super(TartanAirDataLoader, self).__init__(tartanair_data_root)

        self.modality_default_seq_length = {
            'image': 1,
            'depth': 1,
            'seg': 1,
            'flow': 1,
            'pose': 1,
            'imu': 10,
            'lidar': 1,
        }

        self.modality_default_cacher_size = {
            'image': [640, 640],
            'depth': [640, 640],
            'seg': [640, 640],
            'flow': [640, 640],
            'pose': [7],
            'imu': [6],
            'lidar': [3],
        }

    def generate_data_file(self, env, difficulty, trajectory_id, onemodfolder, breakdown = True):
        print_highlight("Generating datafile...")

        # find all the local trajectories with the given difficulty level
        difficulty = ['Data_' + dd for dd in difficulty]
        local_traj_dict = self.enumerate_trajs(difficulty)

        # Iterate over all environments and trajectories, and fine all the trajectories the user is interested.
        trajstrlist, framelist = [], []
        for env_name in env:
            print("Building data cacher for env {}...".format(env_name))
            if not (env_name in local_traj_dict):
                print_error("Could not find env {} in the local directory {}".format(env_name, local_traj_dict.keys))
                return None

            # find the trajectories that satisfy the user requirement
            local_traj_env = local_traj_dict[env_name]
            for traj_str in local_traj_env:
                dif_str, trajid = traj_str.split('/')
                if trajid in trajectory_id or len(trajectory_id) == 0:
                    trajstr = join(env_name, traj_str)
                    trajstrlist.append(trajstr)
                    frames = enumerate_frames(join(self.tartanair_data_root, trajstr, onemodfolder))
                    framelist.append(frames)

            if len(trajstrlist) == 0:
                print_error("Could not find any trajectory to load! ")
                return None

        # hard coded, we use the following datafile
        datafile = 'ta_datafile.txt'
        if isfile(datafile):
            print_warn("Removing the existing datafile {}".format(datafile))
            os.remove(datafile)

        if breakdown: # break down the trajectory so that it is more random when the buffer size is limited
            trajstrlist, framelist = breakdown_trajectories(trajstrlist, framelist)

        generate_datafile(datafile, trajstrlist, framelist)
        return datafile


    def generate_config_file(self, datafile, foldernames, image_shape_hw, seq_length, 
                                    subset_framenum, seq_stride, frame_skip, num_workers):
        config = {}
        config['task'] = 'tartanair'
        config['global'] = {}

        config['global']['modality'] = {}

        config['global']['cacher'] = {}
        config['global']['cacher']['load_traj'] = False
        config['global']['cacher']['data_root_path_override'] = self.tartanair_data_root
        config['global']['cacher']['subset_framenum'] = subset_framenum
        config['global']['cacher']['worker_num'] = num_workers

        config['global']['dataset'] = {}
        config['global']['dataset']['frame_skip'] = frame_skip
        config['global']['dataset']['seq_stride'] = seq_stride
        config['global']['dataset']['frame_dir'] = True

        config['global']['parameter'] = {}

        # Build the data entry. It is shared for the entire dataset.
        data_entry = {}
        data_entry['file'] = datafile
        data_entry['modality'] = {}
        data_entry['cacher'] = {}
        data_entry['dataset'] = {}
        data_entry['parameter'] = {}

        self.add_modality_entries(data_entry['modality'], foldernames, image_shape_hw, seq_length)

        config['data'] = {}
        config['data']['1'] = data_entry # the data_cacher support loading multiple datasets, we are only using one here

        print(config)

        return config


    def get_data_cacher(self, 
                        env, 
                        difficulty = None, 
                        trajectory_id = None, 
                        modality = None, 
                        camera_name = None, 
                        new_image_shape_hw = [640, 640],  # This can also be a dictionary, mapping each modality name to a shape.
                        seq_length = 1, # This can also be a dictionary, mapping each modality name to a sequence length.
                        subset_framenum = 56, # <--- Note in the docs that this is an upper bound on the batch size. In general, this should be as large as possible 
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
            difficulty = self.difficulty_names # default to all difficulties 
        if not trajectory_id:
            trajectory_id = [] # Empty list will default to all trajs down the line.
        if not modality:
            print_warn("WARNING: No modality was specified. Defaulting to _some_ modalities: ('image', 'depth', 'seg')")
            modality = ['image', 'depth', 'seg']
        if not camera_name:
            camera_name = self.camera_names #['lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_top', 'lcam_bottom', 'lcam_fish', 'lcam_equirect', 'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_top', 'rcam_bottom', 'rcam_fish', 'rcam_equirect']

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

        # Check that the environments are valid.
        if not self.check_env_valid(env):
            return False
        # Check that the modalities are valid
        if not self.check_modality_valid(modality):
            return False
        # Check that the difficulty are valid
        if not self.check_difficulty_valid(difficulty):
            return False
        # Check that the camera names are valid
        if not self.check_camera_valid(camera_name):
            return False

        # figuring out the combination of modality and camera_name
        # folderlist consists all the folders that need to be load under each trajectory
        folderlist = self.compile_modality_and_cameraname(modality, camera_name)
        # find one folder that's not imu, because imu is not frame-based
        onemodfolder = None
        for fl in folderlist:
            if not fl.endswith('imu'):
                onemodfolder = fl
                break
        if not onemodfolder:
            print_error("No frame-based modality is available")

        datafile = self.generate_data_file(env, difficulty, trajectory_id, onemodfolder)
        if not datafile:
            return False

        # process pose modality seperately here because it is not frame-based 
        if 'pose' in modality:
            for camname in camera_name:
                folderlist.append('pose_' + camname)

        config = self.generate_config_file(datafile, folderlist, new_image_shape_hw, seq_length, 
                                            subset_framenum, seq_stride, frame_skip, num_workers)

        # Create the data loader from the config.
        trainDataloader = MultiDatasets(config, 
                        'local', 
                        batch= batch_size, 
                        workernum= 1,
                        shuffle= shuffle,
                        verbose= verbose)

        return trainDataloader

    def add_modality_entries(self, data_entry, foldernames, image_shape_hw, seq_length):

        '''
        Mutate the data_entry dictionary to add modality entries for all types in the combined modality and camera_name lists.
        Convert a modality list and camera name list to a list of modcamalities. Those are the modality names that remain after combining the modality and camera name lists. For example, if 'image' and 'lidar' is in modality, and 'lcam_front' is in camera_name, then 'rgb_lcam_front', 'lidar' will be in the returned list, as 'lidar' is not a modality that requires a camera name.
        '''
        # Allow to take in a dictionary for the modality sequence length.
        # If an integer is given, then use it for all modalities.
        if type(seq_length) is int:
            new_seq_length = {mod: seq_length for mod in foldernames}
        elif type(seq_length) is dict:
            new_seq_length = {}
            for foldername in foldernames:
                modname = foldername.split('_')[0]
                if modname in seq_length:
                    new_seq_length[foldername] = seq_length[modname]
                else:
                    print_warn("WARNING: No sequence length was given for modality {}. Defaulting to {}.".format(modname, self.modality_default_seq_length[modname]))
                    new_seq_length[foldername] = self.modality_default_seq_length[modname]
        else:
            raise ValueError("seq_length must be an integer or a dictionary.")

        if type(image_shape_hw) is list:
            new_image_shape_hw = {mod: image_shape_hw for mod in foldernames}
        elif type(image_shape_hw) is dict:
            new_image_shape_hw = {}
            for foldername in foldernames:
                modname = foldername.split('_')[0]
                if modname in image_shape_hw:
                    new_image_shape_hw[foldername] = image_shape_hw[modname]
                else:
                    print_warn("WARNING: No shape was given for modality {}. Defaulting to {}.".format(modname, self.modality_default_cacher_size[modname]))
                    new_image_shape_hw[foldername] = self.modality_default_cacher_size[modname]
        else:
            raise ValueError("image_shape_hw must be an integer or a dictionary.")

        for folder in foldernames:
            data_entry[folder] = {} # modality name
            data_entry[folder][folder] = {} # the key name returned by the dataloader
            data_entry[folder][folder]['cacher_size'] = new_image_shape_hw[folder]
            data_entry[folder][folder]['length'] = new_seq_length[folder]

if __name__=="__main__":
    loader = TartanAirDataLoader('/data/tartanair_v2')
    datafile = loader.generate_data_file(env=['coalmine'], difficulty=['easy', 'hard'], trajectory_id=[], onemodfolder='seg_rcam_right', breakdown = True)

    loader.generate_config_file(datafile, 
                                foldernames=['depth_lcam_back','image_lcam_back'], 
                                image_shape_hw=[640, 640], 
                                seq_length=1, 
                                subset_framenum=100, 
                                seq_stride=1, 
                                frame_skip=0, 
                                num_workers=4)
