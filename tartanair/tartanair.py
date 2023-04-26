# Local imports.
from .downloader import TartanAirDownloader
from .dataset import TartanAirDataset, TartanAirSlowLoaderCreator
from .customizer import TartanAirCustomizer
from .lister import TartanAirLister
from .visualizer import TartanAirVisualizer
from .iterator import TartanAirIterator
from .evaluator import TartanAirEvaluator
from .reader import TartanAirTrajectoryReader
from .dataloader import TartanAirDataLoader

print("TartanAir toolbox initialized.")

tartanair_data_root = ""
downloader = None
dataset = None
customizer = None
lister = None
visualizer = None
iterator = None
evaluator = None
dataloader = None
slowloader = None

# Flag for initialization.
is_init = False

def init(tartanair_root, azure_token = None):
    """
    Initialize the TartanAir toolbox. Call this method before using any other method in the toolbox.

    :param tartanair_root: The root directory of the TartanAir dataset.
    :type tartanair_root: str
    """

    global tartanair_data_root
    tartanair_data_root = tartanair_root

    global downloader
    # If a token is provided, use it. Otherwise, let the downloader use self.azure_token from the parent class.
    downloader = TartanAirDownloader(tartanair_data_root, azure_token = azure_token)

    global dataset
    dataset = TartanAirDataset(tartanair_data_root)

    global customizer
    try:
        customizer = TartanAirCustomizer(tartanair_data_root)
    except:
        print("Could not initialize customizer.")

    global lister
    lister = TartanAirLister(tartanair_data_root)

    global visualizer
    visualizer = TartanAirVisualizer(tartanair_data_root)

    global iterator
    iterator = TartanAirIterator(tartanair_data_root)

    global traj_reader
    traj_reader = TartanAirTrajectoryReader(tartanair_data_root)

    global evaluator
    evaluator = TartanAirEvaluator(tartanair_data_root)

    global dataloader
    dataloader = TartanAirDataLoader(tartanair_data_root)

    global slowloader
    slowloader = TartanAirSlowLoaderCreator(tartanair_data_root)

    global is_init 
    is_init = True
    
    return True
    

def download(env = [], difficulty = [], trajectory_id = [], modality = [], camera_name = [], config = None):
    """
    Download data from the TartanAir dataset. This method will download the data from the Azure server and store it in the `tartanair_root` directory.

    :param env: The environment to download. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: easy, hard.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to download. Can be a list of trajectory ids of form P000, P001, etc.
    :type trajectory_id: str or list
    :param modality: The modality to download. Can be a list of modalities. Valid modalities are: image, depth, seg, imu{_acc, _gyro...}, lidar. Default will include all.
    :type modality: str or list
    :param camera_name: The camera name to download. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
     Modalities IMU and LIDAR do not need camera names specified.
    :type camera_name: str or list
    :param config: Optional. Path to a yaml file containing the download configuration. If a config file is provided, the other arguments will be ignored.
    :type config: str
    """

    global downloader
    check_init()
    downloader.download(env, difficulty, trajectory_id, modality, camera_name, config)

def customize(env, difficulty, trajectory_id, modality, new_camera_models_params = [{}], num_workers = 1, device = "cpu"):
    """
    Synthesizes raw data into new camera-models. A few camera models are provided, although you can also provide your own camera models. The currently available camera models are:

    * 'pinhole': A pinhole camera model.
    * 'doublesphere': A wide-angle camera model with a double sphere distortion model. Source: https://arxiv.org/abs/1807.08957
    * 'linearsphere': A wide-angle camera model with a custom "linear sphere" distortion model.
    * 'equirect': An equirectangular camera model.
    
    :param env: The environment to customize. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to customize. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str or list
    :param modality: The modality to be customized. Can be a list of modalities. Valid modalities are: `image`, `depth`, `seg`.
    :type modality: str or list
    :param new_camera_models_params: A list of dictionaries containing the parameters for the new camera models. Each dictionary should contain the following keys:
            
        * `name`: The name of the camera model. Valid camera models are: `pinhole`, `doublesphere`, `linearsphere`, `equirect`.
        * `raw_side`: The raw camera side. Can be one of `left`, `right`.
        * `R_raw_new`: The rotation matrix from the raw camera frame, with z pointing out of the image frame, x right, y down, to the new camera frame.
        * `params`: A dictionary containing the parameters for the new camera model. The parameters for each camera model are:
            * `pinhole`: `fx`, `fy`, `cx`, `cy`, `height`, `width`.
            * `doublesphere`: `fx`, `fy`, `cx`, `cy`, `height`, `width`, `xi`, `alpha`, `fov_degree`.
            * `linearsphere`: `fx`, `fy`, `cx`, `cy`, `height`, `width`, `fov_degree`.
            * `equirect`: `height`, `width`.
    :type new_camera_models_params: list
    :param num_workers: The number of workers to use for the customizer. Default is 1.
    :type num_workers: int
    """
    global customizer
    check_init()
    customizer.customize(env, difficulty, trajectory_id, modality, new_camera_models_params, num_workers=num_workers, device=device)

def dataloader(env, 
            difficulty = [], 
            trajectory_id = [], 
            modality = [], 
            camera_name = [], 
            new_image_shape_hw = [640, 640], 
            subset_framenum = 360, 
            seq_length = 1, 
            seq_stride = 1, 
            frame_skip = 0, 
            batch_size = 8, 
            num_workers = 0, 
            shuffle = False, 
            verbose = False):
    """
    Create a dataloader object, reading data from the TartanAir-V2 dataset and serving it in mini-batches. Note that under the hood a powerful data-cacher is employed, which allows for efficient data loading and mini-batch serving concurrently. The dataloader operates in the following way:

    1. It loads a subset of the dataset to RAM.
    2. It serves mini-batches from this subset. The mini-batches are of data-sequences. So for example, if the sequence length is 2, then the mini-batch will have samples of 2 frames each. A batch size of 16 means that the mini-batch will contain 16 pairs of images, for the example of images. The samples do not have to be consecutive, and the 'skip' between the samples can be specified. The sequences also do not have to start from consecutive indices, and the stride between the sequences can be specified.
    3. The dataloader will load a new subset of the dataset to RAM while the mini-batches are loaded from the first subset, and switch the subsets when the first subset is exhausted. If the first subset is exhausted before the mini-batches are loaded, then the dataloader will keep loading mini-batches from the first subset until the second subset is loaded.


    :param env: The environments to load. Can be a list of environments or a single environment.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties or a single difficulty. Valid difficulties are: easy, hard. If empty, all difficulties will be loaded.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to load. Can be a list of trajectory ids of form P000, P001, etc.
    :type trajectory_id: str or list
    :param modality: The modality to load. Can be a list of modalities or a single modality. Valid modalities are: image, depth, seg, flow, imu{_acc, _gyro, ...}, lidar. If empty, a sample of a few modalities will be loaded. Please specify your requested modalities explicitly or be pleasantly surprised by the data you get.
    :type modality: str or list
    :param camera_name: The camera name to load. Can be a list of camera names or a single camera name. Valid camera names are: lcam_front, lcam_rear, lcam_left, lcam_right, lcam_fish, lcam_equirect, rcam_front, rcam_rear, rcam_left, rcam_right, rcam_fish, rcam_equirect. If empty, all cameras will be loaded.
    :type camera_name: str or list
    :param new_image_shape_hw: The new image shape to resize the images to [height, width]. If empty, the original image shape [640, 640] will be used.
    :type new_image_shape_hw: list
    :param subset_framenum: The number of frames to load in a single subset on the RAM, per modality type. If empty, 360 frames will be loaded. Notice that this is an upper bound on the bath size as well, as batches are harvested from the subset that has been loaded to RAM, so they can be at most as large as the subset.
    :type subset_framenum: int
    :param frame_skip: The number of frames to skip between consecutive frames in a sequence. If empty, no frames will be skipped.
    :type frame_skip: int
    :param seq_length: The length of the sequences to be loaded. If empty, a sequence length of 1 will be used. It is possible to pass a dictionary mapping modalities to sequence lengths, in which case the sequence length for each modality will be set to the corresponding value. For example, if the dictionary is {'image': 2, 'depth': 1}, then the sequence length for the image modality will be 2 (pairs of consecutive images), and the sequence length for the depth modality will be 1.
    :type seq_length: int or dict
    :param seq_stride: The stride between the sequences. If empty, a stride of 1 will be used.  
    :type seq_stride: int
    :param batch_size: The batch size to load. If empty, a batch size of 8 will be used.    
    :type batch_size: int
    :param num_workers: The number of workers to use for the dataloader. If empty, 0 workers will be used.
    :type num_workers: int
    :param shuffle: Whether to shuffle the data. If empty, the data will not be shuffled. Note that the shuffle is within the subset, not across subsets.
    :type shuffle: bool
    :param verbose: Whether to print information regarding memory usage and trajectory loading. If empty, no verbose information will be printed.
    :type verbose: bool
    """
    global dataloader
    check_init()
    return dataloader.get_data_cacher(env = env, 
            difficulty = difficulty, 
            trajectory_id = trajectory_id, 
            modality = modality, 
            camera_name = camera_name, 
            new_image_shape_hw = new_image_shape_hw, 
            subset_framenum = subset_framenum, 
            seq_length = seq_length, 
            seq_stride = seq_stride, 
            frame_skip = frame_skip, 
            batch_size = batch_size, 
            num_workers = num_workers, 
            shuffle = shuffle, 
            verbose = verbose)

def create_image_dataset(env, difficulty = None, trajectory_id = None, modality = None, camera_name = None, transform = None, num_workers = 1):
    """
    Creates a frame-pair PyTorch dataset for a specified subset of the TartanAir dataset. Pairs of consecutive frames are returned, alongside the transform between the poses of the cameras that took the images.
    This dataset handles images in modalities such as 'image', 'depth', and 'seg'.
    Each dataset batch contains samples of the form, for example:

    >>> {'lcam_front': 
            {'image_0': tensor(B, 3, H, W), 
             'image_1': tensor(B, 3, H, W), 
             'depth_0': tensor(B, H, W), 
             'depth_1': tensor(B, H, W), 
             'motion': tensor (B, 6)}
            }

    :param env: The environment to create the dataset from. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to create the dataset from. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str or list
    :param modality: The modality to create the dataset from. Can be a list of modalities. Valid modalities are: `image`, `depth`, `seg`.
    :type modality: str or list
    :param camera_name: The camera name to create the dataset from. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :type camera_name: str or list
    :param num_workers: The number of workers to use for the dataset preprocessing. Default is 1.
    :type num_workers: int
    :return: A MultiDatasets object. Please see the examples for usage.
    :rtype: torch.utils.data.Dataset
    """
    global dataset
    check_init()
    return dataset.create_image_dataset(env, difficulty, trajectory_id, modality, camera_name, transform, num_workers)


def create_image_slowloader(env,  difficulty = None,  trajectory_id = None,  modality = None,  camera_name = None, batch_size = 1, shuffle = True, num_workers=1):
    """
    Creates a frame-pair PyTorch dataset for a specified subset of the TartanAir dataset. Pairs of consecutive frames are returned, alongside the poses of the cameras that took the images.
    This dataset handles images in modalities such as 'image', 'depth', and 'seg'.
    Each dataset batch contains samples of the form, for example:

    >>> {'rgb_lcam_front': tensor(B, S, H, W, 3),
        ...
        'depth_lcam_front': tensor(B, S, H, W),
        ...
        'pose_lcam_front': tensor(B, S, 7), # xyz, xyzw.
        ...
        'motion_lcam_front': tensor(B, S-1, 6), # Optional.
    }

    :param env: The environment to create the dataset from. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to create the dataset from. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str or list
    :param modality: The modality to create the dataset from. Can be a list of modalities. Valid modalities are: `image`, `depth`, `seg`.
    :type modality: str or list
    :param camera_name: The camera name to create the dataset from. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :type camera_name: str or list
    :param num_workers: The number of workers to use for the dataset preprocessing. Default is 1.
    :type num_workers: int
    :return: A MultiDatasets object. Please see the examples for usage.
    :rtype: torch.utils.data.Dataset
    """
    global slowloader
    check_init()
    return slowloader.create_image_slowloader(env,  difficulty,  trajectory_id,  modality,  camera_name, batch_size, shuffle, num_workers)


def list_envs():
    """
    List all the environments in the TartanAir dataset.
    :return: A dictionary with the local and remote environment names.
    :rtype: dict
    """
    global lister    
    check_init()
    return lister.list_envs()

def visualize(env, difficulty, trajectory_id, modality, camera_name = None):
    """
    Interactively visualizes a trajectory from the TartanAir dataset that is saved locally.

    :param env: The environment to visualize. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to visualize. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str or list
    :param modality: The modality to visualize. Can be a list of modalities. Valid modalities are: `image`, `depth`, `seg`.
    :type modality: str or list
    :param camera_name: The camera name to visualize. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :type camera_name: str or list

    """
    global visualizer    
    check_init()
    visualizer.visualize(env, difficulty, trajectory_id, modality, camera_name)

def check_init():
    global is_init
    if not is_init:
            raise Exception("TartanAir toolbox not initialized. Please call tartanair.init(tartanair_data_root) first.")

def iterator( env = None, difficulty = None, trajectory_id = None, modality = None, camera_name = None):
    """
    Creates an iterator for the TartanAir dataset.

    :param env: The environment to iterate over. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to iterate over. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str or list
    :param modality: The modality to iterate over. Can be a list of modalities. Valid modalities are: `image`, `depth`, `seg`.
    :type modality: str or list
    :param camera_name: The camera name to iterate over. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :type camera_name: str or list
    :return: An iterator over the TartanAir dataset.
    :rtype: generator
    """
    global iterator
    global tartanair_data_root
    check_init()
    return iterator.get_iterator(env, difficulty, trajectory_id, modality, camera_name)

def get_traj_np(env, difficulty, trajectory_id, camera_name = None):
    """
    Returns the trajectory as a numpy array.

    :param env: The environment to get the trajectory from. Can be a list of environments.
    :type env: str
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`.
    :type difficulty: str
    :param trajectory_id: The id of the trajectory to get. Can be a list of trajectory ids of form `P000`, `P001`, etc.
    :type trajectory_id: str
    :param camera_name: The camera name to get the trajectory from. Can be a list of camera names. Default will include all. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :return: The trajectory as a numpy array. The array is of shape (N, 7) where N is the number of poses in the trajectory. The poses are in NED format and are of the form [x, y, z, qx, qy, qz, qw].
    :rtype: np.array
    """
    global tartanair_data_root
    global traj_reader
    check_init()
    return traj_reader.get_traj_np(env, difficulty, trajectory_id, camera_name)

def evaluate_traj(est_traj,
             gt_traj = None,
             env = None, 
             difficulty = None, 
             trajectory_id = None, 
             camera_name = None, 
             enforce_length = True, 
             plot = False, 
             plot_out_path = None, 
             do_scale = True, 
             do_align = True):
    """
    Evaluates a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion. In progress.

    :param est_traj: The estimated trajectory to evaluate. This is specified as an array of 3D poses in NED and format [x, y, z, qx, qy, qz, qw].
    :type est_traj: np.array
    :param gt_traj: The ground truth trajectory to evaluate against. This is specified as an array of 3D poses in NED and format [x, y, z, qx, qy, qz, qw]. If None, will use the ground truth trajectory from the TartanAir dataset.
    :type gt_traj: np.array
    :param env: The environment to evaluate the trajectory from. If passing a gt_traj, this is ignored.
    :type env: str
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: `easy`, `hard`. If passing a gt_traj, this is ignored.
    :type difficulty: str
    :param trajectory_id: The id of the trajectory to ground truth trajectory. Of form `P000`, `P001`, etc. If passing a gt_traj, this is ignored.
    :type trajectory_id: str
    :param camera_name: The camera name to evaluate the trajectory for. Choices are `lcam_front`, `lcam_right`, `lcam_back`, `lcam_left`, `lcam_top`, `lcam_bottom`, `rcam_front`, `rcam_right`, `rcam_back`, `rcam_left`, `rcam_top`, `rcam_bottom`, `lcam_fish`, `rcam_fish`, `lcam_equirect`, `rcam_equirect`.
    :type camera_name: str
    :param enforce_length: If False, the ground truth trajectory will be truncated to the length of the estimated trajectory. If True, the estimated trajectory will be required to match the length of the ground truth trajectory.
    :type enforce_length: bool
    :param plot: If True, will plot the trajectory and save to plot_out_path.
    :type plot: bool
    :param plot_out_path: The path to save the plot to. Disregarded if plot is False.
    :type plot_out_path: str
    :param do_scale: If True, will scale the estimated trajectory to match the ground truth trajectory. A single scale factor will be applied to all dimensions.
    :type do_scale: bool
    :param do_align: If True, will align the estimated trajectory to match the ground truth trajectory.
    :type do_align: bool

    :return: A dictionary containing the evaluation metrics, which include ATE, RPE, the ground truth trajectory, and the estimated trajectory after alignment and scaling if those were requested
    :rtype: dict

    """
    global evaluator    
    check_init()
    return evaluator.evaluate_traj(est_traj,
                                   gt_traj, 
                                   env, 
                                   difficulty, 
                                   trajectory_id, 
                                   camera_name, 
                                   enforce_length, 
                                   plot, 
                                   plot_out_path, 
                                   do_scale, 
                                   do_align)


