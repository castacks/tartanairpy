# Local imports.
from .downloader import TartanAirDownloader
from .dataset import TartanAirDataset
# from .customizer import TartanAirCustomizer
from .lister import TartanAirLister
from .visualizer import TartanAirVisualizer
from .iterator import TartanAirIterator
from .evaluator import TartanAirEvaluator
from .reader import TartanAirTrajectoryReader

print("TartanAir toolbox initialized.")

tartanair_data_root = ""
downloader = None
dataset = None
customizer = None
lister = None
visualizer = None
iterator = None
evaluator = None

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

    # global customizer
    # try:
    #     customizer = TartanAirCustomizer(tartanair_data_root)
    # except:
    #     print("Could not initialize customizer.")

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

    global is_init 
    is_init = True
    

def download(env = [], difficulty = [], trajectory_id = [], modality = [], camera_name = [], config = None):
    """
    Download data from the TartanAir dataset. This method will download the data from the Azure server and store it in the `tartanair_root` directory.

    :param env: The environment to download. Can be a list of environments.
    :type env: str or list
    :param difficulty: The difficulty of the trajectory. Can be a list of difficulties. Valid difficulties are: easy, hard.
    :type difficulty: str or list
    :param trajectory_id: The id of the trajectory to download. Can be a list of trajectory ids of form P000, P001, etc.
    :type trajectory_id: str or list
    :param modality: The modality to download. Can be a list of modalities. Valid modalities are: image, depth, seg, imu, lidar. Default will include all.
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

def customize(env, difficulty, trajectory_id, modality, new_camera_models_params = [{}], num_workers = 1):
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
    # global customizer
    check_init()
    # customizer.customize(env, difficulty, trajectory_id, modality, new_camera_models_params, num_workers=num_workers)

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
    :param transform: Optional. A transform to apply to the images. Default is None.
    :type transform: torchvision.transforms
    :param num_workers: The number of workers to use for the dataset preprocessing. Default is 1.
    :type num_workers: int
    :return: A PyTorch dataset.
    :rtype: torch.utils.data.Dataset
    """
    global dataset
    check_init()
    return dataset.create_image_dataset(env, difficulty, trajectory_id, modality, camera_name, transform, num_workers)

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
    """
    global tartanair_data_root
    global traj_reader
    check_init()
    return traj_reader.get_traj_np(env, difficulty, trajectory_id, camera_name)

def evaluate(est_traj, env, difficulty, trajectory_id, modality, camera_name = None):
    """
    Evaluates a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion. In progress.

    Args:
        est_traj (str or list): The estimated trajectory to evaluate. This is speficied as a list of 3D poses in NED and format [x, y, z, qx, qy, qz, qw].
        env (str or list): The environment to evaluate the trajectory from. 
        difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, medium, hard.
        trajectory_id (int or list): The id of the trajectory to evaluate.
        modality (str or list): The modality to evaluate. Valid modalities are: rgb, depth, seg. Default is rgb.
    """
    global evaluator    
    check_init()
    return evaluator.evaluate(est_traj, env, difficulty, trajectory_id, modality, camera_name)