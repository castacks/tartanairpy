# General imports.
import os
import numpy as np

# Local imports.
from .downloader import TartanAirDownloader
from .dataset import TartanAirDataset
from .customizer import TartanAirCustomizer
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
    Initialize the TartanAir toolbox.

    :param tartanair_root: The root directory of the TartanAir dataset.
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
    :param camera_name: The camera name to download. Can be a list of camera names. Default is will include all. Modalities IMU and LIDAR do not need camera names specified.
    :type camera_name: str or list
    :param config: Optional. Path to a yaml file containing the download configuration. If a config file is provided, the other arguments will be ignored.
    :type config: str
    """

    global downloader
    check_init()
    downloader.download(env, difficulty, trajectory_id, modality, camera_name, config)

def customize(env, difficulty, trajectory_id, modality = 'image', new_camera_models_params = [{}], num_workers = 1):
    """"
    Synthesizes data in new camera-models from the TartanAir dataset.
    
    :param env: The environment to customize. Can be a list of environments.
    """
    global customizer
    check_init()
    customizer.customize(env, difficulty, trajectory_id, modality, new_camera_models_params, num_workers=num_workers)

def create_image_dataset(env, difficulty = None, trajectory_id = None, modality = None, camera_name = None, transform = None):
    """
    Return the relevant data from the TartanAir dataset.
    This dataset will only handle image data in modalities such as 'image', depth, and segmentation.
    """
    global dataset
    check_init()
    return dataset.create_image_dataset(env, difficulty, trajectory_id, modality, camera_name, transform)

def list_envs():
    """
    List all the environments in the TartanAir dataset.
    """
    global lister    
    check_init()
    return lister.list_envs()

def visualize(env, difficulty, trajectory_id, modality, camera_name = None):
    """
    Visualizes a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

    Args:
        env (str or list): The environment to visualize the trajectory from. 
        difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, medium, hard.
        trajectory_id (int or list): The id of the trajectory to visualize.
        modality (str or list): The modality to visualize. Valid modalities are: rgb, depth, seg. Default is rgb.
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
    """
    global iterator
    global tartanair_data_root
    check_init()
    return iterator.get_iterator(env, difficulty, trajectory_id, modality, camera_name)

def get_traj_np(env, difficulty, trajectory_id, camera_name = None):
    """
    Returns the trajectory as a numpy array.
    """
    global tartanair_data_root
    global traj_reader
    check_init()
    return traj_reader.get_traj_np(env, difficulty, trajectory_id, camera_name)

def evaluate(est_traj, env, difficulty, trajectory_id, modality, camera_name = None):
    """
    Evaluates a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

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