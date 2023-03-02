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

def init(tartanair_data_root_input, azure_token = None):
    """
    Initialize the TartanAir toolbox.
    """

    global tartanair_data_root
    tartanair_data_root = tartanair_data_root_input

    global downloader
    # If a token is provided, use it. Otherwise, let the downloader use self.azure_token from the parent class.
    downloader = TartanAirDownloader(tartanair_data_root, azure_token = azure_token)

    global dataset
    dataset = TartanAirDataset(tartanair_data_root)

    global customizer
    customizer = TartanAirCustomizer(tartanair_data_root)

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
    

def download(env = [], difficulty = [], trajectory_id = [], modality = 'image', camera_name = 'lcam_front', config = None):
    """
    Download the relevant data from the TartanAir dataset.
    """
    global downloader
    check_init()
    downloader.download(env, difficulty, trajectory_id, modality, camera_name, config)

def customize(env, difficulty, trajectory_id, modality = 'image', new_camera_models_params = [{}], num_workers = 1):
    """"
    Synthesizes data in new camera-models from the TartanAir dataset.
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