# General imports.
import os
import numpy as np

# Local imports.
from .downloader import TartanAirDownloader
from .dataset import TartanAirDataset
from .customizer import TartanAirCustomizer

# TODO(yoraish):
'''
[ ] Auto install azcopy.
[ ] Check that the inputs are valid.
[ ] Add a function to customize a trajectory.
[ ] Verify download of flow.
[ ] Remove spamming text.
'''
print("TartanAir toolbox initialized.")

tartanair_data_root = ""
downloader = None
dataset = None
customizer = None

def init(tartanair_data_root_input, azure_token = None):
    """
    Initialize the TartanAir toolbox.
    """

    # Until the official release, ask for an azure token.
    if not azure_token:
        print("TEST ERROR: azure_token is None. Please pass a valid azure_token to the init function. This will no longer be necessary when TartanAir will be officially released.")

    global tartanair_data_root
    tartanair_data_root = tartanair_data_root_input

    global downloader
    # If a token is provided, use it. Otherwise, let the downloader use self.azure_token from the parent class.
    downloader = TartanAirDownloader(tartanair_data_root, azure_token = azure_token)

    global dataset
    dataset = TartanAirDataset(tartanair_data_root)

    global customizer
    customizer = TartanAirCustomizer(tartanair_data_root)
    

def download(env = [], difficulty = [], trajectory_id = [], modality = 'image', camera_name = 'lcam_front'):
    """
    Download the relevant data from the TartanAir dataset.
    """
    global downloader
    global tartanair_data_root

    if not tartanair_data_root:
        raise Exception("TartanAir toolbox not initialized. Please call tartanair.init(tartanair_data_root) first.")
    else:
        downloader.download(env, difficulty, trajectory_id, modality, camera_name)

def customize(env, difficulty, trajectory_id, modality = 'image', new_camera_models_params = [{}], num_workers = 1):
    """"
    Synthesizes data in new camera-models from the TartanAir dataset.
    """
    global customizer
    global tartanair_data_root

    if not tartanair_data_root:
        raise Exception("TartanAir toolbox not initialized. Please call tartanair.init(tartanair_data_root) first.")
    customizer.customize(env, difficulty, trajectory_id, modality, new_camera_models_params, num_workers=num_workers)

def create_image_dataset(env, difficulty, trajectory_id, modality = 'image', camera_name = 'lcam_front', transform = None):
    """
    Return the relevant data from the TartanAir dataset.
    This dataset will only handle image data in modalities such as 'image', depth, and segmentation.
    """
    global dataset
    global tartanair_data_root
    
    if not tartanair_data_root:
        raise Exception("TartanAir toolbox not initialized. Please call tartanair.init(tartanair_data_root) first.")

    return dataset.create_image_dataset(env, difficulty, trajectory_id, modality, camera_name, transform)