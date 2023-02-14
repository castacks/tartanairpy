# General imports.
import os
import numpy as np

# Local imports.
from .tartanair_module import TartanAirModule
from .downloader import TartanAirDownloader
from .dataset import TartanAirDataset

# TODO(yoraish):
'''
[ ] Auto install azcopy.
[ ] Check that the inputs are valid.
[ ] Add a function to customize a trajectory.
[ ] Verify download of flow.
[ ] Remove spamming text.
'''


class TartanAir(TartanAirModule):

    def __init__(self, tartanair_data_root):
        # Initialize the TartanAirModule.
        super().__init__(tartanair_data_root)

        # Modules.
        self.downloader = TartanAirDownloader(tartanair_data_root)
        self.dataset = TartanAirDataset(tartanair_data_root)
    
    def download(self, env, difficulty, trajectory_id, modality = 'image', camera_name = 'lcam_front'):
        """
        Download the relevant data from the TartanAir dataset.
        """
        self.downloader.download(env, difficulty, trajectory_id, modality, camera_name)

    def customize(self, env, difficulty, trajectory_id, modality = 'image', camera_name = 'lcam_front', R_raw_new = np.eye(4), allow_download = True):
        """"
        Checks if the trajectory exists locally. Otherwise, downloads it.
        Use the relevant raw trajectory files, and create a custom image, and delete those the downloaded raw files.
        """
        pass

    def create_image_dataset(self, env, difficulty, trajectory_id, modality = 'image', camera_name = 'lcam_front', transform = None):
        """
        Return the relevant data from the TartanAir dataset.
        This dataset will only handle image data in modalities such as 'image', depth, and segmentation.
        """
        return self.dataset.create_image_dataset(env, difficulty, trajectory_id, modality, camera_name, transform)