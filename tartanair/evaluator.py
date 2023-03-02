'''
Author: Yorai Shaoul
Date: 2023-03-01

This file contains the evaluator class, which evaluated estimated trajectories against ground truth.
'''

# General imports.
import os
from scipy.spatial.transform import Rotation
from colorama import Fore, Style
import numpy as np

# Local imports.
from .tartanair_module import TartanAirModule
from .reader import TartanAirTrajectoryReader

class TartanAirEvaluator(TartanAirModule):
    def __init__(self, tartanair_data_root):
        '''
        Iterate over the TartanAir dataset. This is only valid for images at this point. So imu and lidar are not supported.
        '''
        super().__init__(tartanair_data_root)
        self.tartanair_data_root = tartanair_data_root
