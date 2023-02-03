'''
Author: Yorai Shaoul
Date: 2023-02-03

Test file for the TartanAir dataset toolbox.
'''

# General imports.
import os
import unittest
import sys

# Local imports.
sys.path.append('../src/')
from tartanair.tartanair import TartanAir

class TestTartanAir(unittest.TestCase):
    def test_download(self):
        # Create a TartanAir object.
        tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
        ta = TartanAir(tartanair_data_root)

        # Download a trajectory.
        ta.download(env = 'AmericanDinerExposure', difficulty = 'easy', trajectory_id = ['P000', 'P003'], modality = ['imu', 'image'], camera_name = ['lcam_fish'])

        # Check that the trajectory was downloaded.
        self.assertTrue(os.path.exists(tartanair_data_root + '/AmericanDinerExposure/Data_easy/P000/imu/lcam_fish'))

        # Delete the trajectory.
        # os.system('rm -rf ' + tartanair_data_root + '/AmericanDinerExposure/Data_easy/P000')
# 
if __name__ == '__main__':
    unittest.main()
