'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
azure_token = "?sv=2021-10-04&st=2023-03-01T16%3A34%3A50Z&se=2023-03-30T15%3A34%3A00Z&sr=c&sp=rl&sig=LojCTa60TcA9ApMiMofliedxujeX0AOZdoC3O5u5cxg%3D"
 
ta.init(tartanair_data_root, azure_token)

# Download a trajectory.
# env = [
#                 "HQWesternSaloonExposure",
#                 "ModularNeighborhoodIntExt",
#                 "PolarSciFiExposure",
#                 "PrisonExposure",
#                 "RuinsExposure",
#                 "TerrainBlendingExposure",
#                 "UrbanConstructionExposure",
#                 "VictorianStreetExposure",
#                 "WaterMillDayExposure",
#                 "WaterMillNightExposure"
# ]
# ta.download(env = "OldScandinaviaExposure", difficulty = ['easy', 'hard'], trajectory_id = ["P004", "P005", "P003", "P008"],  modality = ['image', 'depth'],  camera_name = ['lcam_front', 'lcam_right', 'lcam_back', 'lcam_left', 'lcam_top', 'lcam_bottom'])

# Can also download via a yaml config file.
ta.download(config = 'download_config.yaml')
