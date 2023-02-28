'''
Author: Yorai Shaoul
Date: 2023-02-03
'''

# General imports.
import os

class TartanAirModule():
    '''
    The main building block of the TartanAir toolbox. This class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root):
        # Check if tartanair_data_root exists, otherwise create it.
        if not os.path.exists(tartanair_data_root):
            os.makedirs(tartanair_data_root)        
        self.tartanair_data_root = tartanair_data_root

        # The allowed names for the cameras, modalities, environments and difficulties.
        self.camera_names = ['lcam_front',
                            'lcam_left',
                            'lcam_right',
                            'lcam_back',
                            'lcam_top',
                            'lcam_bottom',
                            'lcam_fish',
                            'lcam_equirect',
                            'rcam_front',
                            'rcam_left',
                            'rcam_right',
                            'rcam_back',
                            'rcam_top',
                            'rcam_bottom',
                            'rcam_fish',
                            'rcam_equirect']

        self.modality_names = ['image', 'depth', 'seg', 'imu', 'lidar']

        self.env_names = [
                        "AbandonedCableExposure",
                        "OldScandinaviaExposure",
                        "ShoreCavesExposure",
                        "PolarSciFiExposure",
                        "PrisonExposure",
                        "AmericanDinerExposure",
                        "ArchVizTinyHouseDayExposure",
                        "DesertGasStationExposure",
                        "GothicIslandExposure",
                        "ArchVizTinyHouseNightExposure",
                        "TerrainBlendingExposure",
                        "VictorianStreetExposure",
                        "OldIndustrialCityExposure",
                        "SupermarketExposure",
                        "WaterMillDayExposure",
                        "UrbanConstructionExposure",
                        "WaterMillNightExposure",
                        "ModularNeighborhoodIntExt",
                        "ConstructionSite",
                        "ForestEnvExposure",
                        "NordicHarborExposure",
                        "HQWesternSaloonExposure",
        ]

        self.difficulty_names = ['easy', 'hard']
