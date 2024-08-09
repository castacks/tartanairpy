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

        self.camera_directions = ["front", "right", "back", "left", "top", "bottom"]

        self.modality_names = ['image', 'depth', 'seg', 'imu', 'lidar', 'flow']

        self.env_names = [
            'AbandonedCable', 
            'AbandonedFactory', 
            'AbandonedFactory2', 
            'AbandonedSchool', 
            'AmericanDiner', 
            'AmusementPark', 
            'AncientTowns', 
            'Antiquity3D', 
            'Apocalyptic', 
            'ArchVizTinyHouseDay', 
            'ArchVizTinyHouseNight', 
            'BrushifyMoon', 
            'CarWelding', 
            'CastleFortress', 
            'CoalMine', 
            'ConstructionSite', 
            'CountryHouse', 
            'CyberPunkDowntown', 
            'Cyberpunk', 
            'DesertGasStation', 
            'Downtown', 
            'EndofTheWorld', 
            'FactoryWeather', 
            'Fantasy', 
            'ForestEnv', 
            'Gascola', 
            'GothicIsland', 
            'GreatMarsh', 
            'HQWesternSaloon', 
            'HongKong', 
            'Hospital', 
            'House', 
            'IndustrialHangar', 
            'JapaneseAlley', 
            'JapaneseCity', 
            'MiddleEast', 
            'ModUrbanCity', 
            'ModernCityDowntown', 
            'ModularNeighborhood', 
            'ModularNeighborhoodIntExt', 
            'NordicHarbor', 
            'Ocean', 
            'Office', 
            'OldBrickHouseDay', 
            'OldBrickHouseNight', 
            'OldIndustrialCity', 
            'OldScandinavia', 
            'OldTownFall', 
            'OldTownNight', 
            'OldTownSummer', 
            'OldTownWinter', 
            'PolarSciFi', 
            'Prison', 
            'Restaurant', 
            'RetroOffice', 
            'Rome', 
            'Ruins', 
            'SeasideTown', 
            'SeasonalForestAutumn', 
            'SeasonalForestSpring', 
            'SeasonalForestSummerNight', 
            'SeasonalForestWinter', 
            'SeasonalForestWinterNight', 
            'Sewerage', 
            'ShoreCaves', 
            'Slaughter', 
            'SoulCity', 
            'Supermarket', 
            'TerrainBlending', 
            'UrbanConstruction', 
            'VictorianStreet', 
            'WaterMillDay', 
            'WaterMillNight', 
            'WesternDesertTown',
        ]

        self.difficulty_names = ['easy', 'hard']
