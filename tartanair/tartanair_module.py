'''
Author: Yorai Shaoul
Date: 2023-02-03
'''

# General imports.
import os
from colorama import Fore, Style

def print_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)

def print_warn(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)

def print_highlight(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)

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

        self.modality_names = ['image', 'depth', 'seg', 'imu', 'lidar', 'flow', 'event']

        self.cam_modalities = ['image', 'depth', 'seg'] # the modalities that support all camera names

        self.flow_camlist = ['lcam_front'] # valid camera name for the flow modality

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

        # for ground dataset
        self.ground_camera_names = ['lcam_front',
            'lcam_left',
            'lcam_right',
            'lcam_back',
            'lcam_top',
            'lcam_bottom',
            'rcam_front',
            'rcam_left',
            'rcam_right',
            'rcam_back',
            'rcam_top',
            'rcam_bottom']
        self.ground_modality_names = ['image', 'meta', 'depth', 'seg', 'imu', 'lidar', 'rosbag', 'sem_pcd', 'rgb_pcd', 'seg_labels']
        self.ground_omni_env_names = ['AbandonedCable',
            'AbandonedFactory',
            'AbandonedFactory2',
            'AbandonedSchool',
            'AmusementPark',
            'AncientTowns',
            'Antiquity3D',
            'BrushifyMoon',
            'CarWelding',
            'CastleFortress',
            'CoalMine',
            'ConstructionSite',
            'CyberPunkDowntown',
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
            'Office',
            'OldBrickHouseDay',
            'OldBrickHouseNight',
            'OldIndustrialCity',
            'OldScandinavia',
            'OldTownFall',
            'OldTownNight',
            'OldTownSummer',
            'OldTownWinter',
            'Prison',
            'Restaurant',
            'Rome',
            'Ruins',
            'SeasideTown',
            'SeasonalForestAutumn',
            'SeasonalForestSpring',
            'SeasonalForestSummerNight',
            'SeasonalForestWinter',
            'SeasonalForestWinterNight',
            'Sewerage',
            'Slaughter',
            'SoulCity',
            'Supermarket',
            'UrbanConstruction',
            'VictorianStreet',
            'WaterMillDay',
            'WaterMillNight',
            'WesternDesertTown']

        self.ground_diff_env_names = ['AbandonedCable',
            'AbandonedFactory2',
            'AbandonedSchool',
            'Antiquity3D',
            'CarWelding',
            'CastleFortress',
            'CoalMine',
            'ConstructionSite',
            'CyberPunkDowntown',
            'Downtown',
            'Fantasy',
            'ForestEnv',
            'Gascola',
            'GothicIsland',
            'GreatMarsh',
            'Hospital',
            'IndustrialHangar',
            'JapaneseAlley',
            'MiddleEast',
            'ModernCityDowntown',
            'ModularNeighborhood',
            'ModularNeighborhoodIntExt',
            'NordicHarbor',
            'OldIndustrialCity',
            'OldScandinavia',
            'OldTownFall',
            'OldTownNight',
            'OldTownSummer',
            'OldTownWinter',
            'Prison',
            'Rome',
            'SeasideTown',
            'SeasonalForestAutumn',
            'SeasonalForestSpring',
            'SeasonalForestSummerNight',
            'SeasonalForestWinter',
            'SeasonalForestWinterNight',
            'Sewerage',
            'Supermarket',
            'UrbanConstruction',
            'WaterMillDay',
            'WaterMillNight']

        self.ground_anymal_env_names = ['Downtown',
            'ForestEnv',
            'Gascola',
            'GreatMarsh',
            'ModernCityDowntown',
            'ModularNeighborhood',
            'NordicHarbor',
            'OldTownFall',
            'OldTownSummer',
            'SeasonalForestAutumn',
            'SeasonalForestSpring',
            'SeasonalForestWinter']
        
        self.ground_version_names = ['omni', 'diff', 'anymal']
    ###############################
    # Data enumeration.
    ###############################
    def enumerate_trajs(self, data_folders = ['Data_easy','Data_hard']):
        '''
        Return a dict:
            res['env0']: ['Data_easy/P000', 'Data_easy/P001', ...], 
            res['env1']: ['Data_easy/P000', 'Data_easy/P001', ...], 
        '''
        env_folders = os.listdir(self.tartanair_data_root)    
        env_folders = [ee for ee in env_folders if os.path.isdir(os.path.join(self.tartanair_data_root, ee))]
        env_folders.sort()
        trajlist = {}
        for env_folder in env_folders:
            env_dir = os.path.join(self.tartanair_data_root, env_folder)
            trajlist[env_folder] = []
            for data_folder in data_folders:
                datapath = os.path.join(env_dir, data_folder)
                if not os.path.isdir(datapath):
                    continue

                trajfolders = os.listdir(datapath)
                trajfolders = [ os.path.join(data_folder, tf) for tf in trajfolders if tf[0]=='P' ]
                trajfolders.sort()
                trajlist[env_folder].extend(trajfolders)
        return trajlist

    def check_env_valid(self, envlist):
        invalid_list = []
        for env in envlist:
            if not env in self.env_names:
                invalid_list.append(env)
        
        if len(invalid_list) == 0:
            return True
        
        print_error(f"The following envs are invalid: {invalid_list}")
        print_warn(f"The available envs are: {self.env_names}")
        return False

    def check_modality_valid(self, modlist, check_ground = False):
        invalid_list = []
        valid_modality = self.modality_names
        if check_ground:
            valid_modality = self.ground_modality_names

        for mod in modlist:
            if not mod in valid_modality:
                invalid_list.append(mod)
        
        if len(invalid_list) == 0:
            return True
        
        print_error(f"The following modalities are invalid: {invalid_list}")
        print_warn(f"The available modalities are: {valid_modality}")
        return False

    def check_camera_valid(self, camlist, check_ground = False):
        invalid_list = []
        valid_camera = self.camera_names
        if check_ground:
            valid_camera = self.ground_camera_names

        for cam in camlist:
            if not cam in valid_camera:
                invalid_list.append(cam)
        
        if len(invalid_list) == 0:
            return True
        
        print_error(f"The following camera names are invalid: {invalid_list}")
        print_warn(f"The available camera names are: {valid_camera}")
        return False

    def check_difficulty_valid(self, difflist):
        invalid_list = []
        for diff in difflist:
            if not diff in self.difficulty_names:
                invalid_list.append(diff)
        
        if len(invalid_list) == 0:
            return True
        
        print_error(f"The following difficulties are invalid: {invalid_list}")
        print_warn(f"The available difficulties are: {self.difficulty_names}")
        return False

    def compile_modality_and_cameraname(self, modalities, camera_names):
        folderlist = []
        for mod in modalities:
            if mod in self.cam_modalities:
                for camname in camera_names:
                    folderstr =  mod + '_' + camname 
                    folderlist.append(folderstr)
            elif mod == 'flow':
                for camname in camera_names:
                    if camname in self.flow_camlist:
                        folderstr =  mod + '_' + camname 
                        folderlist.append(folderstr)
                    else:
                        print_warn("Warn: flow modality doesn't have {}! We only have flow for {}".format(camname, self.flow_camlist))
            elif mod == 'lidar' or mod == 'imu': # for lidar and imu
                folderstr = mod
                folderlist.append(folderstr)
            else:
                if mod != "pose" and mod != "event":
                    print_warn("Warn: note modality {} needs to be processed separately".format(mod))
                
        return folderlist

    def compile_ground_modality_and_cameraname(self, modalities, camera_names):
        '''
        Compile the trajectory, modality and camera name into a list of folder names.
        '''

        folderlist = []
        for mod in modalities:
            if mod in self.cam_modalities:
                for camname in camera_names:
                    folderstr =  mod + '_' + camname 
                    folderlist.append(folderstr)
            elif mod == 'flow':
                print_warn("Warn: flow modality doesn't exist for TartanGround dataset. You need to compute the flow using provided scripts")
            elif mod == 'meta':
                folderlist.append('metadata') # always add metadata folder
            elif mod == 'lidar' or mod == 'imu': # for lidar and imu
                folderstr = mod
                folderlist.append(folderstr)
            elif mod == 'pose':
                print_warn("Warn: All Pose files are provided in the metadata folder so use 'meta' as the modality")
            elif mod == 'rosbag':
                folderlist.append('rosbags')
                
        return folderlist