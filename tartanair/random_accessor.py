'''
Author: Yuchen Zhang
Date: 2024-06-25

This class provides conveient random access throughout the entire dataset

'''

# General imports.
import torch
import os
import pandas as pd
from colorama import Fore, Back, Style
import concurrent.futures
import threading
from tqdm import tqdm
import json

# Local imports.
from .customizer import TartanAirCustomizer
from .tartanair_module import TartanAirModule
from .data_cacher.MultiDatasets import MultiDatasets
from .image_resampling.mvs_utils.shape_struct import ShapeStruct
from .image_resampling.mvs_utils.camera_models import CameraModel

class TartanAirRandomAccessor(TartanAirModule):
    '''
    The TartanAirDataset class contains the _information_ about the TartanAir dataset, and implements no functionality. All functionalities are implemented in inherited classes like the TartanAirDownloader, and the interface is via the TartanAir class.   
    '''
    def __init__(self, tartanair_data_root, trajectory_reader):
        # Call the parent class constructor.
        super(TartanAirRandomAccessor, self).__init__(tartanair_data_root)

        self.tartanair_data_root = tartanair_data_root
        self.trajectory_reader = trajectory_reader
        self.path_dict = None

        mapdict = {
            'AbandonedCableExposure':               'AbandonedCable',
            'AbandonedFactoryExposure':             'AbandonedFactory',
            'AbandonedSchoolExposure':              'AbandonedSchool',
            'abandonfactory2':                      'AbandonedFactory2',
            'AmericanDinerExposure':                'AmericanDiner',
            'AmusementParkExposure':                'AmusementPark',
            'AncientTownsExposure':                 'AncientTowns',
            'Antiquity3DExposure':                  'Antiquity3D',
            'ApocalypticExposure':                  'Apocalyptic',
            'ArchVizTinyHouseDayExposure':          'ArchVizTinyHouseDay',
            'ArchVizTinyHouseNightExposure':        'ArchVizTinyHouseNight',
            'BrushifyMoonExposure':                 'BrushifyMoon',
            'CarweldingExposure':                   'CarWelding',
            'CastleFortressExposure':               'CastleFortress',
            'coalmine':                             'CoalMine',
            'ConstructionSite':                     'ConstructionSite',
            'CountryHouseExposure':                 'CountryHouse',
            'CyberPunkDowntownExposure':            'CyberPunkDowntown',
            'CyberpunkExposure':                    'Cyberpunk',
            'DesertGasStationExposure':             'DesertGasStation',
            'DowntownExposure':                     'Downtown',
            'EndofTheWorld':                        'EndofTheWorld',
            'FactoryWeatherExposure':               'FactoryWeather',
            'FantasyExposure':                      'Fantasy',
            'ForestEnvExposure':                    'ForestEnv',
            'GascolaExposure':                      'Gascola',
            'GothicIslandExposure':                 'GothicIsland',
            'GreatMarshExposure':                   'GreatMarsh',
            'HongKong':                             'HongKong',
            'HospitalExposure':                     'Hospital',
            'HouseExposure':                        'House',
            'HQWesternSaloonExposure':              'HQWesternSaloon',
            'IndustrialHangarExposure':             'IndustrialHangar',
            'JapaneseAlleyExposure':                'JapaneseAlley',
            'JapaneseCityExposure':                 'JapaneseCity',
            'MiddleEastExposure':                   'MiddleEast',
            'ModernCityDowntownExposure':           'ModernCityDowntown',
            'ModularNeighborhoodExposure':          'ModularNeighborhood',
            'ModularNeighborhoodIntExt':            'ModularNeighborhoodIntExt',
            'ModUrbanCityExposure':                 'ModUrbanCity',
            'NordicHarborExposure':                 'NordicHarbor',
            'OceanExposure':                        'Ocean',
            'OfficeExposure':                       'Office',
            'OldBrickHouseDayExposure':             'OldBrickHouseDay',
            'OldBrickHouseNightExposure':           'OldBrickHouseNight',
            'OldIndustrialCityExposure':            'OldIndustrialCity',
            'OldScandinaviaExposure':               'OldScandinavia',
            'OldTownFallExposure':                  'OldTownFall',
            'OldTownNightExposure':                 'OldTownNight',
            'OldTownSummerExposure':                'OldTownSummer',
            'OldTownWinterExposure':                'OldTownWinter',
            'PolarSciFiExposure':                   'PolarSciFi',
            'PrisonExposure':                       'Prison',
            'RestaurantExposure':                   'Restaurant',
            'RetroOfficeExposure':                  'RetroOffice',
            'RomeExposure':                         'Rome',
            'RuinsExposure':                        'Ruins',
            'SeasideTownExposure':                  'SeasideTown',
            'SeasonalForestAutumnExposure':         'SeasonalForestAutumn',
            'SeasonalForestSpringExposure':         'SeasonalForestSpring',
            'SeasonalForestSummerNightExposure':    'SeasonalForestSummerNight',
            'SeasonalForestWinterExposure':         'SeasonalForestWinter',
            'SeasonalForestWinterNightExposure':    'SeasonalForestWinterNight',
            'SewerageExposure':                     'Sewerage',
            'ShoreCavesExposure':                   'ShoreCaves',
            'Slaughter':                            'Slaughter',
            'SoulCityExposure':                     'SoulCity',
            'SupermarketExposure':                  'Supermarket',
            'TerrainBlendingExposure':              'TerrainBlending',
            'UrbanConstructionExposure':            'UrbanConstruction',
            'VictorianStreetExposure':              'VictorianStreet',
            'WaterMillDayExposure':                 'WaterMillDay',
            'WaterMillNightExposure':               'WaterMillNight',
            'WesternDesertTownExposure':            'WesternDesertTown'
        }
        self.TARTANAIR_ENV_LIST = list(mapdict.values())

        # read the trajectories and their length

        df_data = []
        for env in self.TARTANAIR_ENV_LIST:
            for difficulty in ["easy", "hard"]:

                folderpath = os.path.join(tartanair_data_root, env, "Data_%s" % difficulty)

                if not os.path.exists(folderpath):
                    continue

                traj_dirs = os.listdir(folderpath)

                for traj_dir in traj_dirs:

                    data_entry = {
                        "env": env,
                        "difficulty": difficulty,
                        "path_id": traj_dir,
                        "folder_path": os.path.join(folderpath, traj_dir),
                        "num_frames": len(os.listdir(os.path.join(folderpath, traj_dir, "image_lcam_front")))
                    }

                    df_data.append(data_entry)

        self.TARTANAIR_METADATA_DF = pd.DataFrame(df_data)

        self.CAMERA_DIRECTION_LIST = ["front", "right", "back", "left", "top", "bottom"]

        self.GLASS_PROBLEM_ENVS = [
            "NordicHarbor", "AbandonedCable", "AbandonedFactory2", "GothicIsland", "Slaughter", "VictorianStreet", "Prison", "WesternDesertTown", "Office", "OldBrickHouseDay", "OldBrickHouseNight"
        ]

        self.reader = TartanAirCustomizer(tartanair_data_root)

        self.MODALITY_TO_READER = {"image": self.reader.read_rgb, "depth": self.reader.read_dist, "seg": self.reader.read_seg}
        self.MODALITY_TO_INTERPOLATION = {"image": "linear", "seg": "nearest", "depth": "blend"}
        self.MODALITY_TO_WRITER = {"image": self.reader.write_as_is, "seg": self.reader.write_as_is, "depth": self.reader.write_float_depth}

    def cache_tartanair_pose(self):
        path_dict = {}
        print("Caching pose...")

        # Prepare arguments for parallel execution
        args_list = []
        for _, row in self.TARTANAIR_METADATA_DF.iterrows():
            for camera_side in ["lcam", "rcam"]:
                args_list.append((row["env"], row["difficulty"], row["path_id"], camera_side))
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Using tqdm for progress bar
            results = list(tqdm(executor.map(self.get_trajectory_pose_wrapper, args_list), total=len(args_list)))
        
        # Collecting results
        for key, value in results:
            path_dict[key] = value

        self.path_dict = path_dict
    
    def get_trajectory_pose_wrapper(self, args):
        env, difficulty, path_id, camera_side = args
        traj_pose = self.get_trajectory_pose({
            "env": env,
            "difficulty": difficulty,
            "id": path_id,
            "cam_side": camera_side
        })

        return (env, difficulty, path_id, camera_side), torch.from_numpy(traj_pose["front"])
    
    def get_trajectory_pose(self, traj_idx):
        dir_path_map = {}
        for dir_ in self.CAMERA_DIRECTION_LIST:
            cam_name = traj_idx["cam_side"] + "_" + dir_
            
            pose_arr = self.trajectory_reader.get_traj_np(traj_idx["env"], traj_idx["difficulty"], traj_idx["id"], camera_name=cam_name)

            dir_path_map[dir_] = pose_arr

        return dir_path_map

    def get_metadata_df(self):
        return self.TARTANAIR_METADATA_DF
    
    def load_image(self, traj_folder, modality, direction, frame_idx, traj_idx, results):
        folder_name = "%s_%s_%s" % (modality, traj_idx["cam_side"], direction)
        if modality == "image":
            img_filename = "%06d_%s_%s.png" % (frame_idx, traj_idx["cam_side"], direction)
        elif modality == "depth":
            img_filename = "%06d_%s_%s_%s.png" % (frame_idx, traj_idx["cam_side"], direction, modality)
        
        img_path = os.path.join(traj_folder, folder_name, img_filename)
        results[(modality, direction)] = self.MODALITY_TO_READER[modality](img_path)

    def get_front_cam_NED_pose(self, traj_idx, frame_idx):

        assert self.path_dict is not None, "Please cache the pose first by calling cache_tartanair_pose()"

        env = traj_idx["env"]
        difficulty = traj_idx["difficulty"]
        path_id = traj_idx["id"]
        cam_side = traj_idx["cam_side"]

        pose = self.path_dict[(env, difficulty, path_id, cam_side)][frame_idx]

        return pose

    def get_cubemap_images_parallel(self, traj_idx, frame_idx):
        traj_folder = os.path.join(self.tartanair_data_root, traj_idx["env"], "Data_" + traj_idx["difficulty"], traj_idx["id"])
        modalities = ["image", "depth"]
        threads = []
        results = {}

        for modality in modalities:
            for direction in self.camera_directions:
                thread = threading.Thread(target=self.load_image, args=(traj_folder, modality, direction, frame_idx, traj_idx, results))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        cubemap_images = {modality: {} for modality in modalities}
        for (modality, direction), image in results.items():
            cubemap_images[modality][direction] = image

        return cubemap_images
    
    def generate_camera_model_object_from_config(self, config) -> CameraModel:

        # Create a deep copy.
        new_cam_model_params_copy = json.loads(json.dumps(config))

        # The name of the new camera model that is used to find the camera model class.
        new_cam_model_name = new_cam_model_params_copy['name']

        # The new camera model object. We need to convert the width and height to a ShapeStruct.
        new_cam_model_params_copy['params']['shape_struct'] = ShapeStruct(H=new_cam_model_params_copy['params']['height'], W=new_cam_model_params_copy['params']['width'])
        del new_cam_model_params_copy['params']['height']
        del new_cam_model_params_copy['params']['width']
        
        CAMERA_MODEL_NAME_TO_CLASS_MAP = self.reader.camera_model_name_to_class

        # Create the new camera model object.
        if not(new_cam_model_name in CAMERA_MODEL_NAME_TO_CLASS_MAP):
            print("class name %s not found when processing %s" % (new_cam_model_name, new_cam_model_name))
            exit()

        new_cam_model_object = CAMERA_MODEL_NAME_TO_CLASS_MAP[new_cam_model_name](**new_cam_model_params_copy['params'])

        return new_cam_model_object