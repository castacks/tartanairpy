import os
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import tartanair as ta


def download_dataset(env, modality, cam_name):
    try:
        # Attempt to download the dataset
        success, filelist = ta.download(env=env,
                                        difficulty=['easy', 'hard'],
                                        modality=modality,
                                        camera_name=cam_name,
                                        unzip=False)
    except Exception as e:
        logging.error(f"Failed to download {env} {modality} {cam_name}: {e}")


def download_all_in_parallel(trajectories, modalities, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for env in trajectories:
            for modality in modalities:
                if modality in ['imu', 'lidar', 'flow']:
                    cam_names = ["lcam_front"]
                else:
                    cam_names = ["lcam_back", "lcam_bottom", "lcam_equirect", "lcam_fish", "lcam_front", 
                                "lcam_left", "lcam_right", "lcam_top", "rcam_back", "rcam_bottom", 
                                "rcam_equirect", "rcam_fish", "rcam_front", "rcam_left", "rcam_right", "rcam_top"]
                for cam_name in cam_names:
                    futures.append(executor.submit(download_dataset, env, modality, cam_name))
                    # Wait for a few seconds to avoid overloading the data server
                    time.sleep(10)
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will re-raise any exceptions caught during the futures' execution


def retry_failed_downloads(error_log_path, num_workers):
    # Read list of environments, modalities and camera names from the error log
    trajectories = []
    modalities = []
    cam_names = []
    with open(error_log_path, 'r') as f:
        for line in f:
            env, modality, cam_name = line.split(" ")[4:7]
            cam_name = cam_name.replace(":", "")
            trajectories.append(env)
            modalities.append(modality)
            cam_names.append(cam_name)
    # Download data in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for data_idx in range(len(trajectories)):
            env = trajectories[data_idx]
            modality = modalities[data_idx]
            cam_name = cam_names[data_idx]
            futures.append(executor.submit(download_dataset, env, modality, cam_name))
            # Wait for a few seconds to avoid overloading the data server
            time.sleep(10)
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will re-raise any exceptions caught during the futures' execution


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download TartanAir datasets.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for TartanAir data.")
    parser.add_argument("--retry_failed", action='store_true', help="Retry failed downloads.")
    parser.add_argument("--error_log_name", type=str, default="error_log.txt", help="Name of the error log file.")
    parser.add_argument("--error_log_path", type=str, default="", help="Path to store the error log file.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of workers for parallel downloads.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    error_log_path = args.error_log_path if args.error_log_path else '.'
    error_log_file = os.path.join(error_log_path, args.error_log_name)
    
    # Create the log directory if it doesn't exist
    if not os.path.exists(error_log_path):
        os.makedirs(error_log_path)
    
    # Setup logging
    logging.basicConfig(filename=error_log_file, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')
    
    # Initialize TartanAir Module.
    tartanair_data_root = args.data_root
    ta.init(tartanair_data_root)
    
    # Define Trajectories and Modalities to be downloaded
    trajectories = [
        "AbandonedCable", "AbandonedFactory", "AbandonedFactory2", "AbandonedSchool", 
        "AmericanDiner", "AmusementPark", "AncientTowns", "Antiquity3D", "Apocalyptic", 
        "ArchVizTinyHouseDay", "ArchVizTinyHouseNight", "BrushifyMoon", "CarWelding", 
        "CastleFortress", "CoalMine", "ConstructionSite", "CountryHouse", "CyberPunkDowntown", 
        "Cyberpunk", "DesertGasStation", "Downtown", "EndofTheWorld", "FactoryWeather", "Fantasy", 
        "ForestEnv", "Gascola", "GothicIsland", "GreatMarsh", "HQWesternSaloon", "HongKong", "Hospital", 
        "House", "IndustrialHangar", "JapaneseAlley", "JapaneseCity", "MiddleEast", "ModUrbanCity", 
        "ModernCityDowntown", "ModularNeighborhood", "ModularNeighborhoodIntExt", "NordicHarbor", 
        "Ocean", "Office", "OldBrickHouseDay", "OldBrickHouseNight", "OldIndustrialCity", "OldScandinavia", 
        "OldTownFall", "OldTownNight", "OldTownSummer", "OldTownWinter", "PolarSciFi", "Prison", "Restaurant", 
        "RetroOffice", "Rome", "Ruins", "SeasideTown", "SeasonalForestAutumn", "SeasonalForestSpring", 
        "SeasonalForestSummerNight", "SeasonalForestWinter", "SeasonalForestWinterNight", "Sewerage", 
        "ShoreCaves", "Slaughter", "SoulCity", "Supermarket", "TerrainBlending", "UrbanConstruction", 
        "VictorianStreet", "WaterMillDay", "WaterMillNight", "WesternDesertTown"
    ]
    modalities = ['imu', 'lidar', 'flow', 'image', 'depth', 'seg']
    
    download_all_in_parallel(trajectories, modalities, args.num_workers)
    
    if args.retry_failed:
        retry_failed_downloads(error_log_file, args.num_workers)
