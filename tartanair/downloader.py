'''
Author: Yorai Shaoul
Date: 2023-02-03

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os
# import sys
from copy import copy
import re
from colorama import Fore, Style
import yaml

# Local imports.
from .tartanair_module import TartanAirModule, print_error, print_highlight, print_warn
from os.path import isdir, isfile, join
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

class AirLabDownloader(object):
    def __init__(self, bucket_name = 'tartanair2') -> None:
        from minio import Minio

        if bucket_name == 'tartanair2':
            endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"
            # public key (for downloading): 
            access_key = "4e54CkGDFg2RmPjaQYmW"
            secret_key = "mKdGwketlYUcXQwcPxuzinSxJazoyMpAip47zYdl"
        elif bucket_name == 'tartanground':
            endpoint_url = "airlab-share-02.andrew.cmu.edu:9000"
            # public key (for downloading): 
            access_key = "nu8ylTnuSBKmHtPgj6xB"
            secret_key = "3njOB53mTzrvMRkBEm8MN8GvGrKuKvtwg1Bh4QLS"
        else:
            print_error("Error: Invalid bucket name. Please use 'tartanair2' or 'tartanground'.")
            return
        
        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    def download(self, filelist, targetfilelist):
        success_source_files, success_target_files = [], []
        for source_file_name, target_file_name in zip(filelist, targetfilelist):
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                continue
                # return False, success_source_files, success_target_files

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
            success_source_files.append(source_file_name)
            success_target_files.append(target_file_name)
        
        print(len(targetfilelist), len(success_target_files))
        if len(success_target_files) == len(targetfilelist):
            return True, success_source_files, success_target_files
        else:
            return False, success_source_files, success_target_files

class CloudFlareDownloader(object):
    def __init__(self, bucket_name = "tartanair-v2") -> None:
        import boto3
        access_key = "f1ae9efebbc6a9a7cebbd949ba3a12de"
        secret_key = "0a21fe771089d82e048ed0a1dd6067cb29a5666bf4fe95f7be9ba6f72482ec8b"
        endpoint_url = "https://0a585e9484af268a716f8e6d3be53bbc.r2.cloudflarestorage.com"

        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      endpoint_url=endpoint_url)

    def download(self, filelist, targetfilelist):
        """
        Downloads a file from Cloudflare R2 storage using S3 API.

        Args:
        - filelist (list): List of names of the files in the bucket you want to download
        - destination_path (str): Path to save the downloaded file locally
        - bucket_name (str): The name of the Cloudflare R2 bucket

        Returns:
        - str: A message indicating success or failure.
        """

        from botocore.exceptions import NoCredentialsError, ClientError
        success_source_files, success_target_files = [], []
        for source_file_name, target_file_name in zip(filelist, targetfilelist):
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, success_source_files, success_target_files
            try:
                print(f"  Downloading {source_file_name} from {self.bucket_name}...")
                self.s3.download_file(self.bucket_name, source_file_name, target_file_name)
                print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
                success_source_files.append(source_file_name)
                success_target_files.append(target_file_name)
            except ClientError:
                print_error(f"Error: The file {source_file_name} was not found in the bucket {self.bucket_name}.")
                return False, success_source_files, success_target_files
            except NoCredentialsError:
                print_error("Error: Credentials not available.")
                return False, success_source_files, success_target_files
            except Exception:
                print_error("Error: Failed for some reason.")
                return False, success_source_files, success_target_files
        return True, success_source_files, success_target_files

    def get_all_s3_objects(self):
        continuation_token = None
        content_list = []
        while True:
            list_kwargs = dict(MaxKeys=1000, Bucket = self.bucket_name)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = self.s3.list_objects_v2(**list_kwargs)
            content_list.extend(response.get('Contents', []))
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')
        return content_list

class TartanAirDownloader(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

        # self.downloader = CloudFlareDownloader()
        self.downloader = AirLabDownloader(bucket_name = 'tartanair2')

    def generate_filelist(self, envs, difficulties, modalities, camera_names): 
        '''
        Return a list of zipfiles to be downloaded
        Example: 
        [
            "abandonedfactory/Data_easy/depth_lcam_equirect.zip",
            "abandonedfactory/Data_easy/flow_lcam_front.zip",
            ...
        ]

        '''
        zipfilelist = []
        for env in envs: 
            envstr = env + '/'
            for difficulty in difficulties:
                diffstr = envstr + 'Data_' + difficulty + '/'
                folderlist = self.compile_modality_and_cameraname(modalities, camera_names)
                zipfiles = [diffstr + fl + '.zip' for fl in folderlist]
                zipfilelist.extend(zipfiles)
            if 'event' in modalities:
                zipfile = envstr + 'Data_easy/events.zip' # hard code here, only easy events are available
                zipfilelist.append(zipfile)

        return zipfilelist

    def doublecheck_filelist(self, filelist, gtfile=''):
        '''
        '''
        with open(gtfile, 'r') as f:
            lines = f.readlines()

        filesizedict = {}
        for line in lines:
            name, size, _ = line.split(' ')
            filesizedict[name] = float(size)

        totalsize = 0
        for ff in filelist:
            if not ff in filesizedict:
                print_error("Error: invalid file {}".format(ff))
                return False
            totalsize += filesizedict[ff]

        print("*****")
        print("The following {} files are going to be downloaded".format(len(filelist)))
        for ff in filelist:
            print("  - ", ff)
        print_highlight("The total size is {} GB! Please make sure you have enough space!".format(totalsize))
        print("*****")
        return True

    def unzip_files(self, zipfilelist):
        print_warn('Note unzipping will overwrite existing files ...')
        for zipfile in zipfilelist:
            if not isfile(zipfile) or (not zipfile.endswith('.zip')):
                print_error("The zip file is missing {}".format(zipfile))
                return False
            print('  Unzipping {} ...'.format(zipfile))
            cmd = 'unzip -q -o ' + zipfile + ' -d ' + self.tartanair_data_root
            os.system(cmd)
        print_highlight("Unzipping Completed! ")

    def refine_parameters(self, env, difficulty, modality, camera_name, unzip, config):
        if config is not None:
            print("Using config file: {}".format(config))
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

            # Update the parameters.
            env = config['env']
            difficulty = config['difficulty']
            modality = config['modality']
            camera_name = config['camera_name']
            unzip = config['unzip']
        
        # Check that the inputs are all lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(difficulty, list):
            difficulty = [difficulty]
        if not isinstance(modality, list):
            modality = [modality]
        if not isinstance(camera_name, list):
            camera_name = [camera_name]

        # download all if not specified
        if len(env) == 0:
            env = self.env_names
        if len(difficulty) == 0:
            difficulty = self.difficulty_names
        if len(modality) == 0:
            modality = self.modality_names
        if len(camera_name) == 0:
            camera_name = self.camera_names

        return env, difficulty, modality, camera_name, unzip    
            
    def download(self, env = [], difficulty = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, **kwargs):
        """
        Downloads a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to download the trajectory from. Valid envs are: AbandonedCable, AbandonedFactory, AbandonedFactory2, AbandonedSchool, AmericanDiner, AmusementPark, AncientTowns, Antiquity3D, Apocalyptic, ArchVizTinyHouseDay, ArchVizTinyHouseNight, BrushifyMoon, CarWelding, CastleFortress, CoalMine, ConstructionSite, CountryHouse, CyberPunkDowntown, Cyberpunk, DesertGasStation, Downtown, EndofTheWorld, FactoryWeather, Fantasy, ForestEnv, Gascola, GothicIsland, GreatMarsh, HQWesternSaloon, HongKong, Hospital, House, IndustrialHangar, JapaneseAlley, JapaneseCity, MiddleEast, ModUrbanCity, ModernCityDowntown, ModularNeighborhood, ModularNeighborhoodIntExt, NordicHarbor, Ocean, Office, OldBrickHouseDay, OldBrickHouseNight, OldIndustrialCity, OldScandinavia, OldTownFall, OldTownNight, OldTownSummer, OldTownWinter, PolarSciFi, Prison, Restaurant, RetroOffice, Rome, Ruins, SeasideTown, SeasonalForestAutumn, SeasonalForestSpring, SeasonalForestSummerNight, SeasonalForestWinter, SeasonalForestWinterNight, Sewerage, ShoreCaves, Slaughter, SoulCity, Supermarket, TerrainBlending, UrbanConstruction, VictorianStreet, WaterMillDay, WaterMillNight, WesternDesertTown. 
            difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, hard.
            modality (str or list): The modality to download. Valid modalities are: image, depth, seg, imu, lidar, flow. Default is image.
            camera_name (str or list): The name of the camera to download. Valid names are: lcam_back, lcam_bottom, lcam_equirect, lcam_fish, lcam_front, lcam_left, lcam_right, lcam_top, rcam_back, rcam_bottom, rcam_equirect, rcam_fish, rcam_front, rcam_left, rcam_right, rcam_top
        
        Note: 
            for imu and lidar, no camera_name needs to be specified. 
            for flow, only lcam_front is available. 
        """

        env, difficulty, modality, camera_name, unzip = self.refine_parameters(env, difficulty, modality, camera_name, unzip, config)
            
        # Check that the environments are valid.
        if not self.check_env_valid(env):
            return False
        # Check that the modalities are valid
        if not self.check_modality_valid(modality):
            return False
        # Check that the difficulty are valid
        if not self.check_difficulty_valid(difficulty):
            return False
        # Check that the camera names are valid
        if not self.check_camera_valid(camera_name):
            return False
        
        # download all if not specified
        if len(env) == 0:
            env = self.env_names
        if len(difficulty) == 0:
            difficulty = self.difficulty_names
        if len(modality) == 0:
            modality = self.modality_names
        if len(camera_name) == 0:
            camera_name = self.camera_names

        zipfilelist = self.generate_filelist(env, difficulty, modality, camera_name)
        # import ipdb;ipdb.set_trace()
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_files.txt'
        if not self.doublecheck_filelist(zipfilelist, gtfile = gtfile):
            return False

        # generate the target file list: 
        targetfilelist = [join(self.tartanair_data_root, zipfile.replace('/', '_')) for zipfile in zipfilelist]
        all_success_filelist = []

        suc, success_source_files, success_target_files = self.downloader.download(zipfilelist, targetfilelist)
        all_success_filelist.extend(success_target_files)

        # download failed files untill success
        trail_count = 0
        while not suc: 
            zipfilelist = [ff for ff in zipfilelist if ff not in success_source_files]
            if len(zipfilelist) == 0:
                print_warn("No failed files are found! ")
                break

            targetfilelist = [join(self.tartanair_data_root, zipfile.replace('/', '_')) for zipfile in zipfilelist]
            suc, success_source_files, success_target_files = self.downloader.download(zipfilelist, targetfilelist)
            all_success_filelist.extend(success_target_files)
            trail_count += 1
            if trail_count >= max_failure_trial:
                break

        if suc:
            print_highlight("Download completed! Enjoy using TartanAir!")
        else:
            print_warn("Download with failure! The following files are not downloaded ..")
            for ff in zipfilelist:
                print_warn(ff)

        if unzip:
            self.unzip_files(all_success_filelist)

        return True

    def download_multi_thread(self, env = [], difficulty = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, num_workers = 8, **kwargs):

        env, difficulty, modality, camera_name, unzip = self.refine_parameters(env, difficulty, modality, camera_name, unzip, config)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ee in env:
                for dd in difficulty:
                    futures.append(executor.submit(self.download, env = [ee], difficulty = [dd], modality = modality, camera_name = camera_name, 
                                    config = config, unzip = unzip, max_failure_trial = max_failure_trial,))
                    # Wait for a few seconds to avoid overloading the data server
                    time.sleep(2)
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()  # This will re-raise any exceptions caught during the futures' execution                

class TartanGroundDownloader(TartanAirDownloader):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

        self.downloader = AirLabDownloader(bucket_name = 'tartanground')

    def extract_existing_trajectories(self):
        """
        Parses the zip file list and extracts all unique trajectory names (e.g., P0001) per environment.
        Returns:
            {
                'AbandonedCable': {
                    'Data_diff': ['P0000', 'P0001', ...],
                    ...
                },
                ...
            }
        """
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_ground_files.txt'

        env_to_traj = defaultdict(lambda: defaultdict(set))

        with open(gtfile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                rel_path = parts[0]  # e.g., AbandonedCable/Data_easy/P0000/depth_lcam_back.zip
                tokens = rel_path.split('/')
                if len(tokens) < 3:
                    continue
                env = tokens[0]
                subfolder = tokens[1]
                traj = tokens[2]
                if traj.startswith("P"):
                    env_to_traj[env][subfolder].add(traj)

        # Convert sets to sorted lists
        for env in env_to_traj:
            for subfolder in env_to_traj[env]:
                env_to_traj[env][subfolder] = sorted(env_to_traj[env][subfolder])

        return env_to_traj

    def generate_filelist(self, envs, versions, trajectories, modalities, camera_names): 
        '''
        Return a list of zipfiles to be downloaded
        Example: 
        [
            "AbandonedCable/Data_diff/P0000/depth_lcam_back.zip",
            "AbandonedCable/Data_omni/P0001/depth_lcam_front.zip",
            ...
        ]

        '''
        zipfilelist = []
        env_to_traj = self.extract_existing_trajectories()

        for env in envs: 
            envstr = env + '/'
            # # Add the seg_label.zip file
            # zipfilelist.append(envstr + 'seg_labels.zip')
            if "seg_labels" in modalities:
                zipfilelist.append(envstr + 'seg_labels.zip')
            if "sem_pcd" in modalities:
                zipfilelist.append(envstr + f'{env}_sem_pcd.zip')
            if "rgb_pcd" in modalities:
                zipfilelist.append(envstr + f'{env}_rgb_pcd.zip')
            for version in versions:
                diffstr = envstr + 'Data_' + version + '/'
                # Remove rosbag modility from modalities if version is not anymal
                current_modalities = modalities.copy()

                if version != 'anymal':
                    if 'rosbag' in current_modalities:
                        print_warn(f"Rosbag modality is not available for {env} with version {version}. Removing from modalities for {version}")
                        current_modalities = [mod for mod in current_modalities if mod != 'rosbag']
                
                # Find the available trajectories for the current version
                available_trajs = env_to_traj[env][f'Data_{version}']

                if len(trajectories) == 0: # If User did not specify trajectory and single threaded version
                    env_ver_trajs = env_to_traj[env][f'Data_{version}']
                else:
                    env_ver_trajs = trajectories
                    

                if not available_trajs:
                    print_warn(f"No trajectories found for {env} with version {version}. Skipping...")
                    continue

                # Find intersection with available trajs and trajectories
                current_trajs = list(set(available_trajs) & set(env_ver_trajs))

                if not current_trajs:
                    print_warn(f"No trajectories match for {env} with version {version} for provided Traj List. Skipping...")
                    continue

                # Loop over for the trajectories
                for traj in current_trajs:
                    trajstr = traj + '/'
                    folderlist = self.compile_ground_modality_and_cameraname(current_modalities, camera_names)
                    zipfiles = [diffstr + trajstr + fl + '.zip' for fl in folderlist]
                    zipfilelist.extend(zipfiles)

        return zipfilelist
    
    def prepare_download_list(self, env=[], version=[], traj=[], modality=[], camera_name=[], config=None):
        """
        Generate and validate the complete file list for download.
        Returns validated zipfilelist and targetfilelist.
        """
        # Refine parameters
        env, version, traj, modality, camera_name, _ = self.refine_parameters(env, version, traj, modality, camera_name, False, config)

        # Check that the environments are valid.
        if not self.check_env_valid(env):
            return None, None
        # Check that the modalities are valid
        if not self.check_modality_valid(modality, check_ground = True):
            return None, None
        # Check that the camera names are valid
        if not self.check_camera_valid(camera_name, check_ground= True):
            return None, None
        
        # Generate the file list
        zipfilelist = self.generate_filelist(env, version, traj, modality, camera_name)

        if len(zipfilelist) == 0:
            return [], []  # Nothing to download
        
        # Validate against ground truth
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_ground_files.txt'
        if not self.doublecheck_filelist(zipfilelist, gtfile=gtfile):
            return None, None

        # Generate target file list
        targetfilelist = [join(self.tartanair_data_root, zipfile.replace('/', '_')) for zipfile in zipfilelist]
        
        return zipfilelist, targetfilelist
    
    def refine_parameters(self, env, version, traj, modality, camera_name, unzip, config):
        if config is not None:
            print("Using config file: {}".format(config))
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

            # Update the parameters.
            env = config['env']
            version = config['version']
            traj = config['traj']
            modality = config['modality']
            camera_name = config['camera_name']
            unzip = config['unzip']
        
        # Check that the inputs are all lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(version, list):
            version = [version]
        if not isinstance(traj, list):
            traj = [traj]
        if not isinstance(modality, list):
            modality = [modality]
        if not isinstance(camera_name, list):
            camera_name = [camera_name]

        # download all if not specified
        if len(env) == 0:
            env = copy(self.ground_omni_env_names)
        if len(version) == 0:
            version = copy(self.ground_version_names)
        if len(modality) == 0:
            modality = copy(self.ground_modality_names)
        if len(camera_name) == 0:
            camera_name = copy(self.ground_camera_names)

        return env, version, traj, modality, camera_name, unzip

    def unzip_files(self, zipfilelist):
        print_warn('⚠️  Note: Unzipping will overwrite existing files...')

        for zipfile_path in zipfilelist:
            if not isfile(zipfile_path) or not zipfile_path.endswith('.zip'):
                print_error(f"❌ Invalid zip: {zipfile_path}")
                continue

            filename = os.path.basename(zipfile_path)

            # Case 1: Trajectory-level zips
            match_traj = re.match(r"(.+?)_Data_(\w+)_((?:P\d+))_.+\.zip", filename)

            # Case 2: Metadata/label zips like: Env_seg_labels.zip or Env_seg_label_map.zip
            match_label = re.match(r"(.+?)_(seg_labels)\.zip", filename)

            # Case 3: Zipped PCDs like: ModernCityDowntown_ModernCityDowntown_rgb_pcd.zip
            match_pcd = re.match(r"(.+?)_\1_(rgb|sem)_pcd\.zip", filename)

            if match_traj:
                env_name, data_type, traj_name = match_traj.groups()
                dest_dir = join(self.tartanair_data_root, env_name, f"Data_{data_type}", traj_name)

            elif match_label:
                env_name, _ = match_label.groups()
                dest_dir = join(self.tartanair_data_root, env_name)

            elif match_pcd:
                env_name, pcd_type = match_pcd.groups()
                dest_dir = join(self.tartanair_data_root, env_name)

            else:
                print_error(f"❌ Could not parse zip file name: {filename}")
                continue

            os.makedirs(dest_dir, exist_ok=True)
            print(f"📦 Unzipping {filename} -> {dest_dir}")
            cmd = f'unzip -q -o "{zipfile_path}" -d "{dest_dir}"'
            os.system(cmd)

            print_highlight("✅ Unzipping Completed!")

    def download(self, env = [], version = [], traj = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, **kwargs):
        """
        Downloads a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to download the trajectory from. Valid envs are: AbandonedCable, AbandonedFactory, AbandonedFactory2, AbandonedSchool, AmusementPark, AncientTowns, Antiquity3D, BrushifyMoon, CarWelding, CastleFortress, CoalMine, ConstructionSite, CyberPunkDowntown, DesertGasStation, Downtown, EndofTheWorld, FactoryWeather, Fantasy, ForestEnv, Gascola, GothicIsland, GreatMarsh, HQWesternSaloon, HongKong, Hospital, House, IndustrialHangar, JapaneseAlley, JapaneseCity, MiddleEast, ModUrbanCity, ModernCityDowntown, ModularNeighborhood, ModularNeighborhoodIntExt, NordicHarbor, Office, OldBrickHouseDay, OldBrickHouseNight, OldIndustrialCity, OldScandinavia, OldTownFall, OldTownNight, OldTownSummer, OldTownWinter, Prison, Restaurant, Rome, Ruins, SeasideTown, SeasonalForestAutumn, SeasonalForestSpring, SeasonalForestSummerNight, SeasonalForestWinter, SeasonalForestWinterNight, Sewerage, Slaughter, SoulCity, Supermarket, UrbanConstruction, VictorianStreet, WaterMillDay, WaterMillNight, WesternDesertTown
            version (str or list): The version of the trajectory. Valid versions are: omni, diff and anymal.
            traj (str or list): The trajectory to download. Valid trajectories are: P0000, P0001, P0002, etc. If not specified, all trajectories will be downloaded.
            modality (str or list): The modality to download. Valid modalities are: image, depth, seg, imu, lidar, sem_pcd, rgb_pcd, rosbag. Default is image.
            camera_name (str or list): The name of the camera to download. Valid names are: lcam_back, lcam_bottom, lcam_front, lcam_left, lcam_right, lcam_top, rcam_back, rcam_bottom, rcam_front, rcam_left, rcam_right, rcam_top
        
        Note: 
            for imu and lidar, no camera_name needs to be specified. 
        """
        # Prepare the complete download list
        zipfilelist, targetfilelist = self.prepare_download_list(env, version, traj, modality, camera_name, config)
        
        if zipfilelist is None:
            return False
            
        if len(zipfilelist) == 0:
            return True  # Nothing to download
        
        print(f"Total files to download: {len(zipfilelist)}")
        
        # # Add "TartanGround_v2/" prefix to the zipfilelist 
        # prefixed_zipfilelist = [f"TartanGround_v2/{ff}" for ff in zipfilelist]
        
        # # Use chunked download (single chunk for single-threaded)
        success, all_success_filelist = self._download_chunk(zipfilelist, targetfilelist, max_failure_trial, 1)

        if success:
            print_highlight("Download completed! Enjoy using TartanAir!")
        else:
            print_warn("Download completed with some failures. Check the logs above for details.")

        _, _, _, _, _, unzip = self.refine_parameters(env, version, traj, modality, camera_name, unzip, config)

        if unzip and all_success_filelist:
            self.unzip_files(all_success_filelist)

        return success

    def download_multi_thread(self, env = [], version = [], traj = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, num_workers = 8, **kwargs):
        """
        Multithreaded download that first generates the complete file list, then downloads in chunks.
        """
        # First, prepare the complete download list
        print("Preparing complete download list...")
        zipfilelist, targetfilelist = self.prepare_download_list(env, version, traj, modality, camera_name, config)
        
        if zipfilelist is None:
            return False
            
        if len(zipfilelist) == 0:
            print("No files to download.")
            return True
        
        print(f"Total files to download: {len(zipfilelist)}")
        
        # # Add "TartanGround_v2/" prefix to the zipfilelist 
        # prefixed_zipfilelist = [f"TartanGround_v2/{ff}" for ff in zipfilelist]
        
        # Split files into chunks for multithreaded download
        chunk_size = max(1, len(zipfilelist) // num_workers)
        file_chunks = [zipfilelist[i:i + chunk_size] for i in range(0, len(zipfilelist), chunk_size)]
        target_chunks = [targetfilelist[i:i + chunk_size] for i in range(0, len(targetfilelist), chunk_size)]
        
        all_success_filelist = []
        overall_success = True

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            # Submit download jobs for each chunk
            for i, (zip_chunk, target_chunk) in enumerate(zip(file_chunks, target_chunks)):
                print(f"Submitting chunk {i+1}/{len(file_chunks)} with {len(zip_chunk)} files")
                future = executor.submit(self._download_chunk, zip_chunk, target_chunk, max_failure_trial, i+1)
                futures.append(future)
                # Wait for a few seconds to avoid overloading the data server
                time.sleep(2)
            
            # Wait for all futures to complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    success, success_files = future.result()
                    if success:
                        print(f"✅ Chunk {i+1} completed successfully")
                        all_success_filelist.extend(success_files)
                    else:
                        print(f"❌ Chunk {i+1} had failures")
                        overall_success = False
                except Exception as exc:
                    print(f"❌ Chunk {i+1} generated an exception: {exc}")
                    overall_success = False

        if overall_success:
            print_highlight("All downloads completed successfully! Enjoy using TartanAir!")
        else:
            print_warn("Some downloads failed. Check the logs above for details.")

        _, _, _, _, _, unzip = self.refine_parameters(env, version, traj, modality, camera_name, unzip, config)

        if unzip and all_success_filelist:
            self.unzip_files(all_success_filelist)

        return overall_success

    def _download_chunk(self, zipfilelist, targetfilelist, max_failure_trial, chunk_id):
        """
        Download a chunk of files with retry logic.
        """
        print(f"Starting download for chunk {chunk_id} with {len(zipfilelist)} files")
        
        all_success_filelist = []
        suc, success_source_files, success_target_files = self.downloader.download(zipfilelist, targetfilelist)
        all_success_filelist.extend(success_target_files)

        # download failed files until success
        trail_count = 0
        remaining_files = zipfilelist.copy()

        while not suc and trail_count < max_failure_trial:
            remaining_files = [ff for ff in remaining_files if ff not in success_source_files]
            if len(remaining_files) == 0:
                print_warn(f"Chunk {chunk_id}: No failed files are found!")
                break

            print(f"Chunk {chunk_id}: Retrying {len(remaining_files)} failed files (attempt {trail_count + 1})")

            # Remove the prefix before processing
            # stripped_zipfilelist = [zipfile.replace("TartanGround_v2/", "") for zipfile in remaining_files]
            targetfilelist = [join(self.tartanair_data_root, zipfile.replace('/', '_')) for zipfile in remaining_files]

            suc, success_source_files, success_target_files = self.downloader.download(remaining_files, targetfilelist)
            all_success_filelist.extend(success_target_files)
            trail_count += 1

        if not suc:
            print_warn(f"Chunk {chunk_id}: Download with failure! The following files are not downloaded:")
            for ff in remaining_files:
                print_warn(ff)

        return suc, all_success_filelist