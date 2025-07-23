'''
Author: Yorai Shaoul
Date: 2023-02-03

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os
# import sys

from colorama import Fore, Style
import yaml

# Local imports.
from .tartanair_module import TartanAirModule, print_error, print_highlight, print_warn
from os.path import isdir, isfile, join
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
            self.print_error("Error: Invalid bucket name. Please use 'tartanair2' or 'tartanground'.")
            return
        
        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    def download(self, filelist, targetfilelist):
        success_source_files, success_target_files = [], []
        for source_file_name, target_file_name in zip(filelist, targetfilelist):
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, success_source_files, success_target_files

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
            success_source_files.append(source_file_name)
            success_target_files.append(target_file_name)

        return True, success_source_files, success_target_files

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

    def generate_filelist(self, version_env_dict, modalities, camera_names): 
        '''
        Return a list of zipfiles to be downloaded
        Example: 
        [
            "abandonedfactory/Data_ground/depth_lcam_back.zip",
            "abandonedfactory/Data_ground/flow_lcam_front.zip",
            ...
        ]

        '''
        zipfilelist = []
        for version, envlist  in version_env_dict.items(): 
            for env in envlist:
                envstr = 'TartanGround_' + version + '/' + env + '/Data_ground/'
                folderlist = self.compile_modality_and_cameraname(modalities, camera_names)
                zipfiles = [envstr + fl + '.zip' for fl in folderlist]
                zipfilelist.extend(zipfiles)

        return zipfilelist
    
    def compile_version_and_env(self, envlist, versionlist):
        version_env_dict = {}
        for ver in versionlist:
            if ver == 'v1':
                valid_envs = self.ground_v1_env_names
            elif ver == 'v2':
                valid_envs = self.ground_v2_env_names
            elif ver == 'v3_anymal':
                valid_envs = self.ground_v3_env_names
            else:
                print_error("Error: The version {} is not valid. Please choose from v1, v2 or v3_anymal.".format(ver))
                return None

            version_env_dict[ver] = []
            for env in envlist:
                if env not in valid_envs:
                    print_warn("Warn: The environment {} is not valid for version {}. ".format(env, ver))
                else:
                    version_env_dict[ver].append(env)
        return version_env_dict
    
    def refine_parameters(self, env, version, modality, camera_name, unzip, config):
        if config is not None:
            print("Using config file: {}".format(config))
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

            # Update the parameters.
            env = config['env']
            version = config['version']
            modality = config['modality']
            camera_name = config['camera_name']
            unzip = config['unzip']
        
        # Check that the inputs are all lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(version, list):
            version = [version]
        if not isinstance(modality, list):
            modality = [modality]
        if not isinstance(camera_name, list):
            camera_name = [camera_name]

        # download all if not specified
        if len(env) == 0:
            env = self.ground_v1_env_names
        if len(version) == 0:
            version = self.ground_version_names
        if len(modality) == 0:
            modality = self.ground_modality_names
        if len(camera_name) == 0:
            camera_name = self.ground_camera_names

        return env, version, modality, camera_name, unzip


    def download(self, env = [], version = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, **kwargs):
        """
        Downloads a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to download the trajectory from. Valid envs are: AbandonedCable, AbandonedFactory, AbandonedFactory2, AbandonedSchool, AmericanDiner, AmusementPark, AncientTowns, Antiquity3D, Apocalyptic, ArchVizTinyHouseDay, ArchVizTinyHouseNight, BrushifyMoon, CarWelding, CastleFortress, CoalMine, ConstructionSite, CountryHouse, CyberPunkDowntown, Cyberpunk, DesertGasStation, Downtown, EndofTheWorld, FactoryWeather, Fantasy, ForestEnv, Gascola, GothicIsland, GreatMarsh, HQWesternSaloon, HongKong, Hospital, House, IndustrialHangar, JapaneseAlley, JapaneseCity, MiddleEast, ModUrbanCity, ModernCityDowntown, ModularNeighborhood, ModularNeighborhoodIntExt, NordicHarbor, Ocean, Office, OldBrickHouseDay, OldBrickHouseNight, OldIndustrialCity, OldScandinavia, OldTownFall, OldTownNight, OldTownSummer, OldTownWinter, PolarSciFi, Prison, Restaurant, RetroOffice, Rome, Ruins, SeasideTown, SeasonalForestAutumn, SeasonalForestSpring, SeasonalForestSummerNight, SeasonalForestWinter, SeasonalForestWinterNight, Sewerage, ShoreCaves, Slaughter, SoulCity, Supermarket, TerrainBlending, UrbanConstruction, VictorianStreet, WaterMillDay, WaterMillNight, WesternDesertTown. 
            version (str or list): The version of the trajectory. Valid difficulties are: v1, v2 and v3_anymal.
            modality (str or list): The modality to download. Valid modalities are: image, depth, seg, imu, lidar. Default is image.
            camera_name (str or list): The name of the camera to download. Valid names are: lcam_back, lcam_bottom, lcam_front, lcam_left, lcam_right, lcam_top, rcam_back, rcam_bottom, rcam_front, rcam_left, rcam_right, rcam_top
        
        Note: 
            for imu and lidar, no camera_name needs to be specified. 
        """

        env, version, modality, camera_name, unzip = self.refine_parameters(env, version, modality, camera_name, unzip, config)

        # Check that the environments are valid.
        if not self.check_env_valid(env):
            return False
        # Check that the modalities are valid
        if not self.check_modality_valid(modality, check_ground = True):
            return False
        # Check that the camera names are valid
        if not self.check_camera_valid(camera_name, check_ground= True):
            return False
        
        # Check that the version is valid for certan environments
        version_env_dict = self.compile_version_and_env(env, version)

        zipfilelist = self.generate_filelist(version_env_dict, modality, camera_name)
        # import ipdb;ipdb.set_trace()
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_ground_files.txt'
        if not self.doublecheck_filelist(zipfilelist, gtfile=gtfile):
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

    def download_multi_thread(self, env = [], version = [], modality = [], camera_name = [], config = None, unzip = False, max_failure_trial = 3, num_workers = 8, **kwargs):
        env, version, modality, camera_name, unzip = self.refine_parameters(env, version, modality, camera_name, unzip, config)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ee in env:
                for vv in version:
                    futures.append(executor.submit(self.download, env = [ee], version = [vv], modality = modality, camera_name = camera_name, 
                                    config = config, unzip = unzip, max_failure_trial = max_failure_trial,))
                    # Wait for a few seconds to avoid overloading the data server
                    time.sleep(2)
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()  # This will re-raise any exceptions caught during the futures' execution                                
    