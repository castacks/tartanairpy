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

class AirLabDownloader(object):
    def __init__(self, bucket_name = 'tartanair2') -> None:
        from minio import Minio
        endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"
        # public key (for donloading): 
        access_key = "4e54CkGDFg2RmPjaQYmW"
        secret_key = "mKdGwketlYUcXQwcPxuzinSxJazoyMpAip47zYdl"

        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    def download(self, filelist, destination_path):
        target_filelist = []
        for source_file_name in filelist:
            target_file_name = join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, None

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")

        return True, target_filelist

class CloudFlareDownloader(object):
    def __init__(self, bucket_name = "tartanair-v2") -> None:
        import boto3
        access_key = "be0116e42ced3fd52c32398b5003ecda"
        secret_key = "103fab752dab348fa665dc744be9b8fb6f9cf04f82f9409d79c54a88661a0d40"
        endpoint_url = "https://0a585e9484af268a716f8e6d3be53bbc.r2.cloudflarestorage.com"

        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      endpoint_url=endpoint_url)

    def download(self, filelist, destination_path):
        """
        Downloads a file from Cloudflare R2 storage using S3 API.

        Args:
        - filelist (list): List of names of the files in the bucket you want to download
        - destination_path (str): Path to save the downloaded file locally
        - bucket_name (str): The name of the Cloudflare R2 bucket

        Returns:
        - str: A message indicating success or failure.
        """

        from botocore.exceptions import NoCredentialsError
        target_filelist = []
        for source_file_name in filelist:
            target_file_name = join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if isfile(target_file_name):
                print_error('Error: Target file {} already exists..'.format(target_file_name))
                return False, None
            try:
                print(f"  Downloading {source_file_name} from {self.bucket_name}...")
                self.s3.download_file(self.bucket_name, source_file_name, target_file_name)
                print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")
            except FileNotFoundError:
                print_error(f"Error: The file {source_file_name} was not found in the bucket {self.bucket_name}.")
                return False, None
            except NoCredentialsError:
                print_error("Error: Credentials not available.")
                return False, None
        return True, target_filelist

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

        self.downloader = CloudFlareDownloader()

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
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_files.txt'
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
            
    def download(self, env = [], difficulty = [], modality = [], camera_name = [], config = None, unzip = False, **kwargs):
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

        zipfilelist = self.generate_filelist(env, difficulty, modality, camera_name)
        # import ipdb;ipdb.set_trace()
        if not self.doublecheck_filelist(zipfilelist):
            return False

        suc, targetfilelist = self.downloader.download(zipfilelist, self.tartanair_data_root)
        if suc:
            print_highlight("Download completed! Enjoy using TartanAir!")

        if unzip:
            self.unzip_files(targetfilelist)

        return True
