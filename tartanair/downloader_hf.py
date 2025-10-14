'''
Author: Manthan Patel
Date: 2025-10-10

This file contains the download class, which downloads the data from Hugging Face to the local machine.
'''
# General imports.
import os
from copy import copy
import re
from colorama import Fore, Style
import yaml
from itertools import islice

# Local imports.
from .tartanair_module import TartanAirModule, print_error, print_highlight, print_warn
from os.path import isdir, isfile, join
from collections import defaultdict
from huggingface_hub import snapshot_download

def chunked_iterable(iterable, chunk_size):
    """Yield successive chunks of given size from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

class TartanGroundHFDownloader(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

        self.repo_id = "theairlabcmu/TartanGround"
        self.output_dir = self.tartanair_data_root  # Local directory to save files
        self.chunk_size = 100  # Number of files to download per chunk from Hugging Face
    
    def doublecheck_filelist(self, filelist, gtfile=''):
        """
        Double-checks file validity against the reference ground truth file list and prints total size.
        """
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
                rel_path = parts[0]
                tokens = rel_path.split('/')
                if len(tokens) < 3:
                    continue
                env = tokens[0]
                subfolder = tokens[1]
                traj = tokens[2]
                if traj.startswith("P"):
                    env_to_traj[env][subfolder].add(traj)

        for env in env_to_traj:
            for subfolder in env_to_traj[env]:
                env_to_traj[env][subfolder] = sorted(env_to_traj[env][subfolder])

        return env_to_traj

    def generate_filelist(self, envs, versions, trajectories, modalities, camera_names): 
        """
        Return a list of zipfiles to be downloaded
        Example: 
        [
            "AbandonedCable/Data_diff/P0000/depth_lcam_back.zip",
            "AbandonedCable/Data_omni/P0001/depth_lcam_front.zip",
            ...
        ]
        """
        zipfilelist = []
        env_to_traj = self.extract_existing_trajectories()

        for env in envs: 
            envstr = env + '/'
            if "seg_labels" in modalities:
                zipfilelist.append(envstr + 'seg_labels.zip')
            if "sem_pcd" in modalities:
                zipfilelist.append(envstr + f'{env}_sem_pcd.zip')
            if "rgb_pcd" in modalities:
                zipfilelist.append(envstr + f'{env}_rgb_pcd.zip')
            for version in versions:
                diffstr = envstr + 'Data_' + version + '/'
                current_modalities = modalities.copy()

                if version != 'anymal' and 'rosbag' in current_modalities:
                    print_warn(f"Rosbag modality is not available for {env} with version {version}. Removing from modalities for {version}")
                    current_modalities = [mod for mod in current_modalities if mod != 'rosbag']
                
                available_trajs = env_to_traj[env][f'Data_{version}']

                if len(trajectories) == 0:
                    env_ver_trajs = env_to_traj[env][f'Data_{version}']
                else:
                    env_ver_trajs = trajectories

                if not available_trajs:
                    print_warn(f"No trajectories found for {env} with version {version}. Skipping...")
                    continue

                current_trajs = list(set(available_trajs) & set(env_ver_trajs))
                if not current_trajs:
                    print_warn(f"No trajectories match for {env} with version {version}. Skipping...")
                    continue

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
        env, version, traj, modality, camera_name, _ = self.refine_parameters(env, version, traj, modality, camera_name, False, config)

        if not self.check_env_valid(env):
            return None, None
        if not self.check_modality_valid(modality, check_ground=True):
            return None, None
        if not self.check_camera_valid(camera_name, check_ground=True):
            return None, None
        
        zipfilelist = self.generate_filelist(env, version, traj, modality, camera_name)

        if len(zipfilelist) == 0:
            return [], []
        
        CURDIR = os.path.dirname(os.path.abspath(__file__))
        gtfile = CURDIR + '/download_ground_files.txt'
        if not self.doublecheck_filelist(zipfilelist, gtfile=gtfile):
            return None, None

        # Here we preserve original folder structure, so no flattening.
        targetfilelist = [join(self.tartanair_data_root, zipfile) for zipfile in zipfilelist]
        
        return zipfilelist, targetfilelist
    
    def refine_parameters(self, env, version, traj, modality, camera_name, unzip, config):
        """
        Normalize parameters, fill defaults if empty, and override with config file if provided.
        """
        if config is not None:
            print("Using config file: {}".format(config))
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

            env = config['env']
            version = config['version']
            traj = config['traj']
            modality = config['modality']
            camera_name = config['camera_name']
            unzip = config['unzip']
        
        if not isinstance(env, list): env = [env]
        if not isinstance(version, list): version = [version]
        if not isinstance(traj, list): traj = [traj]
        if not isinstance(modality, list): modality = [modality]
        if not isinstance(camera_name, list): camera_name = [camera_name]

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
            if not os.path.isfile(zipfile_path) or not zipfile_path.endswith('.zip'):
                print_error(f"❌ Invalid zip: {zipfile_path}")
                continue

            # Extract directly into the current folder alongside the zip
            dest_dir = os.path.dirname(zipfile_path)

            print(f"📦 Unzipping {os.path.basename(zipfile_path)} -> {dest_dir}")
            cmd = f'unzip -q -o "{zipfile_path}" -d "{dest_dir}"'
            os.system(cmd)

        print_highlight("✅ Unzipping Completed!")

    def download(self, env = [], version = [], traj = [], modality = [], camera_name = [], config = None, unzip = False, **kwargs):
        """
        Downloads files from Hugging Face TartanGround dataset while preserving folder structure.

        Args:
            env (str or list): Environments to download.
            version (str or list): Versions to download (omni, diff, anymal).
            traj (str or list): Trajectories (P0000, P0001, ...). Empty = all.
            modality (str or list): Modalities (image, depth, seg, imu, lidar, sem_pcd, rgb_pcd, rosbag, seg_labels).
            camera_name (str or list): Cameras (lcam_front, lcam_back, etc.). Ignored for imu, lidar, etc.
            config (yaml): Optional config file path.
            unzip (bool): Whether to unzip downloaded files.
        """
        zipfilelist, targetfilelist = self.prepare_download_list(env, version, traj, modality, camera_name, config)
        
        if zipfilelist is None:
            return False
        if len(zipfilelist) == 0:
            print("✅ Nothing to download.")
            return True

        print(f"🚀 Downloading {len(zipfilelist)} files from Hugging Face in chunks of {self.chunk_size}...")

        for idx, chunk in enumerate(chunked_iterable(zipfilelist, self.chunk_size), start=1):
            print(f"\n📦 Chunk {idx}: Downloading {len(chunk)} files...")
            
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=self.output_dir,
                allow_patterns=chunk
            )

        print(f"✅ Download completed. Files saved in: {self.output_dir}")

        if unzip:
            print("📦 Unzipping is currently not supported by this script. Kindly use `unzip_ground_files_hf.py` script")
            
        return True
