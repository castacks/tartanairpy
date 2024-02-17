'''
Author: Yorai Shaoul
Date: 2023-02-03

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os
import sys

from colorama import Fore, Style
import yaml

# Local imports.
from .tartanair_module import TartanAirModule

class TartanAirDownloader(TartanAirModule):
    def __init__(self, tartanair_data_root, azure_token = None):
        super().__init__(tartanair_data_root)

        if not azure_token:
            print(Fore.RED + "TEST ERROR(download): azure_token is None. Please pass a valid azure_token to the init function. This will no longer be necessary when TartanAir will be officially released." + Style.RESET_ALL)
        else:
            self.azure_token = azure_token

        # The modalities that have a camera associated with them and that we'll download the pose file along with.
        self.modalities_with_camera = ['image', 'depth', 'seg', 'flow', 'pose']

    def check_azcopy(self):
        # Check if azcopy executable exists.
        if not os.path.exists('./azcopy'):
           
            res = input("Azcopy executable not found. Downloading azcopy. Would you like to download it? (Y/n) ")

            if res == 'n':
                raise Exception("Azcopy executable not found. Please download it manually and place it in the current directory.")
                
            # Proceed in a different way depending on the OS.
            # TODO(yoraish): test this on windows.
            if os.name == 'nt':
                os.system('powershell.exe -Command "Invoke-WebRequest -Uri https://aka.ms/downloadazcopy-v10-windows -OutFile downloadazcopy-v10-windows.zip"')
                os.system('powershell.exe -Command "Expand-Archive downloadazcopy-v10-windows.zip"')
                os.system('powershell.exe -Command "Remove-Item downloadazcopy-v10-windows.zip"')
                os.system('azcopy.exe')

            # If on Mac.
            elif os.name == 'posix' and sys.platform == 'darwin':
                os.system('wget https://aka.ms/downloadazcopy-v10-mac')
                os.system('tar -xvf downloadazcopy-v10-mac')
                os.system("mv azcopy_darwin_amd64*/azcopy .")

                os.system('rm downloadazcopy-v10-mac* -r')
                os.system('rm -r azcopy_darwin*')
                os.system('chmod +x azcopy')

            # If on Linux.
            elif os.name == 'posix' and sys.platform == 'linux':
                os.system('wget https://aka.ms/downloadazcopy-v10-linux')
                os.system('tar -xvf downloadazcopy-v10-linux')
                os.system("mv azcopy_linux_amd64*/azcopy .")

                os.system('rm downloadazcopy-v10-linux* -r')
                os.system('rm -r azcopy_linux*')
                os.system('chmod +x azcopy')
            
            else:
                raise Exception("Azcopy executable not found for your OS ({}, {}). Please download it manually and place it in the current directory.".format(os.name, sys.platform))

        
    def download(self, env = [], difficulty = [], trajectory_id = [], modality = [], camera_name = [], config = None, **kwargs):
        """
        Downloads a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to download the trajectory from. 
            difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, medium, hard.
            trajectory_id (int or list): The id of the trajectory to download.
            modality (str or list): The modality to download. Valid modalities are: rgb, depth, seg. Default is rgb.
            camera_name (str or list): The name of the camera to download. 
        """
        if config is not None:
            print("Using config file: {}".format(config))
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

            # Update the parameters.
            env = config['env']
            difficulty = config['difficulty']
            trajectory_id = config['trajectory_id']
            modality = config['modality']
            camera_name = config['camera_name']

        # Check if azcopy executable exists.
        self.check_azcopy()
        
        # Check that the inputs are all lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(difficulty, list):
            difficulty = [difficulty]
        if not isinstance(trajectory_id, list):
            trajectory_id = [trajectory_id]
        if not isinstance(modality, list):
            modality = [modality]
        if not isinstance(camera_name, list):
            camera_name = [camera_name]
            
        # Check that the inputs are valid.
        # TODO(yoraish).

        # Download the trajectories.
        for env_i in env:
            # Start by downloading the trajectory's analyze folder txt files in full.
            self.download_analyze(env_i)

            for difficulty_i in difficulty:
                for trajectory_id_i in trajectory_id:

                    # Download the trajectory.
                    # Source.
                    difficulty_str = "Data_" + difficulty_i
                    
                    dest_env = os.path.join(self.tartanair_data_root, env_i, difficulty_str)

                    print("""\n\nDownload details:
Environment: {}
Difficulty: {}
Trajectory id: {}
Modality: {}
Camera name: {}
Destination: {}
""".format(env_i, difficulty_i, trajectory_id_i, modality, camera_name, dest_env))


                    azure_url = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/" + self.azure_token

                    cmd = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url, dest_env)

                    # Add wildcard options.
                    if modality or camera_name:

                        # Download all images.
                        if camera_name:
                            cmd += " --include-pattern '"
                            
                            for modality_i in modality:
                                for camera_name_i in camera_name:
                                    
                                    ############################
                                    # If requested an image (RGB), then add the image -- the naming convention is a bit different so it gets a special treatment.
                                    ############################
                                    if modality_i == 'image' and camera_name_i.split("_")[1] in ['front', 'left', 'right', 'back', 'top', 'bottom']:
                                        cmd += "*" + camera_name_i + ".png;"
                                    
                                    ############################
                                    # If requesting flow, which is only available for the front camera, then add the flow image.
                                    ############################
                                    elif modality_i == 'flow' and camera_name_i.split("_")[1] in ['front']:
                                        cmd += "*flow.png;"

                                    ############################
                                    # If not rgb image, stick to the regular naming convention: camera_name_modality.
                                    ############################
                                    else:
                                        # NOTE(yorais): This may add weird file names, like lidar_lcam_front, if both a special modality and a camera name are specified. This is okay for now, as those files are not downloaded as they do not exist.
                                        cmd += "*" + camera_name_i + "_" + modality_i + "*;"

                                    
                                    ############################
                                    # Add pose file. If the camera is 'regular', meaning in [front, left, right, back, top, bottom], then add the pose file directly from the name of the camera. Otherwise, add the pose file from the front camera (for fisheye and equirect).
                                    ############################
                                    if camera_name_i.split("_")[1] in ['front', 'left', 'right', 'back', 'top', 'bottom']:
                                        cmd += "pose_" + camera_name_i + ".txt;" 
                                    elif camera_name_i.split("_")[1] in ['fish', 'equirect']:
                                        cmd += "pose_" + camera_name_i.split("_")[0] + "_front.txt;"

                            cmd += "'"
                            # print(Fore.GREEN +  'The cmd: ', cmd, Style.RESET_ALL)
                            os.system(cmd)

                    # Download special modalities that are not images.
                    # The input modality for the imu can be of the form 'imu', 'imu_acc', 'imu_gyro', etc. All of those are in the same directory, so we change them to 'imu' to download the whole directory.
                    
                    modality = ['imu' if 'imu' in m else m for m in modality]
                    modality = list(set(modality))

                    if 'lidar' in modality or 'imu' in modality:
                        for mty in [m for m in modality if m in ['lidar', 'imu']]:

                            azure_url_special = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/" + mty + "/" + self.azure_token
                            dest_env_special = os.path.join(self.tartanair_data_root, env_i, difficulty_str, trajectory_id_i)
                            cmd_special = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url_special, dest_env_special)
                            # print(Fore.GREEN +  'A cmd: ', cmd_special, Style.RESET_ALL)
                            os.system(cmd_special)


                        # Also download the front-facing pose file.
                        azure_url_pose = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/pose_lcam_front.txt" + self.azure_token
                        dest_env_special = os.path.join(self.tartanair_data_root, env_i, difficulty_str, trajectory_id_i)
                        os.system('./azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url_pose, dest_env_special))


            # If requested a segmentation image, also add the seg_label.json file.
            if 'seg' in modality:
                azure_url_seg = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + "seg_label.json" + self.azure_token
                dest_env_seg = os.path.join(self.tartanair_data_root, env_i)
                cmd_seg = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url_seg, dest_env_seg)

                # print(Fore.GREEN +  'seg cmd: ', cmd_seg, Style.RESET_ALL)
                os.system(cmd_seg)

    def download_analyze(self, env):
        """Download the analyze folder of a trajectory. It contains text files enumerating the frames that exist in environment trajectories.

        Args:
            env (str): The environment name.
        """
        # Download the analyze folder.
        azure_url = "https://tartanairv2.blob.core.windows.net/data-raw/" + env + "/analyze/" + self.azure_token
        dest_env = os.path.join(self.tartanair_data_root, env)

        cmd = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url, dest_env)
        cmd += " --include-pattern 'data_*.txt;motion_*.npy'"
        # print(Fore.GREEN +  'analyze cmd: ', cmd, Style.RESET_ALL)
        os.system(cmd)
