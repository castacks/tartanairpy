'''
Author: Yorai Shaoul
Date: 2023-02-03

This file contains the download class, which downloads the data from Azure to the local machine.
'''
# General imports.
import os

from colorama import Fore, Style

# Local imports.
from .tartanair_module import TartanAirModule

class TartanAirDownloader(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

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

            # TODO(yoraish): test this on mac.
            elif os.name == 'posix':
                os.system('wget https://aka.ms/downloadazcopy-v10-linux')
                os.system('tar -xvf downloadazcopy-v10-linux')
                os.system("mv azcopy_linux_amd64*/azcopy .")

                os.system('rm downloadazcopy-v10-linux* -r')
                os.system('rm azcopy_linux* -r')
                os.system('chmod +x azcopy')

        
    def download(self, env, difficulty = ['easy'], trajectory_id = ['P000'], modality = [], camera_name = []):
        """
        Downloads a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to download the trajectory from. 
            difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, medium, hard.
            trajectory_id (int or list): The id of the trajectory to download.
            modality (str or list): The modality to download. Valid modalities are: rgb, depth, seg. Default is rgb.
            camera_name (str or list): The name of the camera to download. 
        """
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
                                    if modality_i == 'image' and camera_name_i.split("_")[1] in ['front', 'left', 'right', 'back', 'top', 'bottom']:
                                        cmd += "*" + camera_name_i + ".png;"
                                    else:
                                        cmd += "*" + camera_name_i + "_" + modality_i + "*;"
                            cmd += "'"
                            print(Fore.GREEN +  'The cmd: ', cmd, Style.RESET_ALL)
                            os.system(cmd)

                    # Download special modalities that are not images.
                    if 'lidar' or 'imu' in modality:
                        for mty in [m for m in modality if m in ['lidar', 'imu']]:

                            azure_url_special = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/" + mty + "/" + self.azure_token
                            dest_env_special = os.path.join(self.tartanair_data_root, env_i, difficulty_str, trajectory_id_i)
                            cmd_special = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url_special, dest_env_special)
                            print(Fore.GREEN +  'A cmd: ', cmd_special, Style.RESET_ALL)
                            os.system(cmd_special)