import os

import numpy as np

# TODO(yoraish):
'''
[ ] Auto install azcopy.
[ ] Check that the inputs are valid.
[ ] Add a function to customize a trajectory.
[ ] Verify download of flow.
[ ] Remove spamming text.
'''

class TartanAir():

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

        self.modality_names = ['image', 'depth', 'seg', 'imu', 'lidar']

        self.env_names = ['DesertGasStation', 'Forest', 'OldIndustrialCity', 'ApocalypticCity']

        self.difficulty_names = ['easy', 'hard']

        # Load the token from a text file.
        self.azure_token = "sv=2021-04-10&st=2022-12-20T15%3A06%3A27Z&se=2023-01-31T15%3A06%3A00Z&sr=c&sp=racwl&sig=JwRHih2ECN7MKyWfX5iFyPvaL%2FMEPHmhZQYgtY5bQI0%3D%"

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

                os.system('rm downloadazcopy-v10-linux -r')
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
                    print("\n\nDownloading from trajectory: ", trajectory_id_i, " from environment: ", env_i, " with difficulty: ", difficulty_i, "\n\n")
                    # Download the trajectory.
                    # Source.
                    difficulty_str = "Data_" + difficulty_i
                    
                    dest_env = os.path.join(self.tartanair_data_root, env_i, difficulty_str)

                    print("Destination: ", dest_env)


                    azure_url = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/" + "?" + self.azure_token

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
                            print('cmd: ', cmd)
                            cmd += "'"
                            os.system(cmd)

                    # Download special modalities that are not images.
                    if 'lidar' or 'imu' in modality:
                        for mty in [m for m in modality if m in ['lidar', 'imu']]:

                            azure_url_special = "https://tartanairv2.blob.core.windows.net/data-raw/" + env_i + "/" + difficulty_str + "/" + trajectory_id_i + "/" + mty + "/?" + self.azure_token
                            dest_env_special = os.path.join(self.tartanair_data_root, env_i, difficulty_str, trajectory_id_i)
                            cmd_special = './azcopy copy "{}" {} --recursive --as-subdir=true' .format(azure_url_special, dest_env_special)
                            os.system(cmd_special)
        

    def customize(self, env, difficulty, trajectory_id, modality = 'rgb', camera_name = 'lcam_front', R_raw_new = np.eye(4), allow_download = True):
        """"
        Checks if the trajectory exists locally. Otherwise, downloads it.
        Use the relevant raw trajectory files, and create a custom image, and delete those the downloaded raw files.
        """
        pass

if __name__ == "__main__":
    tartanair_data_root = '/Users/kunalkapoor/Downloads/tartanairpy/src/tartanair'
    tartanair = TartanAir(tartanair_data_root)
    tartanair.download(env = 'AmericanDinerExposure', difficulty = 'easy', trajectory_id = ['P000', 'P003'], modality = ['imu', 'image'], camera_name = ['lcam_fish'])