'''
Author: Yorai Shaoul
Date: 2023-02-03

This file contains the TartanAirCustomizer class, which reads local data and synthesizes it into a new camera model.
'''
# General imports.
import json
from multiprocessing import Pool
import multiprocessing
import os
import time
from colorama import Fore, Style
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# PyTorch imports.
import torch
import torch.multiprocessing as mp

# Local imports.
from .tartanair_module import TartanAirModule
from .reader import TartanAirImageReader
from .flow_calculation import depthmap_to_absolute_camera_coordinates, flow_occlusion_post_processing

# Image resampling.
from .image_resampling.image_sampler import SixPlanarNumba, SixPlanarTorch

from .image_resampling.mvs_utils.camera_models import Pinhole, DoubleSphere, LinearSphere, Equirectangular #, PinholeRadTanFast, EUCM
from .image_resampling.mvs_utils.shape_struct import ShapeStruct
from .image_resampling.mvs_utils.ftensor import FTensor
from .image_resampling.image_sampler.blend_function import BlendBy2ndOrderGradTorch, BlendBy2ndOrderGradOcv

class TartanAirCustomizer(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

        # Member variables.
        # The root directory for the dataset.
        self.data_root = tartanair_data_root
        assert os.path.exists(self.data_root), "The data root does not exist."

        # Available camera models.
        self.camera_model_name_to_class = {'pinhole': Pinhole, 
                                           'doublesphere': DoubleSphere, 
                                           'linearsphere': LinearSphere, 
                                           'equirect': Equirectangular,
                                           #'radtan': PinholeRadTanFast,
                                           #'eucm': EUCM
                                           }
        
        self.depth_shape = None

    def customize(self, env, difficulty = [], trajectory_id = [], modality = [], new_camera_models_params = [], R_raw_new = np.eye(4), num_workers = 1, device = 'cpu'):
        ###############################
        # Check the input arguments.
        ###############################

        print("Customizing the TartanAir dataset... With {} workers.".format(num_workers))
        # Check that all arguments are lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(difficulty, list):
            difficulty = [difficulty]
        # "easy" and "hard" to "Data_easy" and "Data_hard".
        difficulty = ["Data_"+d for d in difficulty]

        if not isinstance(trajectory_id, list):
            trajectory_id = [trajectory_id]
        if not isinstance(modality, list):
            modality = [modality]
        if not isinstance(new_camera_models_params, list):
            new_camera_models_params = [new_camera_models_params]

        # Keep track of requested device.
        self.device = device

        # Check that all the required files are available.
        required_cam_sides = set([cam_params['raw_side'] for cam_params in new_camera_models_params])
        for raw_side in required_cam_sides:
            self.check_six_images_exist(env, difficulty, trajectory_id, modality, raw_side)

        # Individual environment directories within the root directory to be post-processed. If empty, all the available directories will be processed.
        if not env:
            self.env_folders = []
        else:
            self.env_folders = env
            
        # The data-folders (easy/hard) to be processed within the environments.
        if not difficulty:
            self.data_folders = ['Data_easy', 'Data_hard']
        else:
            self.data_folders = difficulty

        # The modalities to be postprocessed within the environment data.
        if not modality:
            self.modalities = []
        else:
            self.modalities = modality

        # The camera models to be generate.
        if not new_camera_models_params:
            raise ValueError("new_camera_models_params must be specified.")
        else:
            self.new_cam_models_params = new_camera_models_params

        # Number of processes to used.
        self.num_workers = num_workers

        # Store the matrix used for depth-to-distance calcluation
        self.conv_matrix = None
        self.depth_shape = None

        ###############################
        # Postprocess.
        ###############################

        ###############################
        # Initiate camera models.
        ###############################
        new_cam_model_name_to_cam_model_object_R_dict = {}
        for i, new_cam_model_params in enumerate(self.new_cam_models_params):
            # Create a deep copy.
            new_cam_model_params_copy = json.loads(json.dumps(new_cam_model_params))

            # The name of the new camera model that is used to find the camera model class.
            new_cam_model_name = new_cam_model_params_copy['name']
            # The name of the new camera model that is used to save the data.
            new_cam_model_custom_name = 'custom{}_'.format(i) + new_cam_model_params_copy['name']
            # The new camera model object. We need to convert the width and height to a ShapeStruct.
            new_cam_model_params_copy['params']['shape_struct'] = ShapeStruct(H=new_cam_model_params_copy['params']['height'], W=new_cam_model_params_copy['params']['width'])
            # Remove the height and width from the params.
            del new_cam_model_params_copy['params']['height']
            del new_cam_model_params_copy['params']['width']            
            # Create the new camera model object.
            new_cam_model_object = self.camera_model_name_to_class[new_cam_model_name](**new_cam_model_params_copy['params'])
            # Store the new camera model object.
            new_cam_model_name_to_cam_model_object_R_dict[new_cam_model_custom_name] = (new_cam_model_object, new_cam_model_params_copy['R_raw_new'], new_cam_model_params)


        # Some mappings between attributes and parameters.
        self.reader = TartanAirImageReader()
        modality_to_reader = {"image": self.reader.read_bgr, "depth": self.reader.read_dist, "seg": self.reader.read_seg}
        modality_to_interpolation = {"image": "linear", "seg": "nearest", "depth": "blend"}
        modality_to_writer = {"image": self.reader.write_as_is, "seg": self.reader.write_as_is, "depth": self.reader.write_float_depth}
        
        ###############################
        # Enumerate the trajectories.
        ###############################
        required_cam_sides = set([cam_params['raw_side'] for cam_params in new_camera_models_params])
        required_cam_sides = list(required_cam_sides)
        required_cam_sides = ['lcam' if side == 'left' else 'rcam' for side in required_cam_sides]

        # The path to the directory that has been populated with TartanAir data. Immediately in this directory are environment-named directories.
        tartanair_path = self.data_root
        envs_to_trajs = self.enumerate_trajs(self.data_folders)

        for env_name, env_trajs  in envs_to_trajs.items():
            if self.env_folders and env_name not in self.env_folders:
                continue
            for rel_traj_path in env_trajs: 
                # Proceed only if the trajectory is in the list of trajectories to be processed.
                if trajectory_id and rel_traj_path.split("/")[-1] not in trajectory_id:
                    continue

                traj_path = os.path.join(tartanair_path, env_name, rel_traj_path)

                # For this trajectory folder, create the appropriate folders for each new data input and populate those with resampled images.
                for modality in self.modalities:
                    for cam_name in required_cam_sides: # Could be either of lcam or rcam.
                        for new_cam_model_name, (new_cam_model_object, R_raw_new, params_dict) in new_cam_model_name_to_cam_model_object_R_dict.items():
                            
                            # Create directory.
                            new_data_dir_path = os.path.join(tartanair_path, env_name, rel_traj_path, "_".join([modality, cam_name, new_cam_model_name]))
                            print("Creating directory", new_data_dir_path) # Of form Data_easy/env/P001/image_lcam_custom0
                            
                            # Does not overwrite older directories if those exist.
                            if os.path.exists(new_data_dir_path):
                                pass
                                # print("    !! New data directory already exists. {}".format(new_data_dir_path))
                            else:
                                os.makedirs(new_data_dir_path)

                            ###############################
                            # Create a camera sampler.
                            ###############################
                            # Field of view.
                            fov = new_cam_model_object.fov_degree

                            # Rotation to torch.
                            if R_raw_new is not None:
                                R_raw_new = torch.tensor(R_raw_new).float()
                            else:
                                R_raw_new = torch.eye(3).float()

                            ###############################
                            # Enumerate the frames.
                            ###############################
                            # For each frame, get the six raw images. The number of frames is the same for all modalities so just check it for one.
                            num_frames = len(os.listdir(os.path.join(tartanair_path, env_name, rel_traj_path, "{}_{}_front".format(modality, cam_name))))

                            # Get all the frame file names and sort them.
                            side_to_frame_gfps = {}
                            for side in ['front', 'back', 'left', 'right', 'top', 'bottom']:
                                side_to_frame_gfps[side] = [os.path.join(tartanair_path, env_name, rel_traj_path, "{}_{}_{}".format(modality, cam_name, side), f) for f in os.listdir(os.path.join(tartanair_path, env_name, rel_traj_path, "{}_{}_{}".format(modality, cam_name, side))) if f.endswith(".png") and not f.startswith(".")]
                                side_to_frame_gfps[side].sort(key = lambda x: int(x.split("/")[-1].split("_")[0]))

                            # For each frame, get the six raw images and resample them to the new camera model.
                            ###############################
                            # Sample images from frames.
                            ###############################

                            # Keep a list of the arguments to be multiprocess-passed to the function `sample_image_worker`.
                            sample_image_worker_args = []
                            for frame_ix in range(num_frames):
                                sample_image_worker_args.append([frame_ix, 
                                                                new_cam_model_object, R_raw_new, # sampler, 
                                                                modality, 
                                                                new_cam_model_name, 
                                                                cam_name, 
                                                                side_to_frame_gfps, 
                                                                new_data_dir_path, 
                                                                modality_to_reader, 
                                                                modality_to_interpolation, 
                                                                modality_to_writer
                                ])

                            if num_workers <= 1:
                                print("        Running sequentially.")
                                for arglist in sample_image_worker_args:
                                    self.sample_image_worker(arglist)

                            else:
                                # Run in parallel.
                                print("        Running in parallel with", num_workers, "workers.")
                                try:
                 
                                    with Pool(num_workers) as pool:
                                        pool.map(self.sample_image_worker, sample_image_worker_args)

                                except KeyboardInterrupt:
                                    exit()


                            ###############################
                            # Write the camera model parameters.
                            ###############################
                            # Write the valid mask.
                            out_fn = "valid_mask_{}_{}_{}.png".format(cam_name, modality, new_cam_model_name)
                            out_fp = os.path.join(new_data_dir_path, out_fn)
                            print("Writing", out_fp)

                            # Write the camera model parameters.
                            out_fn = "camera_model_params_{}_{}_{}.json".format(cam_name, modality, new_cam_model_name)
                            out_fp = os.path.join(new_data_dir_path, out_fn)

                            with open(out_fp, 'w') as f:
                                json.dump(params_dict, f, indent=4)
                            print("Writing", out_fp)

                            ###############################
                            # Create a rotated pose file.
                            ###############################
                            # TODO(yoraish): compare a 90 degree roll rotation of the front camera to the top camera pose files and images.
                            # The name of the output file.
                            out_fn = "pose_{}_{}.txt".format(cam_name, new_cam_model_name)

                            # Read the pose file. All rotation matrices are with respect to the front camera.
                            pose_fp = os.path.join(tartanair_path, env_name, rel_traj_path, "pose_{}_front.txt".format(cam_name))
                            poses = np.loadtxt(pose_fp)
                            poses_rotated = poses.copy()

                            # Convert the rotation matrix of the sampled image in raw to NED frame.
                            R_edn_ned = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
                            R_ned_edn = R_edn_ned.T
                            R_raw_new_ned= R_ned_edn @ R_raw_new.numpy() @ R_edn_ned

                            for pose_ix, pose in enumerate(poses):
                                # Get the rotation matrix.
                                q_w_raw = pose[3:7]
                                R_w_raw_ned = Rotation.from_quat(q_w_raw).as_matrix()

                                # Rotate the rotation matrix.
                                R_w_new_ned = R_w_raw_ned @ R_raw_new_ned

                                # Convert the rotation matrix to a quaternion.
                                q_w_new = Rotation.from_matrix(R_w_new_ned).as_quat()

                                # Append to the list.
                                poses_rotated[pose_ix, 3:7] = q_w_new

                            # Write the rotated pose file.
                            out_fp = os.path.join(traj_path, out_fn)
                            np.savetxt(out_fp, poses_rotated, fmt='%.6f')
                                

    def sample_image_worker(self, argslist): 
        frame_ix, new_cam_model_object, R_raw_new, modality, new_cam_model_name, cam_name, side_to_frame_gfps, new_data_dir_path, modality_to_reader, modality_to_interpolation, modality_to_writer = argslist

        # sampler = SixPlanarNumba(new_cam_model_object.fov_degree, new_cam_model_object, R_raw_new)
        R_raw_new = FTensor(R_raw_new, f0='raw', f1='fisheye', rotation=True) # TODO: double check
        sampler = SixPlanarTorch(new_cam_model_object, R_raw_new)
        # sampler = SixPlanarNumba(new_cam_model_object, R_raw_new)

        sampler.device = self.device
        create_figures = False
        
        raw_images = {}
        for side in ['front', 'back', 'left', 'right', 'top', 'bottom']:
            # Revert to below:
            if not create_figures:
                raw_images[side] = modality_to_reader[modality](side_to_frame_gfps[side][frame_ix])
            else:
                img = modality_to_reader[modality](side_to_frame_gfps[side][frame_ix])
                img[0:20, :, :] = 255
                img[-20:, :, :] = 255
                img[:, 0:20, :] = 255
                img[:, -20:, :] = 255
                # Draw a circle in the center of the image.
                cv2.circle(img, (320, 320), 100, (0, 0, 255), -1)
                # Write Front in the circle.
                if side == 'bottom':
                    cv2.putText(img, side, (260, 320), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                else:
                    cv2.putText(img, side, (280, 320), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
                raw_images[side] = img
        
        # Resample the six raw images to the new camera model.
        if modality_to_interpolation[modality] == "blend":
            blend_func = BlendBy2ndOrderGradTorch(0.01) # hard code
            new_image, new_image_valid_mask = sampler.blend_interpolation(raw_images, blend_func, invalid_pixel_value=0)  
        else:
            new_image, new_image_valid_mask = sampler(raw_images, interpolation= modality_to_interpolation[modality], invalid_pixel_value=0)

        # Write the new image.
        out_fn = str(frame_ix).zfill(6) + "_{}_{}_{}.png".format(cam_name, modality, new_cam_model_name)
        out_fp = os.path.join(new_data_dir_path, out_fn)
        print("Writing", out_fp)
        modality_to_writer[modality](out_fp, new_image)

        if create_figures:
            # Save the concatenated raw image cross next to the new image.
            out_vis_img = np.zeros((new_image.shape[0], new_image.shape[1] * 3, new_image.shape[2]), dtype=np.uint8) + 255
            front_img = cv2.resize(raw_images['front'], (new_image.shape[1]//3, new_image.shape[0]//3))
            back_img = cv2.resize(raw_images['back'], (new_image.shape[1]//3, new_image.shape[0]//3))
            left_img = cv2.resize(raw_images['left'], (new_image.shape[1]//3, new_image.shape[0]//3))
            right_img = cv2.resize(raw_images['right'], (new_image.shape[1]//3, new_image.shape[0]//3))
            top_img = cv2.resize(raw_images['top'], (new_image.shape[1]//3, new_image.shape[0]//3))
            bottom_img = cv2.resize(raw_images['bottom'], (new_image.shape[1]//3, new_image.shape[0]//3))

            print("Putting image of shape ", left_img.shape, " at (", left_img.shape[0]+out_vis_img.shape[0]//3 - out_vis_img.shape[0]//3 , ", " , left_img.shape[1], ")")

            out_vis_img[out_vis_img.shape[0]//3:left_img.shape[0]+out_vis_img.shape[0]//3, 0:left_img.shape[1], :] = left_img
            out_vis_img[out_vis_img.shape[0]//3:front_img.shape[0]+out_vis_img.shape[0]//3, front_img.shape[1]:front_img.shape[1]*2, :] = front_img
            out_vis_img[out_vis_img.shape[0]//3:right_img.shape[0]+out_vis_img.shape[0]//3, right_img.shape[1]*2:right_img.shape[1]*3, :] = right_img
            out_vis_img[0:top_img.shape[0], top_img.shape[1]:top_img.shape[1]*2, :] = top_img
            out_vis_img[out_vis_img.shape[0]-bottom_img.shape[0]:out_vis_img.shape[0], bottom_img.shape[1]:bottom_img.shape[1]*2, :] = bottom_img
            out_vis_img[out_vis_img.shape[0]//3:back_img.shape[0]+out_vis_img.shape[0]//3, back_img.shape[1]*3:back_img.shape[1]*4, :] = back_img
            
            # Add the new image.
            new_image[new_image == 127] = 255
            out_vis_img[:, out_vis_img.shape[1]//2:out_vis_img.shape[1]//2+new_image.shape[1], :] = new_image

            # Add arrow from left to right. Before teh new image.
            cv2.arrowedLine(out_vis_img, (out_vis_img.shape[1]//2 -125, out_vis_img.shape[0]//2), (out_vis_img.shape[1]//2 - 15, out_vis_img.shape[0]//2), (0, 0, 255), 5)

            out_vis_fn = str(frame_ix).zfill(6) + "_{}_{}_{}_vis.png".format(cam_name, modality, new_cam_model_name)
            out_vis_fp = os.path.join(new_data_dir_path, out_vis_fn)
            print("Writing", out_vis_fp)
            modality_to_writer[modality](out_vis_fp, out_vis_img)



    def check_six_images_exist(self, env = [], difficulty = [], trajectory_id = [], modality = [], raw_side = ''):
        if raw_side == 'right':
            cam_side = 'rcam'
        elif raw_side == 'left':
            cam_side = 'lcam'
        else:
            raise ValueError("raw_side must be 'right' or 'left'.")

        # If any of the arguments is empty, then evaluate all the folders available in this category.
        # Check that all the required files are available.

        # Iterate environments.
        if not env:
            env = os.listdir(self.data_root)
        for env_folder in env:
            
            # Iterate difficulty.
            if not difficulty:
                env_path = os.path.join(self.data_root, env_folder)
                difficulty = os.listdir(env_path)
            for difficulty_folder in difficulty:

                # Iterate trajectory.
                if not trajectory_id:
                    diff_path = os.path.join(self.data_root, env_folder, difficulty_folder)
                    trajectory_id = os.listdir(diff_path)                
                for traj_id_folder in trajectory_id:

                    # Iterate modality.
                    if not modality:
                        raise ValueError("modality must be specified.")
                    
                    # Keep a list of the file-count in each modality folder. We expect those to all be the same.
                    modality_num_files = []
                    for modality_name in modality:
                        for camera_name in ['front', 'left', 'right', 'back', 'top', 'bottom']:
                            
                            modality_folder = modality_name +  '_' + str(cam_side) + '_' + camera_name
                            path = os.path.join(self.data_root, env_folder, difficulty_folder, traj_id_folder, modality_folder)
                            
                            num_files = len(os.listdir(path))
                            modality_num_files.append(num_files)

                    # Check that all the modality folders have the same number of files.
                    if not all(x == modality_num_files[0] for x in modality_num_files):
                        raise ValueError("The number of files in the modality folders is not the same.")
                    else:
                        print("Success: {env_folder}/{difficulty_folder}/{traj_id_folder} All files are available for {modality} on {raw_side} side.".format(env_folder = env_folder, difficulty_folder = difficulty_folder, traj_id_folder = traj_id_folder, modality = modality, raw_side = raw_side))
            
        return True

class TartanAirFlowCustomizer(TartanAirCustomizer):

    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

    def customize_flow(self, env, difficulty = [], trajectory_id = [], camera_name = [], frame_sep=1, num_workers = 1, device = 'cpu'):
        ###############################
        # Check the input arguments.
        ###############################

        # read the depthmap at front and compute the flow/occlusion between them
        print("Computing Flow and Occlusion... With {} workers.".format(num_workers))

        # Check that all arguments are lists. If not, convert them to lists.
        if not isinstance(env, list):
            env = [env]
        if not isinstance(difficulty, list):
            difficulty = [difficulty]
        # "easy" and "hard" to "Data_easy" and "Data_hard".
        difficulty = ["Data_"+d for d in difficulty]

        if not isinstance(trajectory_id, list):
            trajectory_id = [trajectory_id]

        if not self.check_camera_valid(camera_name):
            return False

        # Keep track of requested device.
        self.device = device

        # Individual environment directories within the root directory to be post-processed. If empty, all the available directories will be processed.
        if not env:
            self.env_folders = []
        else:
            self.env_folders = env
            
        # The data-folders (easy/hard) to be processed within the environments.
        if not difficulty:
            self.data_folders = ['Data_easy', 'Data_hard']
        else:
            self.data_folders = difficulty

        # # check that all images exists
        # for cam_side in cam_sides:
        #     self.check_six_images_exist(env, difficulty, trajectory_id, ['depth'], raw_side=cam_side)

        # Number of processes to used.
        self.num_workers = num_workers

        ###############################
        # Prepare argument list for the flow resampling workers.
        ###############################

        # required_cam_sides = set(cam_sides)

        # The path to the directory that has been populated with TartanAir data. Immediately in this directory are environment-named directories.
        tartanair_path = self.data_root
        envs_to_trajs = self.enumerate_trajs(self.data_folders)

        for env_name, env_trajs  in envs_to_trajs.items():
            if self.env_folders and env_name not in self.env_folders:
                continue
            for rel_traj_path in env_trajs: 
                # Proceed only if the trajectory is in the list of trajectories to be processed.
                if trajectory_id and rel_traj_path.split("/")[-1] not in trajectory_id:
                    continue

                traj_path = os.path.join(tartanair_path, env_name, rel_traj_path)

                for camname in camera_name:
                                                
                    # Create directory.
                    new_data_dir_path = os.path.join(tartanair_path, env_name, rel_traj_path, f"flow_{camname}")
                    print("Creating directory", new_data_dir_path) # Of form Data_easy/env/P001/image_lcam_custom0
                    
                    # Does not overwrite older directories if those exist.
                    if os.path.exists(new_data_dir_path):
                        pass
                        # print("    !! New data directory already exists. {}".format(new_data_dir_path))
                    else:
                        os.makedirs(new_data_dir_path)

                    # Enumerate the frames.
                    # For each frame, get number of resampled images. The number of frames is the same for all modalities so just check it for one.
                    files = os.listdir(os.path.join(tartanair_path, env_name, rel_traj_path, f"depth_{camname}"))
                    num_frames = len([f for f in files if f.endswith(".png") and not f.startswith(".")])

                    # Now, we will prepare argument lists for the flow resampling workers. 
                    # We enumerate all the pairs with "frame_sep" separation.
                    frame_pairs = [(frame_ix, frame_ix + frame_sep) for frame_ix in range(0, num_frames - frame_sep)]

                    # read the poses
                    pose_fp = os.path.join(tartanair_path, env_name, rel_traj_path, f"pose_{camname}.txt")
                    poses = np.loadtxt(pose_fp)
                    
                    job_args = [
                        {
                            "source_path": os.path.join(tartanair_path, env_name, rel_traj_path),
                            "output_path": new_data_dir_path,
                            "cam_name": camname,
                            "cam0": {
                                "frame_index": first_frame_index,
                                "pose_raw_ta": poses[first_frame_index]
                            },
                            "cam1": {
                                "frame_index": second_frame_index,
                                "pose_raw_ta": poses[second_frame_index]
                            },
                        }
                        for first_frame_index, second_frame_index in frame_pairs
                    ]

                    if num_workers <= 1:
                        print("        Running sequentially.")
                        for arglist in job_args:
                            self.sample_flow_worker(arglist)
                    else:
                        # Run in parallel.
                        print("        Running in parallel with", num_workers, "workers.")
                        try:
            
                            with Pool(num_workers) as pool:
                                pool.map(self.sample_flow_worker, job_args)

                        except KeyboardInterrupt:
                            exit()

    def sample_flow_worker(self, argslist):

        # Some mappings between attributes and parameters.
        self.reader = TartanAirImageReader()
        modality_to_reader = {"image": self.reader.read_bgr, "depth": self.reader.read_dist, "seg": self.reader.read_seg}
        modality_to_interpolation = {"image": "linear", "seg": "nearest", "depth": "blend"}
        modality_to_writer = {"image": self.reader.write_as_is, "seg": self.reader.write_as_is, "depth": self.reader.write_float_depth}
        
        source_path = argslist["source_path"]
        output_path = argslist["output_path"]
        cam_name = argslist["cam_name"]
        cam0 = argslist["cam0"]
        cam1 = argslist["cam1"]

        # prepare the views for resampling
        views = []
        for cam_info in [cam0, cam1]:
            frame_index = cam_info["frame_index"]

            # cam_name = f"{cam_side[0]}cam_front"

            depth_filepath = os.path.join(source_path, f"depth_{cam_name}", f"{frame_index:06d}_{cam_name}_depth.png")
            
            depth_image = self.reader.read_depth(depth_filepath)

            # populate views
            view = {}

            # depthmap 
            view["depthmap"] = depth_image

            NED_R_cam = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ], dtype=np.float32)

            cam0_R_camsample0 = np.eye(3, dtype=np.float32)

            # camera pose
            view["camera_pose"] = np.eye(4, dtype=np.float32)
            view["camera_pose"][0:3, 0:3] = Rotation.from_quat(cam_info["pose_raw_ta"][3:]).as_matrix() @ NED_R_cam @ cam0_R_camsample0
            view["camera_pose"][0:3, 3] = cam_info["pose_raw_ta"][:3]
            
            # intrinsics
            view["camera_intrinsics"] = np.array([
                [320, 0, 320 - 0.5],
                [0, 320, 320 - 0.5],
                [0, 0, 1]
            ], dtype=np.float32)

            view["pts3d"], view["valid_mask"] = depthmap_to_absolute_camera_coordinates(
                view["depthmap"], view["camera_intrinsics"], view["camera_pose"]
            )

            # convert all array into tensor, and prepend a batch dimension
            for k, v in view.items():
                view[k] = torch.tensor(v, device=self.device).unsqueeze(0)

            views.append(view)

        # invoke the flow and occlusion calculation code
        flow_occlusion_post_processing(
            views,
            depth_error_threshold=0.1,
            depth_error_temperature=0.1,
            relative_depth_error_threshold=0.01,
            opt_iters=2    
        )

        # write flow and occlusion to the output path (npz)
        flow_filepath = os.path.join(output_path, f"{cam0['frame_index']:06d}_{cam1['frame_index']:06d}_{cam_name}_flow.npz")

        content = {
            "flow_fwd": views[0]["flow"].cpu().numpy(),
            "flow_bwd": views[1]["flow"].cpu().numpy(),
            "fov_mask_fwd": views[0]["fov_mask"].cpu().numpy(),
            "fov_mask_bwd": views[1]["fov_mask"].cpu().numpy(),
            "covisible_mask_fwd": views[0]["non_occluded_mask"].cpu().numpy(),
            "covisible_mask_bwd": views[1]["non_occluded_mask"].cpu().numpy(),
        }

        np.savez(flow_filepath, **content)

        # docs: Assumption(static, mention what will happen for dynamic objects; high level description for how flow is computed, explain mask, explain npz, link to detailed description)