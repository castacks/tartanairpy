"""
Author: Manthan Patel
Date: 2025-08-08
Point Cloud Generator for TartanAir Dataset

This script processes depth and RGB images from the TartanAir dataset to generate
colored 3D point clouds. It supports processing multiple trajectories and cameras,
with configurable subsampling and voxel downsampling for memory efficiency.

Usage:
    python pointcloud_generator.py <data_dir> [trajectory_name] [--subsample RATE] [--voxel-size SIZE]

Example:
    python pointcloud_generator.py /path/to/data
    python pointcloud_generator.py /path/to/data P0001 --subsample 5 --voxel-size 0.05
"""
# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanground_data_root = '/my/path/to/root/folder/for/tartan-ground'

ta.init(tartanground_data_root)

# Download data from following environments.
env = [ "AbandonedFactory",
        "ConstructionSite",
        "Hospital",
]

# Following versions are available: ['omni', 'diff', 'anymal']
# Following modalities are available: ['image', 'meta', 'depth', 'seg', 'lidar', 'imu', 'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd']
# Following camera names are available: ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back', 'lcam_top', 'lcam_bottom', 
#                                          rcam_front', 'rcam_right', 'rcam_left', 'rcam_back', 'rcam_top', 'rcam_bottom']
# Trajectories can be specified as a list of strings, e.g., ['P0000', 'P0001', ...]
# "omni" refers to the omnidirectional robot -> Trajectories are in the form of P0000, P0001, etc.
# "diff" refers to the differential drive robot -> Trajectories are in the form of P1000, P1001, etc.
# "anymal" refers to the quadrupedal robot -> Trajectories are in the form of P2000, P2001, etc.

ta.download_ground(env = env, 
              version = ['omni', 'diff', 'anymal'], 
              traj =[],
              modality = ['image', 'meta', 'depth', 'seg', 'lidar', 'imu', 'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd'],  
              camera_name = ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back'], 
              unzip = False)

# Download all modalities from provided environments for the omnidirectional robot
# ta.download_ground(env = env, 
#               version = ['omni'], 
#               traj =[],
#               modality = [],  
#               camera_name = [], 
#               unzip = False)

# Download One Trajectory Front camera from each environement (Omni-directional motion).
# ta.download_ground(env = [], 
#               version = ['omni'], 
#               traj =['P0000'],
#               modality = [],  
#               camera_name = ['lcam_front'], 
#               unzip = False)

# Download all data from all environments.
# ta.download_ground(env = [], 
#               version = [], 
#               traj =[],
#               modality = [],  
#               camera_name = [], 
#               unzip = False)

# Download the semantic occupancy data for all environments.
# ta.download_ground(env = [], 
#               version = [], 
#               traj = [],
#               modality = ['seg_labels', 'sem_pcd'],  
#               camera_name = [], 
#               unzip = False)

# All above downloads can be done in parallel using multi-threading.

# ta.download_ground_multi_thread(env = env, 
#               version = ['omni', 'diff', 'anymal'], 
#               traj =[],
#               modality = ['image', 'meta', 'depth', 'seg', 'lidar', 'imu', 'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd'],  
#               camera_name = ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back'], 
#               unzip = False)

# ta.download_ground_multi_thread(env = [], 
#               version = [], 
#               traj = [],
#               modality = [],  
#               camera_name = [], 
#               unzip = False, 
#               num_workers = 8)