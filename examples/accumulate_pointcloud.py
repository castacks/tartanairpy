#!/usr/bin/env python3
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

import numpy as np
import open3d as o3d
import os
import cv2
import argparse
from scipy.spatial.transform import Rotation
from os.path import join
from tqdm import tqdm

# CAMERAS = ["front", "back", "left", "right"]
CAMERAS = ["front"]  # We assume always the lcam

def depth_to_point_cloud(depth, rgb_img, focalx=320., focaly=320., pu=320., pv=320., filtermin=0.1, filtermax=100):
    """
    Convert depth image to colored point cloud.
    
    Args:
        depth: depth image (H x W)
        rgb_img: RGB image (H x W x 3)
        focal params: camera intrinsics
        filter params: depth filtering thresholds
    
    Returns:
        points: Nx3 numpy array (3D coordinates)
        colors: Nx3 numpy array (RGB colors normalized 0-1)
    """
    h, w = depth.shape
    wIdx, hIdx = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    
    # Filter valid depth values
    valid_mask = (depth > filtermin) & (depth < filtermax)
    
    depth_filtered = depth[valid_mask]
    u = wIdx[valid_mask]
    v = hIdx[valid_mask]
    
    # Convert to 3D coordinates (NED frame)
    x = (u - pu) * depth_filtered / focalx
    y = (v - pv) * depth_filtered / focaly
    points = np.stack([depth_filtered, x, y], axis=1)
    
    # Extract corresponding RGB colors
    colors = rgb_img[valid_mask] / 255.0  # Normalize to 0-1
    
    return points, colors

def transform_points(points, pose_source, pose_target=None):
    """
    Transform point cloud from source pose to target pose (or world frame).
    
    Args:
        points: Nx3 numpy array
        pose_source: 4x4 transformation matrix
        pose_target: 4x4 transformation matrix (default: identity/world frame)
    
    Returns:
        Transformed points (Nx3)
    """
    if pose_target is None:
        pose_target = np.eye(4)
    
    # Transform to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply transformation
    T_rel = np.linalg.inv(pose_target) @ pose_source
    points_transformed = (T_rel @ points_homo.T).T
    
    return points_transformed[:, :3]

def pose_to_matrix(pose):
    """
    Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T

class PointCloudGenerator:
    """
    Point cloud generator for TartanAir dataset.
    
    This class processes depth and RGB images from multiple trajectories and cameras
    to generate colored 3D point clouds in world coordinates.
    
    Attributes:
        data_dir (str): Path to the dataset directory
        subsample_rate (int): Process every N-th frame to reduce memory usage
        voxel_size (float): Voxel size for downsampling point clouds
        trajectories (list): List of trajectory names to process
    """
    
    def __init__(self, data_dir, trajectories=None, subsample_rate=10, voxel_size=0.1):
        """
        Initialize the point cloud generator.
        
        Args:
            data_dir (str): Path to the dataset directory
            trajectories (list, optional): List of specific trajectories to process.
                                         If None, processes all trajectories starting with 'P'
            subsample_rate (int): Process every N-th frame (default: 10)
            voxel_size (float): Voxel size for downsampling (default: 0.1)
        """
        self.data_dir = data_dir
        self.subsample_rate = subsample_rate
        self.voxel_size = voxel_size
        
        if trajectories is None:
            self.trajectories = sorted([traj for traj in os.listdir(data_dir) if traj.startswith("P")])
        else:
            if isinstance(trajectories, str):
                trajectories = [trajectories]
            self.trajectories = trajectories
    
    def load_depth(self, depth_path):
        """
        Load depth image from PNG or NPY file.
        
        Args:
            depth_path (str): Path to depth image file
            
        Returns:
            np.ndarray or None: Depth image as numpy array, None if failed to load
        """
        if depth_path.endswith('.npy'):
            return np.load(depth_path)
        
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            return None
        return depth_rgba.view("<f4").squeeze()
    
    def load_rgb(self, rgb_path):
        """
        Load RGB image and convert from BGR to RGB format.
        
        Args:
            rgb_path (str): Path to RGB image file
            
        Returns:
            np.ndarray or None: RGB image as numpy array, None if failed to load
        """
        img = cv2.imread(rgb_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    def process_trajectory(self, traj_name):
        """Process a single trajectory and generate point cloud."""
        traj_path = join(self.data_dir, traj_name)
        global_pcd = o3d.geometry.PointCloud()
        
        print(f"\n🔹 Processing trajectory: {traj_name}")
        
        for cam in CAMERAS:
            print(f"  📷 Camera: {cam}")
            
            # File paths
            pose_file = join(traj_path, f"pose_lcam_{cam}.txt")
            depth_dir = join(traj_path, f"depth_lcam_{cam}")
            image_dir = join(traj_path, f"image_lcam_{cam}")
            
            if not os.path.exists(pose_file):
                print(f"    ⚠️  Skipping - pose file not found: {pose_file}")
                continue
            
            # Load poses and subsample
            poses = np.loadtxt(pose_file)
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
            
            # Subsample to reduce memory usage
            depth_files = depth_files[::self.subsample_rate]
            poses = poses[::self.subsample_rate]
            
            print(f"    📊 Processing {len(depth_files)} frames (subsampled by {self.subsample_rate})")
            
            # Process each frame
            for i, depth_filename in enumerate(tqdm(depth_files, 
                                                    desc=f"    Frame", 
                                                    leave=False,
                                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Pose {n:04d}]")):
                
                depth_path = join(depth_dir, depth_filename)
                rgb_filename = depth_filename.replace("_depth", "")
                rgb_path = join(image_dir, rgb_filename)
                
                # Load images
                depth = self.load_depth(depth_path)
                rgb_img = self.load_rgb(rgb_path)
                
                if depth is None or rgb_img is None:
                    print(f"    ⚠️  Skipping frame {i:04d} - missing data")
                    continue
                
                # Generate point cloud
                points, colors = depth_to_point_cloud(depth, rgb_img)
                
                if len(points) == 0:
                    continue
                
                # Transform to world coordinates
                current_pose = pose_to_matrix(poses[i])
                points_world = transform_points(points, current_pose)
                
                # Create temporary point cloud
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points_world)
                temp_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Merge with global point cloud
                global_pcd += temp_pcd
        
        # Downsample final point cloud
        print(f"  🔽 Downsampling point cloud (voxel size: {self.voxel_size})")
        global_pcd = global_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        return global_pcd
    
    def run(self):
        """
        Process all configured trajectories in the dataset.
        
        This method iterates through all trajectories, processes each one to generate
        a colored point cloud, and saves the results as .pcd files.
        """
        print(f"🚀 Starting point cloud generation for {len(self.trajectories)} trajectories")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📋 Trajectories: {self.trajectories}")
        print(f"⚙️  Subsample rate: {self.subsample_rate}")
        print(f"📐 Voxel size: {self.voxel_size}")
        
        for traj_name in self.trajectories:
            try:
                # Process trajectory
                pcd = self.process_trajectory(traj_name)
                
                # Save point cloud
                output_path = join(self.data_dir, f"{traj_name}_colored.pcd")
                o3d.io.write_point_cloud(output_path, pcd)
                
                print(f"  ✅ Saved: {output_path}")
                print(f"  📊 Points: {len(pcd.points):,}")
                
            except Exception as e:
                print(f"  ❌ Error processing {traj_name}: {e}")
                continue
        
        print("\n🎉 Point cloud generation completed!")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate colored point clouds from TartanAir dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset (/path/to/tartanground/CastleFortress/Data_omni)
  %(prog)s /path/to/dataset P0000
  %(prog)s /path/to/dataset P0000 --subsample 5 --voxel-size 0.05
  %(prog)s /path/to/dataset --trajectories P0000,P0001,P0002
        """
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to the dataset directory containing trajectory folders'
    )
    
    parser.add_argument(
        'trajectory',
        type=str,
        nargs='?',
        help='Specific trajectory name to process (e.g., P0001). If not provided, processes all trajectories'
    )
    
    parser.add_argument(
        '--trajectories',
        type=str,
        help='Comma-separated list of trajectory names to process (e.g., P0001,P0002,P0003)'
    )
    
    parser.add_argument(
        '--subsample',
        type=int,
        default=10,
        help='Process every N-th frame to reduce memory usage (default: 10)'
    )
    
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.1,
        help='Voxel size for point cloud downsampling (default: 0.1)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine which trajectories to process
    trajectories = None
    if args.trajectories:
        trajectories = [t.strip() for t in args.trajectories.split(',')]
    elif args.trajectory:
        trajectories = [args.trajectory]
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' does not exist!")
        exit(1)
    
    # Run point cloud generation
    generator = PointCloudGenerator(
        data_dir=args.data_dir,
        trajectories=trajectories,
        subsample_rate=args.subsample,
        voxel_size=args.voxel_size
    )
    generator.run()