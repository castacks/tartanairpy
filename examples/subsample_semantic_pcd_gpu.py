"""
Author: Manthan Patel
Date: 2025-08-08
Subsample Semantic Occupancy Maps for a trajectory in TartanGround Dataset to use for training and evaluating Semaantic Occupancy Prediction Networks

This script subsamples per pose semantic occupancy map from the global semantic point cloud. This script requires a GPU to run.

Usage:
    python subsample_semantic_pcd_gpu.py --root_dir /path/to/tartanground --env ENVNAME --traj PXXXX

Example:
    python subsample_semantic_pcd_gpu.py --root_dir /path/to/tartanground --env ConstructionSite --traj P0000
"""


import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import argparse
from tqdm import tqdm
import cupy as cp  # GPU acceleration
from scipy.spatial import cKDTree  # Use scipy KDTree, transfer results to GPU
import gc
from pathlib import Path
import concurrent.futures
from typing import Tuple, Optional
import glob

# ================= Configuration =================

# Default parameters
DEFAULT_BOUNDS = [-25.0, 25.0, -25.0, 25.0, -5.0, 3.0]  # [x_min, x_max, y_min, y_max, z_min, z_max]
DEFAULT_RESOLUTION = 0.2  # meters per voxel

# GPU memory management
GPU_BATCH_SIZE = 100000  # Process this many voxels at once on GPU

# ================= CLI Arguments =================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract local semantic occupancy maps from TartanGround dataset")
    
    parser.add_argument("--root_dir", type=str, required=True, 
                       help="Path to TartanGround dataset root directory")
    parser.add_argument("--env", type=str, required=True,
                       help="Environment name (e.g., CastleFortress)")
    parser.add_argument("--traj", type=str, required=True,
                       help="Trajectory name (e.g., P001)")
    
    # Bounding box parameters
    parser.add_argument("--x_bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[:2],
                       help="Local X bounds [min, max] in meters")
    parser.add_argument("--y_bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[2:4],
                       help="Local Y bounds [min, max] in meters")
    parser.add_argument("--z_bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[4:],
                       help="Local Z bounds [min, max] in meters")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION,
                       help="Voxel resolution in meters")
    
    # Performance parameters
    parser.add_argument("--gpu_batch_size", type=int, default=GPU_BATCH_SIZE,
                       help="GPU batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of CPU workers for parallel processing")
    parser.add_argument("--subsample_poses", type=int, default=1,
                       help="Process every Nth pose (1 = all poses)")
    
    return parser.parse_args()

def find_trajectory_path(root_dir: str, env: str, traj: str) -> str:
    """
    Find trajectory path in the dataset structure: root_dir/env/X/traj
    
    Args:
        root_dir: Path to dataset root
        env: Environment name
        traj: Trajectory name
    
    Returns:
        Full path to trajectory directory
    
    Raises:
        FileNotFoundError: If trajectory is not found
    """
    env_path = os.path.join(root_dir, env)
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment directory not found: {env_path}")
    
    # Search for trajectory in subdirectories (X can be any subdirectory)
    for subdir in os.listdir(env_path):
        subdir_path = os.path.join(env_path, subdir)
        if os.path.isdir(subdir_path):
            traj_path = os.path.join(subdir_path, traj)
            if os.path.exists(traj_path):
                return traj_path
    
    raise FileNotFoundError(f"Trajectory '{traj}' not found in environment '{env}'. Searched in: {env_path}")

def build_file_paths(root_dir: str, env: str, traj: str):
    """
    Build all required file paths based on dataset structure.
    
    Args:
        root_dir: Path to dataset root
        env: Environment name  
        traj: Trajectory name
        
    Returns:
        Dict containing all file paths
    """
    # Find trajectory directory
    traj_path = find_trajectory_path(root_dir, env, traj)
    
    # Build file paths
    paths = {
        'traj_path': traj_path,
        'pose_file': os.path.join(traj_path, 'pose_lcam_front.txt'),
        'pcd_path': os.path.join(root_dir, env, f'{env}_sem.pcd'),
        'output_dir': os.path.join(traj_path, 'sem_occ'),
        'seg_rgb_path': os.path.join(os.path.dirname(__file__), '../tartanair/seg_rgbs.txt')
    }
    
    # Validate required files exist
    required_files = ['pose_file', 'pcd_path', 'seg_rgb_path']
    for file_key in required_files:
        if not os.path.exists(paths[file_key]):
            raise FileNotFoundError(f"Required file not found: {paths[file_key]}")
    
    # Create output directory if it doesn't exist
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    return paths

# ================= Color Mapping Functions =================

def load_rgb_mapping(file_path: str):
    """Load segmentation label to RGB mapping from file."""
    mapping = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            rgb = tuple(map(int, line.strip().split(',')))
            mapping[idx] = rgb[::-1]  # Convert RGB to BGR for OpenCV compatibility
    return mapping

def create_gpu_color_lookup_table(color_mapping: dict):
    """Create GPU lookup table for RGB to class ID conversion."""
    max_class_id = max(color_mapping.keys()) if color_mapping else 0
    
    # Create RGB lookup table on CPU first
    rgb_to_class_cpu = np.full((256, 256, 256), -1, dtype=np.int32)  # -1 for unknown
    
    for class_id, rgb_tuple in color_mapping.items():
        r, g, b = rgb_tuple
        rgb_to_class_cpu[r, g, b] = class_id
    
    # Transfer to GPU
    rgb_to_class_gpu = cp.asarray(rgb_to_class_cpu)
    
    return rgb_to_class_gpu, max_class_id

# ================= GPU-Accelerated Functions =================

class GPUSemanticOccupancyExtractor:
    def __init__(self, pcd_path: str, seg_rgb_path: str, gpu_batch_size: int = GPU_BATCH_SIZE):
        """Initialize GPU-based occupancy extractor with proper class mapping."""
        print("🚀 Loading global semantic point cloud...")
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        
        if len(self.pcd.points) == 0:
            raise ValueError(f"Empty point cloud loaded from {pcd_path}")
        
        # Load color mapping
        print("🎨 Loading semantic class mapping...")
        self.color_mapping = load_rgb_mapping(seg_rgb_path)
        self.rgb_to_class_gpu, self.max_class_id = create_gpu_color_lookup_table(self.color_mapping)
        print(f"📝 Loaded {len(self.color_mapping)} semantic classes")
        
        # Transfer data to GPU
        print("📦 Transferring point cloud to GPU...")
        self.gpu_points = cp.asarray(np.asarray(self.pcd.points), dtype=cp.float32)
        self.gpu_colors = cp.asarray(np.asarray(self.pcd.colors), dtype=cp.float32)
        
        # Convert colors back to proper class IDs on GPU
        self.gpu_class_ids = self._colors_to_class_ids_gpu_correct(self.gpu_colors)
        
        print(f"✅ Loaded {len(self.pcd.points):,} points to GPU")
        self.gpu_batch_size = gpu_batch_size
        
        # Build KDTree on CPU but keep data on GPU for queries
        print("🌳 Building KDTree for spatial queries...")
        self.cpu_points = np.asarray(self.pcd.points, dtype=np.float32)
        self.kdtree = cKDTree(self.cpu_points)
        print("✅ KDTree ready")
    
    def _colors_to_class_ids_gpu_correct(self, gpu_colors):
        """Convert RGB colors to correct class IDs using the original mapping."""
        # Convert colors from [0,1] to [0,255] integers
        gpu_colors_uint8 = cp.round(gpu_colors * 255).astype(cp.uint8)
        
        # Use the lookup table for direct RGB -> class ID mapping
        r = gpu_colors_uint8[:, 0]
        g = gpu_colors_uint8[:, 1] 
        b = gpu_colors_uint8[:, 2]
        
        # Lookup class IDs using the 3D lookup table
        class_ids = self.rgb_to_class_gpu[r, g, b]
        
        # Handle unknown colors (set to 0 for background/unknown)
        class_ids = cp.where(class_ids == -1, 0, class_ids)
        
        return class_ids.astype(cp.int32)
    
    def extract_local_occupancy_gpu_fast(self, pose: np.ndarray, bounds: list, resolution: float) -> np.ndarray:
        """
        Ultra-fast GPU-based local occupancy extraction using voxel grid approach.
        Faster than KDTree for dense point clouds.
        """
        # Create transformation matrix
        T = self._pose_to_transform(pose)
        T_inv_gpu = cp.linalg.inv(cp.asarray(T, dtype=cp.float32))
        
        # Transform all points to local frame
        points_homo = cp.concatenate([self.gpu_points, cp.ones((len(self.gpu_points), 1), dtype=cp.float32)], axis=1)
        local_points = (T_inv_gpu @ points_homo.T).T[:, :3]
        
        # Filter points within bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        mask = ((local_points[:, 0] >= x_min) & (local_points[:, 0] <= x_max) &
                (local_points[:, 1] >= y_min) & (local_points[:, 1] <= y_max) &
                (local_points[:, 2] >= z_min) & (local_points[:, 2] <= z_max))
        
        if cp.sum(mask) == 0:
            # No points in bounds, return empty occupancy map
            x_size = int((x_max - x_min) / resolution) + 1
            y_size = int((y_max - y_min) / resolution) + 1
            z_size = int((z_max - z_min) / resolution) + 1
            return np.zeros((x_size, y_size, z_size), dtype=np.int32)
        
        # Get points and class IDs within bounds
        local_points_filtered = local_points[mask]
        class_ids_filtered = self.gpu_class_ids[mask]
        
        # Convert to voxel indices
        voxel_indices = cp.floor((local_points_filtered - cp.array([x_min, y_min, z_min])) / resolution).astype(cp.int32)
        
        # Calculate occupancy map dimensions
        x_size = int((x_max - x_min) / resolution) + 1
        y_size = int((y_max - y_min) / resolution) + 1
        z_size = int((z_max - z_min) / resolution) + 1
        
        # Clip indices to bounds
        voxel_indices[:, 0] = cp.clip(voxel_indices[:, 0], 0, x_size - 1)
        voxel_indices[:, 1] = cp.clip(voxel_indices[:, 1], 0, y_size - 1)
        voxel_indices[:, 2] = cp.clip(voxel_indices[:, 2], 0, z_size - 1)
        
        # Create occupancy map
        occupancy_map = cp.zeros((x_size, y_size, z_size), dtype=cp.int32)
        
        # Use first-occurrence approach for speed (similar to your point cloud generation)
        flat_indices = (voxel_indices[:, 0] * y_size * z_size + 
                       voxel_indices[:, 1] * z_size + 
                       voxel_indices[:, 2])
        
        # Get unique voxel indices and their first occurrence
        unique_flat_indices, first_occurrence = cp.unique(flat_indices, return_index=True)
        
        # Get the class IDs for first occurrences
        unique_class_ids = class_ids_filtered[first_occurrence]
        
        # Convert back to 3D indices
        unique_x = unique_flat_indices // (y_size * z_size)
        remaining = unique_flat_indices % (y_size * z_size)
        unique_y = remaining // z_size
        unique_z = remaining % z_size
        
        # Set occupancy map
        occupancy_map[unique_x, unique_y, unique_z] = unique_class_ids
        
        return cp.asnumpy(occupancy_map)
    
    def extract_local_occupancy_gpu(self, pose: np.ndarray, bounds: list, resolution: float) -> np.ndarray:
        """
        Extract local semantic occupancy map around a pose using KDTree approach.
        """
        # Create transformation matrix
        T = self._pose_to_transform(pose)
        T_gpu = cp.asarray(T, dtype=cp.float32)
        
        # Create local voxel grid
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        x_coords = cp.arange(x_min, x_max + resolution, resolution, dtype=cp.float32)
        y_coords = cp.arange(y_min, y_max + resolution, resolution, dtype=cp.float32)
        z_coords = cp.arange(z_min, z_max + resolution, resolution, dtype=cp.float32)
        
        # Create 3D meshgrid
        X, Y, Z = cp.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Flatten to get all voxel centers in local frame
        local_voxel_centers = cp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Transform to global frame
        local_homo = cp.concatenate([local_voxel_centers, cp.ones((len(local_voxel_centers), 1), dtype=cp.float32)], axis=1)
        global_voxel_centers = (T_gpu @ local_homo.T).T[:, :3]
        
        # Initialize occupancy map
        occupancy_shape = (len(x_coords), len(y_coords), len(z_coords))
        occupancy_map = cp.zeros(occupancy_shape, dtype=cp.int32)
        
        # Process in batches to manage GPU memory
        n_voxels = len(global_voxel_centers)
        
        for batch_start in range(0, n_voxels, self.gpu_batch_size):
            batch_end = min(batch_start + self.gpu_batch_size, n_voxels)
            batch_centers = global_voxel_centers[batch_start:batch_end]
            
            # Transfer batch to CPU for KDTree query, then back to GPU
            batch_centers_cpu = cp.asnumpy(batch_centers)
            
            # Find nearest neighbors for each voxel center
            distances, indices = self.kdtree.query(batch_centers_cpu, k=1, distance_upper_bound=resolution * 0.7)
            
            # Convert back to GPU arrays
            distances_gpu = cp.asarray(distances)
            indices_gpu = cp.asarray(indices)
            
            # Valid points are those within the search radius
            valid_mask = (distances_gpu < resolution * 0.7) & (indices_gpu < len(self.gpu_class_ids))
            
            if cp.sum(valid_mask) > 0:
                valid_indices = indices_gpu[valid_mask]
                
                # Get class IDs for valid points
                valid_class_ids = self.gpu_class_ids[valid_indices]
                
                # Map back to 3D coordinates
                valid_flat_idx = cp.arange(batch_start, batch_end)[valid_mask]
                
                # Convert flat indices back to 3D indices
                x_idx = valid_flat_idx // (occupancy_shape[1] * occupancy_shape[2])
                remaining = valid_flat_idx % (occupancy_shape[1] * occupancy_shape[2])
                y_idx = remaining // occupancy_shape[2]
                z_idx = remaining % occupancy_shape[2]
                
                # Update occupancy map
                occupancy_map[x_idx, y_idx, z_idx] = valid_class_ids
        
        # Transfer result back to CPU
        return cp.asnumpy(occupancy_map)
    
    def _pose_to_transform(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose to 4x4 transformation matrix."""
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pose[:3]  # translation
        T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix().astype(np.float32)  # rotation
        return T
    
    def __del__(self):
        """Clean up GPU memory."""
        if hasattr(self, 'gpu_points'):
            del self.gpu_points
        if hasattr(self, 'gpu_colors'):
            del self.gpu_colors
        if hasattr(self, 'gpu_class_ids'):
            del self.gpu_class_ids
        if hasattr(self, 'rgb_to_class_gpu'):
            del self.rgb_to_class_gpu
        cp.get_default_memory_pool().free_all_blocks()

# ================= CPU Processing Functions =================

def load_poses(pose_file: str) -> np.ndarray:
    """Load poses from file."""
    poses = np.loadtxt(pose_file, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    return poses

def save_occupancy_map(occupancy_map: np.ndarray, output_path: str, pose: np.ndarray, 
                      bounds: list, resolution: float, color_mapping: dict):
    """Save occupancy map with metadata including class mapping."""
    np.savez_compressed(
        output_path,
        occupancy_map=occupancy_map,
        pose=pose,
        bounds=np.array(bounds),
        resolution=resolution,
        shape=np.array(occupancy_map.shape),
        class_mapping=color_mapping  # Include original class mapping for reference
    )

def process_pose_batch(extractor: GPUSemanticOccupancyExtractor, 
                      poses: np.ndarray, 
                      output_dir: str,
                      bounds: list,
                      resolution: float,
                      start_idx: int) -> int:
    """Process a batch of poses."""
    processed_count = 0
    
    for i, pose in enumerate(poses):
        pose_idx = start_idx + i
        
        try:
            # Extract occupancy map using the fast GPU method
            occupancy_map = extractor.extract_local_occupancy_gpu_fast(pose, bounds, resolution)
            
            # Save result with class mapping
            output_path = os.path.join(output_dir, f"semantic_occupancy_{pose_idx:06d}.npz")
            save_occupancy_map(occupancy_map, output_path, pose, bounds, resolution, extractor.color_mapping)
            
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing pose {pose_idx}: {e}")
            continue
    
    return processed_count

# ================= Main Processing =================

def main():
    args = parse_args()
    
    # Build all file paths based on dataset structure
    print(f"📁 Building file paths for {args.env}/{args.traj}...")
    try:
        paths = build_file_paths(args.root_dir, args.env, args.traj)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Combine bounds
    bounds = [args.x_bounds[0], args.x_bounds[1], 
              args.y_bounds[0], args.y_bounds[1], 
              args.z_bounds[0], args.z_bounds[1]]
    
    print(f"📋 Configuration:")
    print(f"   Environment: {args.env}")
    print(f"   Trajectory: {args.traj}")
    print(f"   Trajectory Path: {paths['traj_path']}")
    print(f"   Point Cloud: {paths['pcd_path']}")
    print(f"   Pose File: {paths['pose_file']}")
    print(f"   Seg RGB File: {paths['seg_rgb_path']}")
    print(f"   Output Dir: {paths['output_dir']}")
    print(f"   Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]")
    print(f"   Resolution: {args.resolution:.3f}m")
    print(f"   GPU Batch Size: {args.gpu_batch_size:,}")
    
    # Load poses
    print(f"📍 Loading poses from {paths['pose_file']}")
    all_poses = load_poses(paths['pose_file'])
    
    # Subsample poses if requested
    poses = all_poses[::args.subsample_poses]
    print(f"📊 Processing {len(poses):,} poses (every {args.subsample_poses} from {len(all_poses):,} total)")
    
    # Calculate expected occupancy map size
    x_size = int((bounds[1] - bounds[0]) / args.resolution) + 1
    y_size = int((bounds[3] - bounds[2]) / args.resolution) + 1
    z_size = int((bounds[5] - bounds[4]) / args.resolution) + 1
    print(f"📐 Occupancy map size: {x_size} x {y_size} x {z_size} = {x_size*y_size*z_size:,} voxels")
    
    # Initialize GPU extractor with proper class mapping
    extractor = GPUSemanticOccupancyExtractor(paths['pcd_path'], paths['seg_rgb_path'], args.gpu_batch_size)
    
    # Process poses with progress bar
    print(f"🏃 Processing {len(poses)} poses...")
    total_processed = 0
    
    batch_size = max(1, args.num_workers)  # Process in small batches to manage memory
    
    with tqdm(total=len(poses), desc="Processing poses") as pbar:
        for batch_start in range(0, len(poses), batch_size):
            batch_end = min(batch_start + batch_size, len(poses))
            pose_batch = poses[batch_start:batch_end]
            
            # Process batch
            processed = process_pose_batch(
                extractor, pose_batch, paths['output_dir'], 
                bounds, args.resolution, batch_start
            )
            
            total_processed += processed
            pbar.update(len(pose_batch))
            
            # Force garbage collection
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    
    print(f"✅ Successfully processed {total_processed}/{len(poses)} poses")
    print(f"💾 Results saved to: {paths['output_dir']}")
    
    # Clean up
    del extractor

if __name__ == "__main__":
    main()