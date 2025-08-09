"""
Simple Interactive Semantic Occupancy Viewer for TartanGround Dataset

A simple viewer that opens occupancy maps one by one using standard Open3D viewer.
Uses keyboard input to navigate between different occupancy maps.

Usage:
    python simple_viewer.py --root_dir /path/to/dataset --env CastleFortress --traj P001

Controls:
    - Close window and press ENTER: Next occupancy map
    - Close window and press 'b': Previous occupancy map  
    - Close window and press 'q': Quit

Author: Simple viewer for TartanGround dataset
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial.transform import Rotation
import json

# ================= Configuration =================

DEFAULT_SEG_RGB_PATH = os.path.join(os.path.dirname(__file__), '../tartanair/seg_rgbs.txt')

# ================= Dataset Path Functions =================

def find_trajectory_path(root_dir: str, env: str, traj: str) -> str:
    """Find trajectory path in the dataset structure: root_dir/env/X/traj"""
    env_path = os.path.join(root_dir, env)
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment directory not found: {env_path}")
    
    # Search for trajectory in subdirectories
    for subdir in os.listdir(env_path):
        subdir_path = os.path.join(env_path, subdir)
        if os.path.isdir(subdir_path):
            traj_path = os.path.join(subdir_path, traj)
            if os.path.exists(traj_path):
                return traj_path
    
    raise FileNotFoundError(f"Trajectory '{traj}' not found in environment '{env}'")

def get_occupancy_files(root_dir: str, env: str, traj: str, skip_samples: int = 1) -> list:
    """Get list of occupancy map files from the dataset structure."""
    traj_path = find_trajectory_path(root_dir, env, traj)
    sem_occ_dir = os.path.join(traj_path, 'sem_occ')
    
    if not os.path.exists(sem_occ_dir):
        raise FileNotFoundError(f"Semantic occupancy directory not found: {sem_occ_dir}")
    
    # Find all .npz files
    npz_files = sorted(glob.glob(os.path.join(sem_occ_dir, "*.npz")))
    
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {sem_occ_dir}")
    
    # Apply skip sampling
    npz_files = npz_files[::skip_samples]
    
    return npz_files

# ================= Utility Functions =================

def load_rgb_mapping(file_path: str) -> dict:
    """Load segmentation label to RGB mapping from file."""
    mapping = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            rgb = tuple(map(int, line.strip().split(',')))
            mapping[idx] = rgb[::-1]  # Convert RGB to BGR for OpenCV compatibility
    return mapping

def load_class_labels(label_map_path: str) -> dict:
    """Load class labels from TartanAir JSON format."""
    if not Path(label_map_path).exists():
        return None
    
    try:
        with open(label_map_path, 'r') as f:
            data = json.load(f)
        
        if "name_map" in data:
            name_to_id = data["name_map"]
            id_to_name = {int(idx): name for name, idx in name_to_id.items()}
            return id_to_name
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ Error loading class labels: {e}")
        return None

def get_class_name(class_id: int, label_mapping: dict = None) -> str:
    """Get human-readable class name from class ID."""
    if label_mapping and class_id in label_mapping:
        return label_mapping[class_id]
    else:
        return f"Class_{class_id}"

def class_id_to_rgb(class_ids: np.ndarray, color_mapping: dict) -> np.ndarray:
    """Convert class IDs to RGB colors for visualization."""
    rgb_colors = np.zeros((len(class_ids), 3), dtype=np.float32)
    for i, class_id in enumerate(class_ids):
        if class_id in color_mapping:
            rgb_colors[i] = np.array(color_mapping[class_id], dtype=np.float32) / 255.0
        else:
            rgb_colors[i] = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # Gray for unknown
    return rgb_colors

def analyze_occupancy_map(occupancy_map: np.ndarray, color_mapping: dict = None, 
                         label_mapping: dict = None) -> None:
    """Analyze and print occupancy map statistics."""
    print("\n" + "="*60)
    print("SEMANTIC OCCUPANCY MAP ANALYSIS")
    print("="*60)
    
    print(f"📊 Occupancy map shape: {occupancy_map.shape}")
    print(f"📐 Total voxels: {occupancy_map.size:,}")
    
    # Count occupied vs empty voxels
    occupied_voxels = np.sum(occupancy_map > 0)
    empty_voxels = np.sum(occupancy_map == 0)
    occupancy_ratio = occupied_voxels / occupancy_map.size * 100
    
    print(f"🏠 Occupied voxels: {occupied_voxels:,} ({occupancy_ratio:.1f}%)")
    print(f"🕳️  Empty voxels: {empty_voxels:,} ({100-occupancy_ratio:.1f}%)")
    
    if occupied_voxels > 0:
        unique_classes, counts = np.unique(occupancy_map[occupancy_map > 0], return_counts=True)
        print(f"\n🎨 Semantic classes present: {len(unique_classes)}")
        print(f"   Class distribution (occupied voxels only):")
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        
        for i in sorted_indices:
            class_id = unique_classes[i]
            count = counts[i]
            percentage = (count / occupied_voxels) * 100
            
            class_name = get_class_name(class_id, label_mapping)
            
            if color_mapping and class_id in color_mapping:
                rgb = color_mapping[class_id]
                color_square = f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m  \033[0m"
                print(f"     Class {class_id:3d}: {count:7,} voxels ({percentage:5.1f}%) {color_square} '{class_name}'")
            else:
                color_square = "\033[48;2;128;128;128m  \033[0m"
                print(f"     Class {class_id:3d}: {count:7,} voxels ({percentage:5.1f}%) {color_square} '{class_name}'")
    
    print("="*60)

def occupancy_to_pointcloud(occupancy_map: np.ndarray, bounds: list, resolution: float, 
                          color_mapping: dict = None, min_class_id: int = 1, 
                          camera_pose: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Convert occupancy map to colored point cloud."""
    
    # Only occupied voxels
    occupied_mask = occupancy_map >= min_class_id
    coords = np.array(np.where(occupied_mask)).T
    class_ids = occupancy_map[occupied_mask]
    
    if len(coords) == 0:
        return o3d.geometry.PointCloud()
    
    # Convert voxel coordinates to local coordinates
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    local_points = coords.astype(np.float32)
    local_points[:, 0] = x_min + local_points[:, 0] * resolution + resolution/2
    local_points[:, 1] = y_min + local_points[:, 1] * resolution + resolution/2
    local_points[:, 2] = z_min + local_points[:, 2] * resolution + resolution/2
    
    # Transform to world coordinates if pose available
    if camera_pose is not None:
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, 3] = camera_pose[:3]
        T_cam_to_world[:3, :3] = Rotation.from_quat(camera_pose[3:]).as_matrix()
        
        local_points_homo = np.hstack([local_points, np.ones((len(local_points), 1))])
        world_points = (T_cam_to_world @ local_points_homo.T).T[:, :3]
        points = world_points
    else:
        points = local_points
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors
    if color_mapping:
        colors = class_id_to_rgb(class_ids, color_mapping)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        max_class = np.max(class_ids) if len(class_ids) > 0 else 1
        colors = plt.cm.tab20(class_ids / max_class)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def create_coordinate_frames(camera_pose: np.ndarray = None, bounds: list = None, 
                           frame_type: str = 'camera') -> list:
    """
    Create coordinate frames for visualization.
    
    Args:
        camera_pose: [x, y, z, qx, qy, qz, qw] camera pose in world
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max] local bounds
        frame_type: 'camera' or 'body' to determine frame setup
    
    Returns:
        List of (name, frame) tuples
    """
    frames = []
    
    if camera_pose is not None:
        # World coordinate frame at world origin
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        world_frame.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for world frame
        frames.append(("World Frame (Origin)", world_frame))
        
        # Camera/Robot coordinate frame at camera pose location
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
        
        # Apply the camera pose transformation to show camera orientation
        R = Rotation.from_quat(camera_pose[3:]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = camera_pose[:3]
        camera_frame.transform(T)
        
        # Color coding for camera frame
        camera_frame.paint_uniform_color([1.0, 0.0, 0.0])  # Red for camera frame
        frames.append(("Camera Frame (Robot Location)", camera_frame))
        
        # If we have bounds, show the local frame center as well
        if bounds is not None:
            # Local frame at center of occupancy map bounds
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            local_center_world = np.array([
                (x_min + x_max) / 2,
                (y_min + y_max) / 2, 
                (z_min + z_max) / 2
            ])
            
            # Transform local center to world coordinates
            local_center_world_homo = np.array([local_center_world[0], local_center_world[1], local_center_world[2], 1])
            local_center_global = T @ local_center_world_homo
            
            local_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            local_frame.translate(local_center_global[:3])
            local_frame.paint_uniform_color([0.0, 1.0, 0.0])  # Green for local occupancy center
            frames.append(("Local Occupancy Center", local_frame))
    
    return frames

def add_frame_legends(vis):
    """Add legend information to the visualizer."""
    # This is mainly for console output since Open3D doesn't have easy text overlay
    print("\n" + "="*60)
    print("COORDINATE FRAME LEGEND")
    print("="*60)
    print("🔴 RED AXES   = Ego Frame (NED) at robot pose")
    print("")
    print("⚪ GRAY AXES  = World Frame at world origin (0,0,0)")
    print("")
    print("🟢 GREEN AXES = Local Occupancy Map Center")
    print("   • Shows center of the local occupancy grid")
    print("="*60 + "\n")

def visualize_occupancy_map(npz_path: str, color_mapping: dict = None, 
                          label_mapping: dict = None, point_size: float = 8.0,
                          background: str = "black") -> None:
    """Visualize a single occupancy map."""
    
    print(f"\n🔄 Loading: {os.path.basename(npz_path)}")
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    occupancy_map = data['occupancy_map']
    bounds = data['bounds']
    resolution = data['resolution']
    
    # Get pose
    pose = None
    if 'camera_pose' in data:
        pose = data['camera_pose']
    elif 'pose' in data:
        pose = data['pose']
    
    # Get frame information
    frame_info = {}
    if 'frame' in data:
        frame_info['frame'] = str(data['frame'])
    if 'coordinate_info' in data:
        frame_info['coordinate_info'] = data['coordinate_info'].item() if hasattr(data['coordinate_info'], 'item') else data['coordinate_info']
    
    if pose is not None:
        print(f"📍 Pose: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}] + quaternion")
    
    print(f"📦 Bounds: X[{bounds[0]:.1f}, {bounds[1]:.1f}] Y[{bounds[2]:.1f}, {bounds[3]:.1f}] Z[{bounds[4]:.1f}, {bounds[5]:.1f}]")
    print(f"📏 Resolution: {resolution:.3f}m per voxel")
    
    # Analyze occupancy map
    analyze_occupancy_map(occupancy_map, color_mapping, label_mapping)
    
    # Convert to point cloud
    pcd = occupancy_to_pointcloud(occupancy_map, bounds, resolution, color_mapping, 
                                 min_class_id=1, camera_pose=pose)
    
    if len(pcd.points) == 0:
        print("❌ No points to visualize!")
        return
    
    print(f"✅ Created point cloud with {len(pcd.points):,} points")
    
    # Create geometries
    geometries = [pcd]
    
    # Add coordinate frames
    if pose is not None:
        coordinate_frames = create_coordinate_frames(
            camera_pose=pose, 
            bounds=bounds, 
            frame_type=frame_info.get('frame', 'camera')
        )
        
        for frame_name, frame_geom in coordinate_frames:
            geometries.append(frame_geom)
            print(f"✅ Added {frame_name}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Occupancy Map: {os.path.basename(npz_path)}", 
                     width=1200, height=800)
    
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    if background == "white":
        render_option.background_color = [1.0, 1.0, 1.0]
    elif background == "gray":
        render_option.background_color = [0.5, 0.5, 0.5]
    else:
        render_option.background_color = [0.0, 0.0, 0.0]
    
    # Set good viewing angle
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    if pose is not None:
        view_control.set_lookat(pose[:3])
    view_control.set_up([0, -1, 0])
    
    # Add frame legend information
    add_frame_legends(vis)
    
    print("\n" + "="*60)
    print("VIEWER INSTRUCTIONS")
    print("="*60)
    print("🖱️  Mouse controls:")
    print("   • Left click + drag: Rotate view")
    print("   • Right click + drag: Pan view") 
    print("   • Scroll wheel: Zoom in/out")
    print("⌨️  Keyboard shortcuts:")
    print("   • H: Show help")
    print("   • R: Reset view")
    print("   • S: Take screenshot")
    print("   • Q/ESC: Quit viewer")
    if pose is not None:
        frame_type = frame_info.get('frame', 'camera')
        print(f"🎯 {frame_type.title()} frame shown at robot location")
    print("🚪 Close window when done, then:")
    print("   • Press ENTER: Next occupancy map")
    print("   • Press 'b': Previous occupancy map")  
    print("   • Press 'q': Quit viewer")
    print("="*60)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

# ================= Simple Interactive Viewer =================

class SimpleViewer:
    """Simple viewer that shows occupancy maps one by one."""
    
    def __init__(self, npz_files: list, color_mapping: dict = None, 
                 label_mapping: dict = None, point_size: float = 8.0, 
                 background: str = "black"):
        self.npz_files = npz_files
        self.color_mapping = color_mapping
        self.label_mapping = label_mapping
        self.point_size = point_size
        self.background = background
        self.current_index = 0
        
        print(f"🎬 Loaded {len(self.npz_files)} occupancy maps")
    
    def run(self):
        """Run the simple interactive viewer."""
        while True:
            if self.current_index >= len(self.npz_files):
                print("🔚 Reached end of occupancy maps")
                break
            if self.current_index < 0:
                print("🔙 At beginning of occupancy maps")
                self.current_index = 0
                continue
            
            print(f"\n📍 Showing [{self.current_index + 1}/{len(self.npz_files)}]")
            
            # Show current occupancy map
            try:
                visualize_occupancy_map(
                    self.npz_files[self.current_index],
                    self.color_mapping,
                    self.label_mapping,
                    self.point_size,
                    self.background
                )
            except Exception as e:
                print(f"❌ Error loading occupancy map: {e}")
                self.current_index += 1
                continue
            
            # Get user input
            print(f"\n📍 Currently at [{self.current_index + 1}/{len(self.npz_files)}]")
            user_input = input("➡️  Action [ENTER=next, b=back, q=quit]: ").strip().lower()
            
            if user_input == 'q' or user_input == 'quit':
                print("👋 Exiting viewer")
                break
            elif user_input == 'b' or user_input == 'back':
                if self.current_index > 0:
                    self.current_index -= 1
                    print("⬅️  Going back")
                else:
                    print("🔙 Already at first occupancy map")
            else:  # Default: next (including empty input)
                if self.current_index < len(self.npz_files) - 1:
                    self.current_index += 1
                    print("➡️  Going forward")
                else:
                    print("🔚 At last occupancy map")
                    break

# ================= CLI Arguments =================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple semantic occupancy map viewer")
    
    parser.add_argument("--root_dir", type=str, required=True,
                       help="Path to TartanGround dataset root directory")
    parser.add_argument("--env", type=str, required=True,
                       help="Environment name (e.g., CastleFortress)")
    parser.add_argument("--traj", type=str, required=True,
                       help="Trajectory name (e.g., P001)")
    parser.add_argument("--skip_samples", type=int, default=1,
                       help="Skip every N samples (1 = use all)")
    parser.add_argument("--seg_rgb_path", type=str, default=DEFAULT_SEG_RGB_PATH,
                       help="Path to seg_rgbs.txt for color mapping")
    parser.add_argument("--point_size", type=float, default=8.0,
                       help="Point size for 3D visualization")
    parser.add_argument("--background", type=str, choices=["black", "white", "gray"], 
                       default="black", help="Background color")
    
    return parser.parse_args()

# ================= Main Function =================

def main():
    """Main function."""
    args = parse_args()
    
    print(f"🚀 Simple occupancy viewer for {args.env}/{args.traj}")
    
    # Get occupancy map files
    try:
        npz_files = get_occupancy_files(args.root_dir, args.env, args.traj, args.skip_samples)
        print(f"📁 Found {len(npz_files)} occupancy maps")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Load color mapping
    color_mapping = None
    seg_rgb_path = os.path.join(os.path.dirname(__file__), args.seg_rgb_path)
    if Path(seg_rgb_path).exists():
        color_mapping = load_rgb_mapping(seg_rgb_path)
        print(f"✅ Loaded color mapping ({len(color_mapping)} classes)")
    else:
        print(f"⚠️ Color mapping not found, using default colors")
    
    # Load label mapping
    label_mapping = None
    try:
        traj_path = find_trajectory_path(args.root_dir, args.env, args.traj)
        env_dir = os.path.dirname(os.path.dirname(traj_path))
        label_map_path = os.path.join(env_dir, 'seg_label_map.json')
        
        if Path(label_map_path).exists():
            label_mapping = load_class_labels(label_map_path)
            if label_mapping:
                print(f"✅ Loaded class labels ({len(label_mapping)} labels)")
    except Exception as e:
        print(f"⚠️ Could not load class labels: {e}")
    
    # Create and run viewer
    viewer = SimpleViewer(
        npz_files=npz_files,
        color_mapping=color_mapping,
        label_mapping=label_mapping,
        point_size=args.point_size,
        background=args.background
    )
    
    viewer.run()

if __name__ == "__main__":
    main()