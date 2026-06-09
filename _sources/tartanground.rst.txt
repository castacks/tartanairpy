##########################################################
TartanGround Dataset
##########################################################

Welcome to the **TartanGround** dataset documentation — a large-scale dataset for ground robot perception and navigation.

.. image:: https://img.shields.io/badge/Dataset-TartanGround-blue
   :alt: TartanGround Dataset Badge

.. image:: https://img.shields.io/badge/arXiv-2505.10696-red
   :alt: arXiv Paper Badge
   :target: https://arxiv.org/pdf/2505.10696

************************************************************
Quick Links
************************************************************

.. list-table::
   :widths: 30 70
   :header-rows: 0
   :class: borderless

   * - 🌐 **Dataset Webpage**
     - `TartanGround <https://tartanair.org/tartanground/>`_
   * - 📄 **Paper**
     - `arXiv:2505.10696 <https://arxiv.org/pdf/2505.10696>`_
   * - 💻 **GitHub Repository**
     - `castacks/tartanairpy <https://github.com/castacks/tartanairpy>`_
   * - 📊 **Metadata**
     - `Google Sheet <https://docs.google.com/spreadsheets/d/1d_px4Ss19OmrJrdOLwPsVNYe7Blcdmr6JKs0GdOORCg/edit?usp=sharing>`_

************************************************************
Installation
************************************************************

.. code-block:: bash
   :linenos:

   # 1. Create and activate conda environment
   conda create -n tartanground python=3.9
   conda activate tartanground

   # 2. Clone repository with all submodules
   git clone --recursive git@github.com:castacks/tartanairpy.git
   cd tartanairpy

   # 3. Ensure submodules are up to date
   git submodule update --init --recursive

   # 4. Install the package
    pip install -e .

.. note::
   Make sure you have ``git`` and ``conda`` installed on your system before proceeding.

************************************************************
Dataset Structure
************************************************************

The TartanGround dataset is organized hierarchically by **environment** and **robot type**. Each environment contains data for multiple robot platforms with comprehensive sensor modalities.

Directory Layout
============================================================

.. code-block:: text
   :linenos:

   TartanGround_Root/
   ├── AbandonedCable/
   │   ├── AbandonedCable_rgb.pcd      # Global RGB point cloud
   │   ├── AbandonedCable_sem.pcd      # Global Semantic point cloud
   │   ├── seg_label_map.json          # Semantic segmentation label map
   │   ├── Data_omni/                  # Omnidirectional robot data
   │   │   ├── P0000/
   │   │   │   ├── image_lcam_front/
   │   │   │   ├── depth_lcam_front/
   │   │   │   ├── seg_lcam_front/
   │   │   │   ├── imu/
   │   │   │   ├── lidar/
   │   │   │   ├── pose_lcam_front.txt
   │   │   │   ├── P0000_metadata.json
   │   │   │   ├── image_lcam_left/
   │   │   │   └── ...
   │   │   └── P00XX/
   │   ├── Data_diff/                  # Differential drive robot data
   │   │   ├── P1000/
   │   │   └── P10XX/
   │   └── Data_anymal/                # Quadrupedal robot data
   │       ├── P2000/
   │       └── P20XX/
   ├── AbandonedFactory/
   │   └── (same structure as above)
   └── ...

Robot Platforms
============================================================

.. list-table:: **Supported Robot Types**
   :widths: 15 25 25 35
   :header-rows: 1
   :class: robot-table

   * - **Robot**
     - **Type**
     - **Trajectory IDs**
     - **Description**
   * - ``omni``
     - Omnidirectional
     - ``P0000``, ``P0001``, ...
     - Holonomic movement in all directions
   * - ``diff``
     - Differential Drive
     - ``P1000``, ``P1001``, ...
     - Differential wheeled robot 
   * - ``anymal``
     - Quadrupedal
     - ``P2000``, ``P2001``, ...
     - Legged robot for complex terrains

Camera Configuration
============================================================

Each robot is equipped with a stereo **6-cam setup** providing full 360° coverage (similar to Tartanair-v2: :doc:`modalities`).

.. list-table:: **Camera Specifications**
   :widths: 25 75
   :header-rows: 1

   * - **Parameter**
     - **Value**
   * - **Camera Positions**
     - ``front``, ``left``, ``right``, ``back``, ``top``, ``bottom``
   * - **Field of View**
     - 90° (each camera)
   * - **Resolution**
     - 640×640 pixels
   * - **Stereo Configuration**
     - Left (``lcam_*``) and Right (``rcam_*``) pairs available

Available Camera Names
------------------------------------------------------------

.. code-block:: python

   # Left cameras
   ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back', 'lcam_top', 'lcam_bottom']
   
   # Right cameras  
   ['rcam_front', 'rcam_right', 'rcam_left', 'rcam_back', 'rcam_top', 'rcam_bottom']

Sensor Modalities
============================================================

The dataset provides multi-modal sensor data to support various robotic perception tasks:

* **RGB Images** (``image``): Color images for visual perception
* **Depth Maps** (``depth``): Accurate depth information for 3D scene understanding
* **Semantic Segmentation** (``seg``): Pixel-wise semantic labels
* **IMU Data** (``imu``): Inertial measurement unit data
* **LiDAR Point Clouds** (``lidar``): 3D point cloud data (32 Beam simulated LiDAR)
* **Robot Poses** (``meta``): Ground truth 6-DOF poses and metadata including robot height
* **Global Point Clouds**:

  * RGB Point Clouds (``rgb_pcd``): Colored 3D representations of entire environments
  * Semantic Point Clouds (``sem_pcd``): Point clouds with semantic labels
* **Segmentation Labels** (``seg_labels``): Label mappings for semantic segmentation tasks
* **ROS Bags** (``rosbag``): Proprioceptive data with joint states (available for ``anymal`` robot only)

Available Modalities
------------------------------------------------------------

.. code-block:: python

   # Complete list of available modalities
   ['image', 'meta', 'depth', 'seg', 'lidar', 'imu', 'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd']

.. note::
   The ``rosbag`` modality is only available for the ``anymal`` (quadrupedal) robot version.

************************************************************
Download Dataset
************************************************************

The TartanGround dataset can be downloaded using the **tartanairpy** Python toolkit. The repository includes ready-to-use examples in ``examples/download_ground_example.py``.

Download Examples
============================================================

**Example 1 – Download all modalities for specific environments and robots:**

.. code-block:: python

   import tartanair as ta

   ta.init('/path/to/tartanground/root')

   env = ["AbandonedFactory", "ConstructionSite", "Hospital"]

   ta.download_ground(
       env = env,
       version = ['omni', 'diff', 'anymal'],
       traj = [],
       modality = [
           'image', 'meta', 'depth', 'seg', 'lidar', 'imu',
           'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd'
       ],
       camera_name = ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back'],
       unzip = False
   )

**Example 2 – Download one trajectory from each environment (Omnidirectional robot only):**

.. code-block:: python

   import tartanair as ta

   ta.init('/path/to/tartanground/root')

   ta.download_ground(
       env = [],
       version = ['omni'],
       traj = ['P0000'],
       modality = [],
       camera_name = ['lcam_front'],
       unzip = False
   )

**Example 3 – Download semantic occupancy data only:**

.. code-block:: python

   import tartanair as ta

   ta.init('/path/to/tartanground/root')

   ta.download_ground(
       env = [],
       version = [],
       traj = [],
       modality = ['seg_labels', 'sem_pcd'],
       camera_name = [],
       unzip = False
   )

**Example 4 – Download entire dataset (~15 TB):**

.. code-block:: python

   import tartanair as ta

   ta.init('/path/to/tartanground/root')

   ta.download_ground(
       env = [],
       version = [],
       traj = [],
       modality = [],
       camera_name = [],
       unzip = False
   )

Multi-threaded Download
============================================================

For faster downloads, use the multi-threaded version:

.. code-block:: python

   import tartanair as ta

   ta.init('/path/to/tartanground/root')

   ta.download_ground_multi_thread(
       env = env,
       version = ['omni', 'diff', 'anymal'],
       traj = [],
       modality = [
           'image', 'meta', 'depth', 'seg', 'lidar', 'imu',
           'rosbag', 'sem_pcd', 'seg_labels', 'rgb_pcd'
       ],
       camera_name = ['lcam_front', 'lcam_right', 'lcam_left', 'lcam_back'],
       unzip = False,
       num_workers = 8
   )

Download Process Notes
============================================================

.. admonition:: Important Download Information
   :class: tip

   * **Pre-download Analysis:** The script lists all files and calculates total space requirements before downloading
   * **File Format:** Data is downloaded as zip files  
   * **Post-processing:** Use ``examples/unzip_ground_files.py`` to extract the downloaded data
   * **Multi-threading:** Adjust ``num_workers`` parameter based on your system capabilities and network bandwidth

.. seealso::
   **Full Examples:** Comprehensive download scripts are available in ``examples/download_ground_example.py`` in the `GitHub repository <https://github.com/castacks/tartanairpy>`_.

************************************************************
Dataset Statistics
************************************************************

.. list-table:: **TartanGround Dataset Overview**
   :widths: 40 60
   :header-rows: 1

   * - **Metric**
     - **Value**
   * - **Total Environments**
     - 63 diverse scenarios
   * - **Total Trajectories**
     - 878 trajectories
   * - **Robot Platforms**
     - 3 types (Omnidirectional, Differential, Quadrupedal)
   * - **Trajectory Distribution**
     - 440 omni, 198 diff, 240 legged
   * - **Total Samples**
     - 1.44 million samples
   * - **RGB Images**
     - 17.3 million images
   * - **Dataset Size**
     - ~16 TB
   * - **Samples per Trajectory**
     - 600-8,000 samples
   * - **Camera Resolution**
     - 640×640 pixels
   * - **Data Collection Frequency**
     - 10 Hz

Environment Categories
============================================================

.. list-table:: **Environment Types**
   :widths: 30 70
   :header-rows: 1

   * - **Category**
     - **Description**
   * - **Indoor**
     - Indoor spaces with complex layouts and lighting
   * - **Nature**
     - Natural environments with varied vegetation, natural terrain and seasons
   * - **Rural**
     - Countryside settings with varied topography
   * - **Urban**
     - City environments with structured layouts
   * - **Industrial/Infrastructure**
     - Construction sites and industrial facilities  
   * - **Historical/Thematic**
     - Heritage sites and specialized environments

************************************************************
Semantic Occupancy Maps
************************************************************

The TartanGround dataset supports **semantic occupancy prediction** research by providing tools to generate local 3D occupancy maps with semantic class labels. These maps are essential for training and evaluating neural networks that predict semantic occupancy from sensor observations.

Use Cases
============================================================

* **Semantic Occupancy Prediction**: Train networks to predict 3D semantic structure from 2D observations
* **3D Scene Understanding**: Evaluate spatial reasoning capabilities of perception models  
* **Navigation Planning**: Generate semantically-aware path planning datasets
* **Multi-modal Fusion**: Combine RGB, depth, and LiDAR data for 3D semantic mapping

Workflow Overview
============================================================

The semantic occupancy map generation follows a two-step process:

**Step 1: Generation** - Extract local occupancy maps around each robot pose

**Step 2: Visualization** - Inspect and validate the generated occupancy maps

Step 1: Generate Semantic Occupancy Maps
============================================================

Use the GPU-accelerated script to extract local semantic occupancy maps from global point clouds:

.. code-block:: bash

   # Generate occupancy maps for a specific trajectory
   python examples/subsample_semantic_pcd_gpu.py \
       --root_dir /path/to/tartanground/dataset \
       --env ConstructionSite \
       --traj P0000

**Key Parameters:**

.. code-block:: bash

   # Customize occupancy map properties
   python examples/subsample_semantic_pcd_gpu.py \
       --root_dir /path/to/dataset \
       --env ConstructionSite \
       --traj P0000 \
       --resolution 0.1 \                    # Voxel size in meters
       --x_bounds -20 20 \                   # Local X bounds [min, max]
       --y_bounds -20 20 \                   # Local Y bounds [min, max]  
       --z_bounds -3 5 \                     # Local Z bounds [min, max]
       --subsample_poses 10                  # Process every 10th pose

**Output:**

The script generates ``.npz`` files in ``{trajectory}/sem_occ/`` containing:

* **occupancy_map**: 3D voxel grid with semantic class IDs
* **pose**: Robot pose [x, y, z, qx, qy, qz, qw] 
* **bounds**: Local coordinate bounds
* **resolution**: Voxel resolution in meters
* **class_mapping**: Semantic class ID to RGB color mapping

.. admonition:: GPU Requirements
   :class: tip

   This script requires:
   
   * **NVIDIA GPU** with CUDA support
   * **CuPy library** for GPU acceleration  
   * **Sufficient GPU memory** (8GB+ recommended for large environments)

Step 2: Visualize Occupancy Maps
============================================================

Use the interactive viewer to inspect generated occupancy maps and verify their quality:

.. code-block:: bash

   # Visualize occupancy maps with navigation controls
   python examples/visualize_semantic_occ_local.py \
       --root_dir /path/to/tartanground/dataset \
       --env ConstructionSite \
       --traj P0000 \
       --skip_samples 100

**Customization Options:**

.. code-block:: bash

   # Customize visualization appearance
   python examples/visualize_semantic_occ_local.py \
       --root_dir /path/to/dataset \
       --env ConstructionSite \
       --traj P0000 \
       --skip_samples 50 \                   # Show every 50th occupancy map
       --point_size 12.0 \                   # Larger point visualization
       --background white                    # White background

Dataset Integration
============================================================

**File Structure:**

After running the generation script, your dataset structure will include:

.. code-block:: text

   TartanGround_Root/
   ├── ConstructionSite/
   │   ├── ConstructionSite_sem.pcd          # Global semantic point cloud (input)
   │   ├── Data_omni/
   │   │   ├── P0000/
   │   │   │   ├── pose_lcam_front.txt       # Robot poses (input)
   │   │   │   ├── sem_occ/                  # Generated occupancy maps
   │   │   │   │   ├── semantic_occupancy_000000.npz
   │   │   │   │   ├── semantic_occupancy_000001.npz
   │   │   │   │   └── ...
   │   │   │   └── (other sensor data)
   │   │   └── P0001/
   │   └── seg_label_map.json                # Semantic class names

**Loading Occupancy Maps in Python:**

.. code-block:: python

   import numpy as np

   # Load a single occupancy map
   data = np.load('semantic_occupancy_000000.npz', allow_pickle=True)
   
   occupancy_map = data['occupancy_map']     # 3D array with class IDs
   pose = data['pose']                       # Robot pose [x,y,z,qx,qy,qz,qw]  
   bounds = data['bounds']                   # Local bounds [x_min,x_max,y_min,y_max,z_min,z_max]
   resolution = data['resolution']           # Voxel size in meters
   class_mapping = data['class_mapping']     # Class ID to RGB mapping

   print(f"Occupancy map shape: {occupancy_map.shape}")
   print(f"Resolution: {resolution}m per voxel")
   print(f"Occupied voxels: {np.sum(occupancy_map > 0)}")

.. seealso::
   **Complete Examples:** Full parameter documentation and advanced usage examples are available in the `GitHub repository examples folder <https://github.com/castacks/tartanairpy/tree/main/examples>`_.

************************************************************
Citation
************************************************************

If you use the TartanGround dataset in your research, please cite:

.. code-block:: bibtex

   @article{patel2025tartanground,
     title={TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation},
     author={Patel, Manthan and Yang, Fan and Qiu, Yuheng and Cadena, Cesar and Scherer, Sebastian and Hutter, Marco and Wang, Wenshan},
     journal={arXiv preprint arXiv:2505.10696},
     year={2025}
   }

************************************************************
Support and Contact
************************************************************

For technical issues, questions, or bug reports, please open an issue on the `GitHub repository <https://github.com/castacks/tartanairpy/issues>`_.

For applications and interesting dataset uses, visit the `dataset webpage <https://tartanair.org/tartanground/>`_.