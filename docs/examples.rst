

Examples
=====================================

TartanAir V2 is a flexible dataset when used with this Python package. Using it, you can download, iterate, and modify the raw data. Here are some examples of what you can do with it.

Download Example
-------------------------------------

Download via Python API
............................

.. code-block:: python

    import tartanair as ta

    # Initialize TartanAir.
    tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'
    ta.init(tartanair_data_root)

    # Download a trajectory.
    ta.download(env = "DesertGasStationExposure", 
                difficulty = ['easy', 'hard'], 
                trajectory_id = ["P000", "P002"],  
                modality = ['image', 'depth'],  
                camera_name = ['lcam_front'])


Download via a yaml config file
................................
.. code-block:: python
    
    ta.download(config = 'download_config.yaml')

The config file if of the following format:

.. code-block:: yaml

    env: ['DesertGasStationExposure']
    difficulty: ['easy']
    trajectory_id: ['P000', 'P002']
    modality: ['image']
    camera_name: ['lcam_front']

Customization Example
-------------------------------------

TartanAir V2 allows you to synthesize your own dataset by modifying the raw data. For example, by specifying a new camera model and generating images using it.

.. code-block:: python

    import tartanair as ta

    # For help with rotations.
    from scipy.spatial.transform import Rotation
    import numpy as np

    # Initialize TartanAir.
    tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'
    ta.init(tartanair_data_root)

    # Create your camera model(s).
    R_raw_new0 = Rotation.from_euler('y', 90, degrees=True).as_matrix().tolist()

    cam_model_0 =  {'name': 'pinhole', 
                    'raw_side': 'left', # TartanAir has two cameras, one on the left and one on the right. This parameter specifies which camera to use.
                    'params': 
                            {'fx': 320, 
                             'fy': 320, 
                             'cx': 320, 
                             'cy': 320, 
                             'width': 640, 
                             'height': 640},
                    'R_raw_new': R_raw_new0}

    R_raw_new1 = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix().tolist()

    cam_model_1 = {'name': 'doublesphere',
                    'raw_side': 'left',
                    'params':
                            {'fx': 300, 
                            'fy': 300, 
                            'cx': 500, 
                            'cy': 500, 
                            'width': 1000, 
                            'height': 1000, 
                            'alpha': 0.6, 
                            'xi': -0.2, 
                            'fov_degree': 195},
                    'R_raw_new': R_raw_new1}

    # Customize the dataset.
    ta.customize(env = 'SupermarketExposure', 
                 difficulty = 'easy', 
                 trajectory_id = ['P000'], 
                 modality = ['image'], 
                 new_camera_models_params=[cam_model_0, cam_model_1], 
                 num_workers = 2)

Data Iteration Example
-------------------------------------

Create a data iterator to get samples from the TartanAir V2 dataset. The samples include data in the specified modalities.

.. code-block:: python

    import tartanair as ta

    # Initialize TartanAir.
    tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'
    ta.init(tartanair_data_root)

    # Create iterator.
    ta_iterator = ta.iterator(env = 'ConstructionSite',
                              difficulty = 'easy',
                              trajectory_id = 'P000',
                              modality = 'image',
                              camera_name = 'lcam_front')

    for i in range(100):
        sample = next(ta_iterator)