# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
sys.path.append(os.path.abspath('..'))


project = 'TartanAir'
copyright = 'Carnegie Mellon University, 2023, Wenshan Wang, Yaoyu Hu, Yuheng Qiu, Shihao Shen, Yorai Shaoul.'
author =  'Wenshan Wang, Yaoyu Hu, Yuheng Qiu, Shihao Shen, Yorai Shaoul.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ["colorama", "cupy", "cupy_cuda117", "kornia", "matplotlib", "networkx", "numba", "numpy", "cv2", "Pillow", "plyfile", "pyquaternion", "pytransform3d", "PyYAML", "yaml", "scipy", "torch", "torchvision", "tqdm", "SixPlanarNumba", ".downloader", ".dataset", ".customizer", ".lister", ".visualizer", ".iterator", ".evaluator", ".reader", "TartanAirDownloader", "TartanAirDataset", "TartanAirCustomizer", "TartanAirLister", "TartanAirVisualizer", "TartanAirIterator", "TartanAirEvaluator", "TartanAirTrajectoryReader", 'tartanair.image_resampling.image_sampler', 'tartanair.image_resampling.mvs_utils.camera_models', 'tartanair.image_resampling.mvs_utils.shape_struct', 'tartanair.image_resampling.mvs_utils.blend_function', 'wget', 'TartanAirDataLoader', 'MultiDatasets', 'data_cacher', 'pandas', 'tartanair.data_cacher.MultiDatasets', '.data_cacher.MultiDatasets', 'data_cacher.MultiDatasets', '.MultiDatasets']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
