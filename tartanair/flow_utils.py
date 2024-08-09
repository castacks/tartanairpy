

import os
import json
import functools
from tqdm import tqdm
import concurrent.futures
import threading

from .image_resampling.mvs_utils.camera_models import Pinhole, DoubleSphere, LinearSphere, Equirectangular, PinholeRadTanFast, EUCM, CameraModel
from .image_resampling.mvs_utils.shape_struct import ShapeStruct
from .image_resampling.image_sampler.six_images import SixPlanarTorch
from .customizer import *

import sys
sys.path.append(".")
import pickle

CAMERA_MODEL_NAME_TO_CLASS_MAP = {
    'pinhole': Pinhole, 
    'doublesphere': DoubleSphere, 
    'linearsphere': LinearSphere, 
    'equirect': Equirectangular,
    'radtan': PinholeRadTanFast,
    'eucm': EUCM
}

MODALITY_TO_INTERPOLATION = {"image": "linear", "seg": "nearest", "depth": "blend"}

def generate_camera_model_object_from_config(config) -> CameraModel:

    # Create a deep copy.
    new_cam_model_params_copy = json.loads(json.dumps(config))

    # The name of the new camera model that is used to find the camera model class.
    new_cam_model_name = new_cam_model_params_copy['type']

    # The new camera model object. We need to convert the width and height to a ShapeStruct.
    new_cam_model_params_copy['params']['shape_struct'] = ShapeStruct(H=new_cam_model_params_copy['params']['height'], W=new_cam_model_params_copy['params']['width'])
    del new_cam_model_params_copy['params']['height']
    del new_cam_model_params_copy['params']['width']
    
    # Create the new camera model object.
    if not(new_cam_model_name in CAMERA_MODEL_NAME_TO_CLASS_MAP):
        print("class name %s not found when processing %s" % (new_cam_model_name, new_cam_model_name))
        exit()

    new_cam_model_object = CAMERA_MODEL_NAME_TO_CLASS_MAP[new_cam_model_name](**new_cam_model_params_copy['params'])

    return new_cam_model_object

def render_images_from_cubemap(cubemap_images, camera_model, rotation, image_modality=None, device="cpu"):
    
    sampler = SixPlanarTorch(camera_model.fov_degree, camera_model, rotation)
    sampler.device = device

    rendered_output = {}

    mask = None
    for modality_name, cube_image in cubemap_images.items():

        # print("rendering", modality_name)        
        modality = modality_name if (image_modality is None) else image_modality[modality_name]

        if MODALITY_TO_INTERPOLATION[modality] == "blend":
            # this requires images to be in torch tensor
            

            blend_func = BlendBy2ndOrderGradTorch(0.01) # hard code
            new_image, new_image_valid_mask = sampler.blend_interpolation(cube_image, blend_func, invalid_pixel_value=0)  
            
        else:
            # while this requires images to be in numpy

            new_image, new_image_valid_mask = sampler(cube_image, interpolation= MODALITY_TO_INTERPOLATION[modality])
    
        if isinstance(new_image_valid_mask, np.ndarray):
            new_image_valid_mask = torch.from_numpy(new_image_valid_mask)

        new_image_valid_mask = new_image_valid_mask.to(dtype=torch.bool)

        if mask is None:
            mask = new_image_valid_mask
        else:
            mask &= new_image_valid_mask

        rendered_output[modality] = torch.from_numpy(new_image).to(device)

    return rendered_output, mask
