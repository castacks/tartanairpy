
from colorama import Fore, Style

from .planar_as_base import ( PlanarAsBase, INTER_MAP, input_2_torch, torch_2_output )
from .camera_model_sampler import CameraModelRotation
from .register import (SAMPLERS, register)
from ..mvs_utils.ftensor import FTensor

def identity(x):
    return x

@register(SAMPLERS)
class GenericCameraModelSampler(CameraModelRotation):
    def __init__(self, camera_model_raw, camera_model_target, R_output_in_input, preprocessing = identity, postprocessing = identity):
        '''
        This is a wrapper class for the CameraModelRotation class.
        It generalizes the operation of the sampler to also take in a preprocessing function and a postprocessing function.

        The raw image is a planer image that described by a camera model. 
        We create the target image by preprocessing the input image, sampling from the preprocessed image (may no longer be an image), and postprocessing the result.

        R_output_in_input is the rotation matrix measured in the raw (input) image frame. 
        The coordinates of a 3D point in the target camera image frame x_f can 
        be transformed to the point in the raw image frame x_p by
        x_p = R_output_in_input @ x_f.

        R_output_in_input is following the naming convention. This means that CIF's orientation
        is measure in CPF.

        The camera model assumes that the raw image frame has its z-axis pointing forward,
        x-axis to the right, and y-axis downwards.

        Arguments:
        R_output_in_input (array): 3x3 rotation matrix. 
        camera_model_raw (camera_model.CameraModel): The camera model for the raw image. 
        camera_model_target (camera_model.CameraModel): The camera model for the target image. 
        preprocessing (function): A function that takes in a raw image and returns a preprocessed image.
        postprocessing (function): A function that takes in a preprocessed image and returns a postprocessed image.
        '''

        # # TODO: Use torch overall.
        # assert camera_model_raw.out_to_numpy, f'Currently only supports numpy version of raw camera model. '
        # assert not camera_model_target.out_to_numpy, f'Currently only supports pytorch version of target camera model. '

        super().__init__(
            camera_model_raw=camera_model_raw, 
            camera_model_target=camera_model_target, 
            R_raw_fisheye=R_output_in_input,
            convert_output=False
        )
        # The passed functions.
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing


    def __call__(self, img, interpolation='linear', invalid_pixel_value=127):
        '''
        img could be an array or a list of arrays.
        '''
        # Preprocess.
        img = self.preprocessing(img)

        # Sample.
        sampled, valid_mask = super().__call__(img, interpolation, invalid_pixel_value)

        # Postprocess.
        sampled, valid_mask = self.postprocessing(sampled, valid_mask)

        return sampled, valid_mask

