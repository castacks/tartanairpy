
import copy
from colorama import Fore, Style
import numpy as np

import torch
import torch.nn.functional as F

from .planar_as_base import ( PlanarAsBase, INTER_MAP )
from .register import (SAMPLERS, register)
from ..mvs_utils.ftensor import FTensor

@register(SAMPLERS)
class CameraModelRotation(PlanarAsBase):
    def __init__(self, camera_model_raw, camera_model_target, R_raw_fisheye, convert_output=True):
        '''
        The raw image is a planer image that described by a camera model. 
        
        We create the target image by sampling from the raw image.

        R_raw_fisheye is the rotation matrix measured in the raw image frame. 
        The coordinates of a 3D point in the target camera image frame x_f can 
        be transformed to the point in the raw image frame x_p by
        x_p = R_raw_fisheye @ x_f.

        R_raw_fisheye is following the naming converntion. This means that CIF's orientation
        is measure in CPF.

        The camera model assumes that the raw image frame has its z-axis pointing forward,
        x-axis to the right, and y-axis downwards.

        Arguments:
        R_raw_fisheye (array): 3x3 rotation matrix. 
        camera_model_raw (camera_model.CameraModel): The camera model for the raw image. 
        camera_model_target (camera_model.CameraModel): The camera model for the target image. '''

        # # TODO: Use torch overall.
        # assert camera_model_raw.out_to_numpy, f'Currently only supports numpy version of raw camera model. '
        # assert not camera_model_target.out_to_numpy, f'Currently only supports pytorch version of target camera model. '

        super().__init__(
            camera_model_target.fov_degree, 
            camera_model=camera_model_target, 
            R_raw_fisheye=R_raw_fisheye,
            cached_raw_shape=(1, 1),
            convert_output=convert_output)

        self.camera_model_raw = copy.deepcopy(camera_model_raw)
        self.camera_model_raw.device = self.device

        # Get the rays in xyz coordinates in the target camera image frame (CIF).
        # The rays has been already transformed to the target image frame.
        xyz, valid_mask_target = self.get_xyz()
        if isinstance(xyz, FTensor):
            xyz = xyz.tensor()

        # Get the sample location in the raw image.
        pixel_coord_raw, valid_mask_raw = camera_model_raw.point_3d_2_pixel( xyz, normalized=True )

        # Reshape the sample location.
        sx = pixel_coord_raw[0, :].reshape( camera_model_target.ss.shape )
        sy = pixel_coord_raw[1, :].reshape( camera_model_target.ss.shape )
        self.grid = torch.stack((sx, sy), dim=-1).unsqueeze(0)

        # Compute the valid mask.
        self.invalid_mask = torch.logical_not( torch.logical_and( valid_mask_raw, valid_mask_target ) )
        self.invalid_mask_reshaped = self.invalid_mask.view( camera_model_target.ss.shape )
        self.valid_mask_reshaped = torch.logical_not( self.invalid_mask_reshaped )
        self.grid[:, self.invalid_mask_reshaped, :] = -1 # NOTE: This might be a bug.

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.camera_model_raw.device = self.device
        self.invalid_mask = self.invalid_mask.to(device=self.device)
        self.invalid_mask_reshaped = self.invalid_mask_reshaped.to(device=self.device)
        self.valid_mask_reshaped = self.valid_mask_reshaped.to(device=self.device)
        self.grid = self.grid.to(device=self.device)

    def check_input_shape(self, img_shape):
        # Get the shape of the input image.
        H, W = img_shape[:2]
        ss = self.camera_model_raw.ss
        assert H == ss.H and W == ss.W, f'Wrong input image shape. Expect {ss}, got {img_shape[:2]}'

    def __call__(self, img, interpolation='linear', invalid_pixel_value=127):
        '''
        img could be an array or a list of arrays.
        '''
        # Convert to torch Tensor with [N, C, H, W] shape.
        img, flag_uint8 = self.convert_input(img, self.device)
        
        self.check_input_shape(img.shape[-2:])

        # Sample.
        sampled = self.grid_sample( img, 
                                 self.grid, 
                                 mode=INTER_MAP[interpolation], 
                                 padding_mode=self.camera_model_raw.padding_mode_if_being_sampled )

        # Handle invalid pixels.
        sampled[..., self.invalid_mask_reshaped] = invalid_pixel_value

        return self.convert_output(sampled, flag_uint8), self.valid_mask_reshaped.cpu().numpy().astype(bool)

    def blend_interpolation(self, img, blend_func, invalid_pixel_value=127):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        '''
        
        # Convert to torch Tensor with [N, C, H, W] shape.
        img, flag_uint8 = self.convert_input(img, self.device)
        
        self.check_input_shape(img.shape[-2:])

        # Sample.
        sampled_linear = self.grid_sample( 
                            img, 
                            self.grid, 
                            mode='linear', 
                            padding_mode=self.camera_model_raw.padding_mode_if_being_sampled )
        
        sampled_nearest = self.grid_sample( 
                            img, 
                            self.grid, 
                            mode='nearest', 
                            padding_mode=self.camera_model_raw.padding_mode_if_being_sampled )
        
        # Blend factor.
        f = blend_func( img )
        
        # Sample from the blend factor.
        f = self.grid_sample(
            f,
            self.grid,
            mode='nearest',
            padding_mode=self.camera_model_raw.padding_mode_if_being_sampled )

        sampled = f * sampled_nearest + (1 - f) * sampled_linear

        # Handle invalid pixels.
        sampled[..., self.invalid_mask_reshaped] = invalid_pixel_value

        return self.convert_output(sampled, flag_uint8), self.valid_mask_reshaped.cpu().numpy().astype(bool)

    def compute_mean_samping_diff(self, support_shape):
        self.check_input_shape(support_shape)

        valid_mask = torch.logical_not( self.invalid_mask )
        
        # Scale the grid back to the image pixel space.
        grid = self.grid.detach().clone()
        grid[..., 0] = ( grid[..., 0] + 1 ) / 2 * support_shape[1]
        grid[..., 1] = ( grid[..., 1] + 1 ) / 2 * support_shape[0]

        d = self.compute_8_way_sample_msr_diff( grid, valid_mask.unsqueeze(0).unsqueeze(0) )
        return self.convert_output(d, flag_uint8=False), valid_mask.cpu().numpy().astype(bool)
