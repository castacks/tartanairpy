# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2022-06-17
# 2022-12-26

import cv2
import numpy as np
import torch

# Local package.
from .planar_as_base import ( PlanarAsBase, IDENTITY_ROT, INTER_MAP_OCV, INTER_MAP )
from .register import (SAMPLERS, register)

from .six_images_common import (FRONT, make_image_cross_npy)
from .six_images_numba import ( sample_coor, sample_coor_cuda )

@register(SAMPLERS)
class SixPlanarNumba(PlanarAsBase):
    def __init__(self, fov, camera_model, R_raw_fisheye=IDENTITY_ROT, cached_raw_shape=(640, 640), convert_output=True):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model: A camera model for the fisheye camera.
        '''
        super().__init__(
            fov, 
            camera_model=camera_model, 
            R_raw_fisheye=R_raw_fisheye,
            cached_raw_shape=cached_raw_shape,
            convert_output=convert_output)
        
        # The 3D coordinates of the hyper-surface.
        # xyz, valid_mask = self.get_xyz(back_shift_pixel=True)
        xyz, valid_mask = self.get_xyz(back_shift_pixel=False)
        self.xyz = xyz.cpu().numpy()
        self.valid_mask = valid_mask.view(self.shape).cpu().numpy().astype(bool)
        
        # Explicity set the device to 'cuda' for better speed during construction.
        # Specifically, the call to self.update_remap_coordinates().
        self.device = 'cuda'
        
        # The remap coordinates.
        self.mx = None
        self.my = None
        self.update_remap_coordinates( self.cached_raw_shape )

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        # Do nothing.

    def __repr__(self):
        s = f'''SixPlanarNumba
fov = {self.fov}
shape = {self.shape}
'''
        return s

    def update_remap_coordinates(self, support_shape):
        H, W = support_shape
        
        if not isinstance(support_shape, np.ndarray):
            support_shape = np.array( [H, W], dtype=np.float32 ).reshape((2, 1))

        # Get the sample locations.
        if ( self.device != 'cpu' ):
            m, offsets = sample_coor_cuda( self.xyz, self.valid_mask.reshape((-1,)) )
        else:
            m, offsets = sample_coor(self.xyz, self.valid_mask.reshape((-1,)))

        # We need to properly scale the dimensionless values in m to use cv2.remap().
        # Refer to self.convert_dimensionless_torch_grid_2_ocv_remap_format() for consistency.
        m[0, :] = W / ( W - 0.5 ) * ( m[0, :] - 0.5 ) + 0.5
        m[1, :] = H / ( H - 0.5 ) * ( m[1, :] - 0.5 ) + 0.5

        m = m * ( support_shape - 1 ) + offsets * support_shape

        self.mx = m[0, :].reshape(self.shape)
        self.my = m[1, :].reshape(self.shape)

    def check_shape_and_make_image_cross(self, imgs):
        global FRONT

        # Get the original shape of the input images.
        img_shape = np.array(imgs[FRONT].shape[:2], dtype=np.float32).reshape((2, 1))
        
        # Check the input shape.
        if not self.is_same_as_cached_shape( img_shape ):
            self.update_remap_coordinates( img_shape )
            self.cached_raw_shape = img_shape
        
        # Make the image cross.
        img_cross = make_image_cross_npy( imgs )
        # cv2.imwrite('img_cross.png', img_cross)
        
        return img_cross

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (list of arrays): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image.
        '''

        global INTER_MAP_OCV
        
        img_cross = self.check_shape_and_make_image_cross(imgs)

        # Get the interpolation method.
        interp_method = INTER_MAP_OCV[interpolation]

        # Sample.
        sampled = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=interp_method )

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(self.valid_mask)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, self.valid_mask
    
    def blend_interpolation(self, imgs, blend_func, invalid_pixel_value=127):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        '''
        
        img_cross = self.check_shape_and_make_image_cross(imgs)

        # Sample.
        sampled_linear = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=cv2.INTER_LINEAR )
        
        sampled_nearest = cv2.remap( 
            img_cross, 
            self.mx, self.my, 
            interpolation=cv2.INTER_NEAREST )
        
        # Blend factor.
        f = blend_func(img_cross)
        
        # Sample from the blend factor.
        f = cv2.remap(
            f,
            self.mx, self.my,
            interpolation=cv2.INTER_NEAREST )
        
        sampled = f * sampled_nearest.astype(np.float32) + (1 - f) * sampled_linear.astype(np.float32)

        # Apply gray color on invalid coordinates.
        invalid = np.logical_not(self.valid_mask)
        sampled[invalid, ...] = invalid_pixel_value

        return sampled, self.valid_mask

    def compute_mean_samping_diff(self, support_shape):
        support_shape = np.array(support_shape, dtype=np.float32).reshape((2, 1))

        # Check the input shape.
        if not self.is_same_as_cached_shape( support_shape ):
            self.update_remap_coordinates( support_shape )
            self.cached_raw_shape = support_shape

        # compute_8_way_sample_msr_diff only supports torch now.
        s = torch.from_numpy(np.stack( (self.mx, self.my), axis=-1 )).unsqueeze(0).to(self.device)
        vm = torch.from_numpy(self.valid_mask).unsqueeze(0).unsqueeze(0).to(self.device)
        d = self.compute_8_way_sample_msr_diff( s, vm )
        
        return self.convert_output(d, flag_uint8=False), self.valid_mask
