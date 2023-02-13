
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2022-06-17
# 2022-12-26

from .numba_support_check import ( is_numba_supported, get_arch )

import cupy
import math
# import numpy as np
import time
import torch
# import torch.nn.functional as F

# Local package.
from .planar_as_base import ( PlanarAsBase, IDENTITY_ROT, INTER_MAP )
from .register import (SAMPLERS, register)

from .six_images_common import ( OFFSETS, make_image_cross_torch )


if is_numba_supported():
    from .six_images_numba import sample_coor

def dummy_debug_callback(blend_factor_ori, blend_factor_sampled):
    pass

@register(SAMPLERS)
class SixPlanarTorch(PlanarAsBase):
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
        self.xyz, self.valid_mask = self.get_xyz()
        self.valid_mask = self.valid_mask.view(self.shape)
        
        self.image_cross_layout = [3, 4]
        self.image_cross_layout_device = \
            torch.Tensor(self.image_cross_layout).to(dtype=torch.float32)
            
        self.OFFSETS_TORCH = torch.from_numpy(OFFSETS).to(dtype=torch.float32).permute((1,0)).contiguous()
        
        # === For the CuPy module. ===
        import os
        
        # Read the CUDA source.
        _CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        with open( os.path.join( _CURRENT_PATH, 'six_images_cupy.cu' ), 'r' ) as fp:
            cuda_source = fp.read()
        
        # Compile the CUDA source.
        cupy_module = cupy.RawModule(code=cuda_source)
        self.sample_coor_cuda = cupy_module.get_function('cu_sample_coor')
        
        # Make a copy of xyz. Make it become Nx3.
        self.xyz_T = self.xyz.permute((1,0)).contiguous()
        
        self.cuda_block_size = 256
        self.cuda_grid_size = int( math.ceil( self.xyz_T.shape[0]*self.xyz_T.shape[1] / self.cuda_block_size ) )
        
        # Create the grid.
        # First dummy self.grid value then create the real one.
        self.grid = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device=self.device)
        
        # Explicity set device to 'cuda' for faster computation during the construction.
        self.device = 'cuda'
        
        self.create_grid()

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.xyz = self.xyz.to(device=self.device)
        self.valid_mask = self.valid_mask.to(device=self.device)
        self.image_cross_layout_device = self.image_cross_layout_device.to(device=self.device)
        self.OFFSETS_TORCH = self.OFFSETS_TORCH.to(device=self.device)
        self.xyz_T = self.xyz_T.to(device=self.device)
        self.grid = self.grid.to(device=self.device)

    def __repr__(self):
        s = f'''SixPlanarTorch
fov = {self.fov}
shape = {self.shape}
'''
        return s

    def create_grid(self):
        # Get the sample locations.
        if ( self.device != 'cpu' ):
            start_time = time.time()
            # Allocate m and offsets.
            m = torch.zeros( (self.xyz_T.shape[0], 2), dtype=torch.float32, device=self.device )
            offsets = m.detach().clone()
            
            # Call the CUDA function.
            self.sample_coor_cuda(
                block=(self.cuda_block_size, ),
                grid=(self.cuda_grid_size, ),
                args=(
                    cupy.int32(self.xyz_T.shape[0]),
                    self.xyz_T.data_ptr(),
                    self.OFFSETS_TORCH.data_ptr(),
                    m.data_ptr(),
                    offsets.data_ptr()
                )
            )
            
            # Handle the valid mask.
            invalid_mask = torch.logical_not(self.valid_mask).view((-1,))
            m[invalid_mask, :] = -1 # NOTE: This might be a bug.
            d = cupy.cuda.Device()
            d.synchronize()
            # print(f'Time for CUDA: {time.time() - start_time}s. ')
        else:
            assert is_numba_supported(), \
                f'numba is not supported on {get_arch()}. Please set device=cuda. '
            
            m, offsets = sample_coor(
                self.xyz.cpu().numpy(), 
                self.valid_mask.view((-1,)).cpu().numpy().astype(bool))

        m[:, 0] = ( m[:, 0] + offsets[:, 0]) / self.image_cross_layout_device[1] * 2 - 1
        m[:, 1] = ( m[:, 1] + offsets[:, 1]) / self.image_cross_layout_device[0] * 2 - 1
        m = m.view( ( 1, *self.shape, 2 ) )
        
        self.grid = m

    def __call__(self, imgs, interpolation='linear', invalid_pixel_value=127):
        '''
        Arguments:
        imgs (dict of arrays or list of dicts): The six images in the order of front, right, bottom, left, top, and back.
        interpolation (str): The interpolation method, could be linear or nearest.
        invalid_pixel_value (int): The value of the invalid pixel. For RGB images, it is normally 127. For depth, it is -1.
        
        Returns:
        The generated fisheye image. The image might be inside a list.
        '''

        # Make the image cross.
        img_cross, flag_uint8, single_support_shape = \
            make_image_cross_torch( imgs, device=self.device )
        N = img_cross.shape[0]
        
        if not self.is_same_as_cached_shape( single_support_shape ):
            self.cached_raw_shape = single_support_shape
        
        sampled = self.grid_sample( 
                                img_cross, 
                                self.grid.repeat((N, 1, 1, 1)), 
                                mode=INTER_MAP[interpolation],
                                padding_mode='border')

        # Apply gray color on invalid coordinates.
        invalid_mask = torch.logical_not(self.valid_mask)
        
        if flag_uint8:
            invalid_pixel_value /= 255.0
        
        sampled[..., invalid_mask] = invalid_pixel_value

        start_time = time.time()
        output_sampled = self.convert_output(sampled, flag_uint8)
        output_mask = self.valid_mask.cpu().numpy().astype(bool)
        # print(f'Transfer from GPU to CPU: {time.time() - start_time}s. ')
        
        return output_sampled, output_mask

    def blend_interpolation(self, imgs, blend_func, invalid_pixel_value=127, debug_callback=dummy_debug_callback):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        '''
        
        # Make the image cross.
        img_cross, flag_uint8, single_support_shape = \
            make_image_cross_torch( imgs, device=self.device )
        N = img_cross.shape[0]
        
        if not self.is_same_as_cached_shape( single_support_shape ):
            self.cached_raw_shape = single_support_shape
        
        # Sample the images.
        grid = self.grid.repeat((N, 1, 1, 1))
        sampled_linear  = self.grid_sample( img_cross, grid, mode='bilinear', padding_mode='border' )
        sampled_nearest = self.grid_sample( img_cross, grid, mode='nearest' , padding_mode='border' )

        # The blend factor.
        bf = blend_func(img_cross)
        
        # Sample from the blend factor.
        f = self.grid_sample( bf, grid, mode='nearest', padding_mode='border' )
        
        # Debug.
        debug_callback(bf, f)
        
        # Blend.
        sampled = f * sampled_nearest + (1 - f) * sampled_linear

        # Apply gray color on invalid coordinates.
        invalid_mask = torch.logical_not(self.valid_mask)
        
        if flag_uint8:
            invalid_pixel_value /= 255.0
        
        sampled[..., invalid_mask] = invalid_pixel_value

        start_time = time.time()
        output_sampled = self.convert_output(sampled, flag_uint8)
        output_mask = self.valid_mask.cpu().numpy().astype(bool)
        # print(f'Transfer from GPU to CPU: {time.time() - start_time}s. ')
        
        return output_sampled, output_mask

    def compute_mean_samping_diff(self, support_shape):
        if not self.is_same_as_cached_shape( support_shape ):
            self.cached_raw_shape = support_shape
        
        # Get to [0, 1] range.
        m = self.grid.detach().clone()
        m = ( m + 1 ) / 2
        
        # Convert back to pixel coordinates.
        m[..., 0] *= support_shape[1]
        m[..., 1] *= support_shape[0]

        d = self.compute_8_way_sample_msr_diff( m, self.valid_mask.unsqueeze(0).unsqueeze(0) )
        
        return self.convert_output(d, flag_uint8=False), self.valid_mask.cpu().numpy().astype(bool)