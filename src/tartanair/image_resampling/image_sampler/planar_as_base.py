
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-05-06

import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .ocv_torch import ( ocv_2_torch, torch_2_ocv, TYPE_OCV_2_TORCH_MAP )
from ..mvs_utils import torch_meshgrid
from ..mvs_utils.ftensor import FTensor, f_eye

IDENTITY_ROT = f_eye(3, f0='raw', f1='fisheye', rotation=True, dtype=torch.float32)

INTER_MAP_OCV = {
    'linear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST
}

INTER_MAP = {
    'nearest': 'nearest',
    'linear': 'bilinear',
}

def to_torch(x, **kwargs):
    if not isinstance(x, torch.Tensor):
        return ocv_2_torch(x, **kwargs)
    
    if x.ndim == 4:
        return x
    elif x.ndim == 3:
        return x.unsqueeze(0)
    else:
        raise Exception(f'ndim must be 4 or 3 if x is Tensor. x.shape = {x.shape}.')

def is_originated_from_uint8(x):
    if isinstance(x, torch.Tensor):
        return x.dtype == TYPE_OCV_2_TORCH_MAP[np.uint8]
    else:
        return x.dtype == np.uint8

def input_2_torch(img, device):
    '''
    img can be a single image represented as a NumPy array, or it could
    be a collection of NumPy arrays, or it could already be a PyTorch Tensor.
    '''
    
    if isinstance(img, (list, tuple)):
        flag_uint8 = is_originated_from_uint8(img[0])
        return torch.cat( [ to_torch(i, keep_dtype=False) for i in img ], dim=0 ).to(device=device), flag_uint8
    else:
        flag_uint8 = is_originated_from_uint8(img)
        return to_torch(img, keep_dtype=False).to(device=device), flag_uint8

def torch_2_output(t, flag_uint8=True):
    if flag_uint8:
        return torch_2_ocv(t, scale=True, dtype=np.uint8)
    else:
        return torch_2_ocv(t, scale=False, dtype=np.float32)
    
def dummy_troch_2_output(t, flag_uint8=True):
    return t

class PlanarAsBase(object):
    def __init__(self, 
                 fov, 
                 camera_model, 
                 R_raw_fisheye=IDENTITY_ROT, 
                 cached_raw_shape=(1024, 2048),
                 convert_output=True):
        '''
        NOTE: If convert_output=False, then the output is a Tensor WITH the batch dimension.
        That is, the output is a 4D Tensor no matter whether the input is a single image
        or a collection of images.
        
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        camera_model (camera_model.CameraModel): Target camera model. 
        R_raw_fisheye (FTensor): The orientation of the fisheye camera.
        cached_raw_shape (two-element): The tentative shape of the support raw image. Use some positive values if not sure.
        convert_output (bool): True if the output needs to be converted to NumPy (OpenCV).
        '''
        # TODO: Fixe the naming of R_raw_fisheye. Target can be any kind of image.
        super(PlanarAsBase, self).__init__()
        
        self.fov = fov # Degree.

        self._device = 'cuda'

        self.camera_model = copy.deepcopy(camera_model)
        self.camera_model.device = self._device
        self.shape = self.camera_model.shape

        # The rotation matrix of the fisheye camera.
        # The notation is R_<to>_<from> or R_<measured in>_<be measured>.
        # This rotation matrix is the orientation of the fisheye camera w.r.t
        # the frame where we take the raw images. And the orientation is measured
        # in the raw image frame.
        self.R_raw_fisheye = R_raw_fisheye.to(device=self.device)
        
        self.cached_raw_shape = cached_raw_shape
        
        # Output converter.
        self.convert_input = input_2_torch
        if convert_output:
            self.convert_output = torch_2_output
        else:
            self.convert_output = dummy_troch_2_output

    def is_same_as_cached_shape(self, new_shape):
        return new_shape[0] == self.cached_raw_shape[0] and new_shape[1] == self.cached_raw_shape[1]

    @property
    def align_corners(self):
        return False
    
    @property
    def align_corners_nearest(self):
        return True
    
    def grid_sample(self, x, grid, mode, padding_mode='zeros'):
        align_corners = self.align_corners_nearest if mode == 'nearest' else self.align_corners
        return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device
        self.camera_model.device = device
        self.R_raw_fisheye = self.R_raw_fisheye.to(device=device)

    def mesh_grid_pixels(self, shape, dimensionless=False, flag_flatten=False):
        '''Get a mesh grid of the pixel coordinates. 
        shape (two-element): H, W.
        '''

        x = torch.arange( shape[1], dtype=torch.float32, device=self.device ) + 0.5 # W
        y = torch.arange( shape[0], dtype=torch.float32, device=self.device ) + 0.5 # H

        xx, yy = torch_meshgrid(x, y, indexing='xy')
        
        # Make contiguous.
        xx = xx.contiguous()
        yy = yy.contiguous()
        
        if dimensionless:
            xx = xx / shape[1] * 2 - 1
            yy = yy / shape[0] * 2 - 1

        if ( flag_flatten ):
            return xx.view((-1)), yy.view((-1))
        else:
            return xx, yy

    def get_xyz(self, back_shift_pixel=False):
        '''
        Compute the ray vectors for all valid pixels in the fisheye image.
        A ray vector is represented as a unit vector.
        All ray vectors will be transformed such that their coordiantes are
        measured in the raw frame where z-forward, x-right, and y-downward.
        
        Some pixels are not going to have valid rays. There is a mask of valid
        pixels that is also returned by this function.

        Returns:
            xyz (Tensor): 3xN, where N is the number of pixels.
            valid_mask (Tensor): 1xN, where N is the number of pixels. A binary mask.
        '''
        # The pixel coordinates.
        # # xx, yy = self.mesh_grid_pixels(self.shape, flag_flatten=True) # 1D.
        # xx, yy = self.camera_model.pixel_meshgrid( flatten=True )
        
        # if back_shift_pixel:
        #     xx -= 0.5
        #     yy -= 0.5
        
        # pixel_coor = torch.stack( (xx, yy), dim=0 ) # 2xN

        # xyz, valid_mask = \
        #     self.camera_model.pixel_2_ray(pixel_coor)
        
        shift = 0 if back_shift_pixel else 0.5
        xyz, valid_mask = self.camera_model.get_rays_wrt_sensor_frame(shift=shift)
        
        # xyz and valid_mask are torch.Tensor.
        # xyz = xyz.astype(np.float32)
        
        xyz = FTensor(xyz, f0='fisheye', f1=None).to(dtype=torch.float32)
        
        # Change of reference frame.
        xyz = self.R_raw_fisheye @ xyz
        return xyz, valid_mask
    
    def convert_dimensionless_torch_grid_2_ocv_remap_format(self, torch_grid: torch.Tensor, raw_shape: list):
        '''
        torch_grid: H x W x 2.
        raw_shape: The shape (H, W) of the suppport raw image.
        '''
        
        if torch_grid.ndim == 4:
            if torch_grid.shape[0] != 1:
                raise Exception(f'Only supports non-bacthed grid. Got torch_grid.shape = {torch_grid.shape}. ')
            
            torch_grid = torch_grid.squeeze(0)
        
        H, W = raw_shape
        
        s = torch_grid.detach().cpu().numpy().astype(np.float32)
        s = ( s + 1 ) / 2
        sx = s[..., 0] # Using slicing instead of np.split() to avoid doing squeezing.
        sy = s[..., 1]
        
        # Scale the dimensionless coordinates.
        sx = W / ( W - 0.5 ) * ( sx - 0.5 ) + 0.5
        sy = H / ( H - 0.5 ) * ( sy - 0.5 ) + 0.5
        
        # Convert to dimensional version.
        sx *= W - 1
        sy *= H - 1
        
        return sx, sy
    
    def compute_8_way_sample_msr_diff(self, s, valid_mask):
        '''
        This function computes the 8-way mean-square-root of the sampling location
        differences specified by s. 
        
        s (Tensor): The sampling location. N x H x W x 2.
        valid_mask: N x 1 x H x W.
        
        Returns:
        A N x 1 x H x W array showing the mean of 8-way msr diff. Measured in the unit of s.
        '''
        
        assert s.ndim == 4, f's.ndim = {s.ndim}'
        
        s = s.permute((0, 3, 1, 2)).contiguous()
        N, _, H, W = s.shape
        
        # Augment the s array by 1s.
        all_ones = torch.zeros(( N, 1, H, W ), dtype=s.dtype, device=self.device)
        all_ones[valid_mask] = 1

        a = torch.cat( ( s, all_ones ), dim=1 )
        
        # Make a sampling grid.
        # xx, yy = self.mesh_grid_pixels( (H, W), dimensionless=True )
        xx, yy = self.camera_model.pixel_meshgrid(normalized=True)
        grid = torch.stack( (xx, yy), dim=-1 ).unsqueeze(0).repeat(N, 1, 1, 1)
        
        shifts = [
            [  1,  0 ], # 0
            [  1,  1 ], # 1
            [  0,  1 ], # 2
            [ -1,  1 ], # 3
            [ -1,  0 ], # 4
            [ -1, -1 ], # 5
            [  0, -1 ], # 6
            [  1, -1 ], # 7
        ]
        
        shifts = torch.Tensor(shifts).to(dtype=torch.float32, device=self.device)
        shifts[:, 0] = shifts[:, 0] / W * 2
        shifts[:, 1] = shifts[:, 1] / H * 2
        
        acc_d = torch.zeros((N, 1, H, W), dtype=torch.float32, device=self.device)

        for shift in shifts:
            grid_shifted = grid + shift

            s_a = self.grid_sample( a, 
                                 grid_shifted, 
                                 mode='nearest', 
                                 padding_mode='border' )
            
            d = ( s[:, :2, :, :] - s_a[:, :2, :, :] ) * s_a[:, 2, :, :].unsqueeze(1)
            d = torch.linalg.norm( d, dim=1, keepdim=True )
            acc_d = d + acc_d
            
        return acc_d / shifts.shape[0]