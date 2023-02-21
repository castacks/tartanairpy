
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .planar_as_base import (INTER_MAP_OCV, INTER_MAP, PlanarAsBase)
from .register import (SAMPLERS, register)

@register(SAMPLERS)
class FullViewRotation(PlanarAsBase):
    def __init__(self, camera_model, R_raw_fisheye, cached_raw_shape=(1024, 2048), convert_output=True):
        '''
        Note: Full view is the Unreal Engine's setting. It is NOT the same as the conventional 
        equirectangular projection. In Unreal Engine, the forward direction (that is where the 
        AirSim x-axis points when you set the camera pose) is located at the 3/4 of the image 
        width. 
        
        Note: R_raw_fisheye is the rotation matrix measured in the camera panorama frame. 
        The coordinates of a 3D point in the fisheye camera image frame x_f can 
        be transformed to the point in the camera panorama frame x_p by
        x_p = R_raw_fisheye @ x_f.

        R_raw_fisheye is following the naming converntion. This means that CIF's orientation
        is measure in CPF.

        Arguments:
        R_raw_fisheye (array): 3x3 rotation matrix. 
        camera_model (camera_model.CameraModel): The camera model. '''
        super().__init__(
            camera_model.fov_degree, 
            camera_model=camera_model, 
            R_raw_fisheye=R_raw_fisheye,
            cached_raw_shape=cached_raw_shape,
            convert_output=convert_output)
        # import ipdb; ipdb.set_trace()
        # Get the longitude and latitude coordinates.
        self.lon_lat, invalid_mask = self.get_lon_lat()

        # Reshape.
        self.lon_lat = self.lon_lat.permute((1, 0)).view((*self.shape, 2))
        self.invalid_mask = invalid_mask.view(self.shape)
        
        # The grid.
        self.grid = torch.zeros( (1, *self.shape, 2), dtype=torch.float32, device=self.device )
        self.grid[0, :, :, 0] = self.lon_lat[:, :, 0] / ( 2 * np.pi ) * 2 - 1
        self.grid[0, :, :, 1] = self.lon_lat[:, :, 1] / np.pi * 2 - 1
        
        # Backward compatibility with OpenCV remap().
        self.use_ocv = False
        self.ocv_remaps = self.convert_dimensionless_torch_grid_2_ocv_remap_format(self.grid.squeeze(0), self.cached_raw_shape)
        self.ocv_invalid_mask = self.invalid_mask.detach().cpu().numpy().astype(bool)

    @PlanarAsBase.device.setter
    def device(self, device):
        PlanarAsBase.device.fset(self, device)
        
        self.lon_lat = self.lon_lat.to(device=device)
        self.invalid_mask = self.invalid_mask.to(device=device)
        self.grid = self.grid.to(device=device)

    def get_lon_lat(self):
        # Get the rays in xyz coordinates in the fisheye camera image frame (CIF).
        # The valid mask is saved in self.temp_valid_mask for compatibility concern.
        xyz, valid_mask = self.get_xyz()

        # The distance projected into the xz plane in the panorama frame.
        d = torch.linalg.norm( xyz[[0, 2], :], dim=0, keepdim=True )

        # Longitude and latitude.
        lon_lat = torch.zeros( (2, xyz.shape[1]), dtype=torch.float32, device=self.device )
        # lon_lat[0, :] = np.pi - np.arctan2( xyz[2, :], xyz[0, :] ) # Longitude.
        # lon_lat[1, :] = np.pi - np.arctan2( d, xyz[1, :] )         # Latitude.
        lon_lat[0, :] = np.pi - torch.atan2( xyz[2, :], xyz[0, :] ) # Longitude.
        lon_lat[1, :] = np.pi - torch.atan2( d, xyz[1, :] )         # Latitude.

        return lon_lat, torch.logical_not( valid_mask )

    def execute_using_ocv(self, img, interpolation='linear'):
        global INTER_MAP_OCV
        
        flag_input_is_list = isinstance(img, (list, tuple) )
        if not flag_input_is_list:
            img = [img]
        
        # Loop.
        outputs_sampled = []
        outputs_mask = []
        for i in img:
            # Convert the PyTorch grid to OpenCV remap() format.
            if not self.is_same_as_cached_shape( i.shape[:2] ):
                self.ocv_remaps = self.convert_dimensionless_torch_grid_2_ocv_remap_format(self.grid.squeeze(0), i.shape[:2])
                self.cached_raw_shape = i.shape[:2]
            
            # Do the remap.
            sampled = cv2.remap(i, 
                                self.ocv_remaps[0], 
                                self.ocv_remaps[1], 
                                INTER_MAP_OCV[interpolation],
                                borderMode=cv2.BORDER_WRAP)
            
            # Handle the masked values.
            sampled[self.ocv_invalid_mask, ...] = 0
            
            outputs_sampled.append( sampled )
            outputs_mask.append( np.logical_not(self.ocv_invalid_mask) )
            
        if not flag_input_is_list:
            return outputs_sampled[0], outputs_mask[0]
        else:
            return outputs_sampled, outputs_mask

    def execute_using_torch(self, img, interpolation='linear'):
        global INTER_MAP

        # Convert to torch.Tensor.
        t, flag_uint8 = self.convert_input(img, self.device)

        # Get the shape of the input image.
        N, C = t.shape[:2]

        # # Get the sample location. 20220701.
        # sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        # sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)
        
        grid = self.grid.repeat( (N, 1, 1, 1) )

        # Sample.
        sampled = self.grid_sample( t, 
                                 grid, 
                                 mode=INTER_MAP[interpolation], 
                                 padding_mode='border')

        # Handle invalid pixels.
        sampled = sampled.view((N*C, *self.shape))
        sampled[:, self.invalid_mask] = 0
        sampled = sampled.view((N, C, *self.shape))

        return self.convert_output(sampled, flag_uint8), np.logical_not(self.invalid_mask.cpu().numpy().astype(bool))
    
    def __call__(self, img, interpolation='linear'):
        if self.use_ocv:
            return self.execute_using_ocv( img, interpolation )
        else:
            return self.execute_using_torch( img, interpolation )
    
    def blend_interpolation_ocv(self, img, blend_func):
        flag_input_is_list = isinstance(img, (list, tuple) )
        if not flag_input_is_list:
            img = [img]
        
        # Loop.
        outputs_sampled = []
        outputs_mask = []
        for i in img:
            # Convert the PyTorch grid to OpenCV remap() format.
            if not self.is_same_as_cached_shape( i.shape[:2] ):
                self.ocv_remaps = self.convert_dimensionless_torch_grid_2_ocv_remap_format(self.grid.squeeze(0), i.shape[:2])
                self.cached_raw_shape = i.shape[:2]
            
            # Do the remap.
            sampled_linear = cv2.remap(i, 
                                self.ocv_remaps[0], 
                                self.ocv_remaps[1], 
                                cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_WRAP)
            
            sampled_nearest = cv2.remap(i, 
                                self.ocv_remaps[0], 
                                self.ocv_remaps[1], 
                                cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_WRAP)
            
            f = blend_func( img )
            
            # Sample from the blend factor.
            f = cv2.remap(f,
                          self.ocv_remaps[0],
                          self.ocv_remaps[1],
                          cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_WRAP)
            
            sampled = f * sampled_nearest.astype(np.float32) + (1-f) * sampled_linear.astype(np.float32)
            
            # Handle the masked values.
            sampled[self.ocv_invalid_mask, ...] = 0
            
            outputs_sampled.append( sampled )
            outputs_mask.append( np.logical_not(self.ocv_invalid_mask) )
            
        if not flag_input_is_list:
            return outputs_sampled[0], outputs_mask[0]
        else:
            return outputs_sampled, outputs_mask
    
    def blend_interpolation_torch(self, img, blend_func):
        # Convert to torch.Tensor.
        t, flag_uint8 = self.convert_input(img, self.device)

        # Get the shape of the input image.
        N, C = t.shape[:2]
        
        grid = self.grid.repeat( (N, 1, 1, 1) )

        # Sample.
        sampled_linear = self.grid_sample( t, 
                                           grid, 
                                           mode='bilinear', 
                                           padding_mode='border')
        
        sampled_nearest = self.grid_sample( t, 
                                            grid,
                                            mode='nearest', 
                                            padding_mode='border')

        # Get the blend factor.
        f = blend_func( t )
        
        # Sample from the blend factor.
        f = self.grid_sample( f, 
                              grid, 
                              mode='nearest', 
                              padding_mode='border')
        
        # Blend.
        sampled = f * sampled_nearest + (1 - f) * sampled_linear

        # Handle invalid pixels.
        sampled = sampled.view((N*C, *self.shape))
        sampled[:, self.invalid_mask] = 0
        sampled = sampled.view((N, C, *self.shape))

        return self.convert_output(sampled, flag_uint8), np.logical_not(self.invalid_mask.cpu().numpy().astype(bool))
    
    def blend_interpolation(self, img, blend_func):
        '''
        This function blends the results of linear interpolation and nearest neighbor interpolation. 
        The user is supposed to provide a callable object, blend_func, which takes in img and produces
        a blending factor. The blending factor is a float number between 0 and 1. 1 means only nearest.
        
        blend_func needs to be able to handle the PyTorch version of img if self.use_ocv is False.
        '''
        if self.use_ocv:
            return self.blend_interpolation_ocv( img, blend_func )
        else:
            return self.blend_interpolation_torch( img, blend_func )
        

    def compute_mean_samping_diff(self, support_shape):
        '''
        support_shape is the shape of the support image.
        '''
        H, W = self.shape

        # # Get the sample location. 20220701.
        # sx = ( self.lon_lat[0, :] / ( 2 * np.pi ) * ( W - 1 ) ).reshape(self.shape)
        # sy = ( self.lon_lat[1, :] / np.pi * ( H - 1 ) ).reshape(self.shape)

        grid = self.grid.detach().clone()
        grid[0, :, :, 0] = ( grid[0, :, :, 0] + 1 ) / 2 * support_shape[1]
        grid[0, :, :, 1] = ( grid[0, :, :, 1] + 1 ) / 2 * support_shape[0]

        valid_mask = torch.logical_not( self.invalid_mask )

        d = self.compute_8_way_sample_msr_diff( grid, valid_mask.unsqueeze(0).unsqueeze(0) )
        
        return self.convert_output(d, flag_uint8=False), valid_mask.cpu().numpy().astype(bool)
