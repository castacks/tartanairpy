
import copy
import torch
import torch.nn.functional as F
import math
import sys

from .debug import ( show_msg, show_obj, show_sum )

from .ftensor import ( FTensor, f_eye )
from .shape_struct import ShapeStruct

from .compatible_torch import torch_meshgrid

CAMERA_MODELS = dict()
LIDAR_MODELS = dict()

LOCAL_PI = math.pi

def deg2rad(deg):
    global LOCAL_PI
    return deg / 180.0 * LOCAL_PI

def register(dst):
    '''Register a class to a destination dictionary. '''
    def dec_register(cls):
        dst[cls.__name__] = cls
        return cls
    return dec_register

def make_object(typeD, argD):
    '''Make an object from type collection typeD. '''

    assert( isinstance(typeD, dict) ), f'typeD must be dict. typeD is {type(typeD)}'
    assert( isinstance(argD,  dict) ), f'argD must be dict. argD is {type(argD)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(argD)

    # Get the type.
    typeName = typeD[ d['type'] ]

    # Remove the type string from the input dictionary.
    d.pop('type')

    # Create the model.
    return typeName( **d )

def x2y2z_2_z_angle(x2, y2, z):
    '''
    Compute the angle (in radian) with respect to the z-axis.
    
    Arguments:
    x2 (Tensor or scalar): x**2.
    y2 (Tensor or scalar): y**2.
    z (Tensor or scalar): z.
    '''

    return torch.atan2( torch.sqrt( x2 + y2 ), z )

def xyz_2_z_angle(x, y, z):
    '''
    Compute the angle (in radian) with respect to the z-axis.

    Arguments:
    x (Tensor or scalar): x.
    y (Tensor or scalar): y.
    z (Tensor or scalar): z. 
    '''

    return x2y2z_2_z_angle(x**2.0, y**2.0, z)


class SensorModel(object):
    def __init__(self, name, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super().__init__()
        
        self.name = name
        
        self._ss = None # Initial value.
        self.ss = shape_struct # Update self._ss.
        
        self._device = None
        self.in_to_tensor = in_to_tensor
        self.out_to_numpy = out_to_numpy
    
    @staticmethod
    def make_shape_struct_from_repr(shape_struct):
        if isinstance( shape_struct, dict ):
            return ShapeStruct( **shape_struct )
        elif isinstance( shape_struct, ShapeStruct ):
            return shape_struct
        else:
            raise Exception(f'shape_struct must be a dict or ShapeStruct object. Get {type(shape_struct)}')
    
    @property
    def ss(self):
        return self._ss

    @ss.setter
    def ss(self, shape_struct):
        self._ss = SensorModel.make_shape_struct_from_repr(shape_struct)
    
    @property
    def shape(self):
        return self.ss.shape
        
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d

    def in_wrap(self, x):
        if self.in_to_tensor and not isinstance(x, torch.Tensor):
            return torch.as_tensor(x).to(device=self._device)
        else:
            return x

    def out_wrap(self, x):
        if self.out_to_numpy and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return x

    def __deepcopy__(self, memo):
        '''
        https://stackoverflow.com/questions/57181829/deepcopy-override-clarification#:~:text=In%20%22How%20to%20override%20the%20copy%2Fdeepcopy%20operations%20for,setattr%20%28result%2C%20k%2C%20deepcopy%20%28v%2C%20memo%29%29%20return%20result
        '''
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[ id(self) ] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result
    
    def get_rays_wrt_sensor_frame(self, shift=0.5):
        '''
        This function returns the rays shoting from the sensor and a valid mask.
        '''
        raise NotImplementedError()
        
class CameraModel(SensorModel):
    def __init__(self, name, fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(CameraModel, self).__init__(
            name=name, shape_struct=shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov_degree = fov_degree 
        self.fov_rad = deg2rad( self.fov_degree )
        
        self.padding_mode_if_being_sampled = 'zeros'
        
        # Will be populated once get_valid_mask() is called for the first time.
        self.valid_mask = None

    def _f(self):
        assert self.fx == self.fy
        return self.fx

    @property
    def f(self):
        # _f() is here to be called by child classes.
        return self._f()

    @SensorModel.device.setter
    def device(self, d):
        SensorModel.device.fset(self, d)
        
        if self.valid_mask is not None:
            if isinstance(self.valid_mask, torch.Tensor):
                self.valid_mask = self.valid_mask.to(device=d)

    def pixel_meshgrid(self, shift=0.5, normalized=False, skip_out_wrap=False, flatten=False):
        '''
        Get the meshgrid of the pixel centers.
        shift is applied along the x and y directions.
        If normalized is True, then the pixel coordinates are normalized to [-1, 1].
        '''
        
        H, W = self.shape
        
        x = torch.arange(W, dtype=torch.float32, device=self._device) + shift
        y = torch.arange(H, dtype=torch.float32, device=self._device) + shift
        
        # Compatibility issue with Jetpack 4.6 where
        # Python is 3.6, PyTorch is 1.8.
        xx, yy = torch_meshgrid(x, y, indexing='xy')
        
        xx, yy = xx.contiguous(), yy.contiguous()
        
        if normalized:
            xx = xx / W * 2 - 1
            yy = yy / H * 2 - 1
        
        if flatten:
            xx = xx.view((-1))
            yy = yy.view((-1))
        
        if skip_out_wrap:
            return xx, yy
        else:
            return self.out_wrap(xx), self.out_wrap(yy)
    
    def pixel_coordinates(self, shift=0.5, normalized=False, flatten=False):
        '''
        Get the pixel coordinates.
        shift is appllied along the x and y directions.
        If normalized is True, then the pixel coordinates are normalized to [-1, 1].
        '''
        xx, yy = self.pixel_meshgrid(shift=shift, normalized=normalized, skip_out_wrap=True, flatten=flatten)
        return self.out_wrap( torch.stack( (xx, yy), dim=0 ).contiguous() )

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number.
        
        Returns:
        A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number.
        
        Returns: 
        A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()
    
    def get_rays_wrt_sensor_frame(self, shift=0.5):
        '''
        This function returns the rays shooting from the sensor and a valid mask.
        '''
        pixel_coor = self.pixel_coordinates(shift=shift, flatten=True)
        return self.pixel_2_ray( pixel_coor )
    
    def _resize(self, new_shape_struct):
        '''
        In place operation. Child class should overload this method to perform appropriate operations.
        '''
        
        factor_x = new_shape_struct.W / self.ss.W
        factor_y = new_shape_struct.H / self.ss.H
        
        self.fx = self.fx * factor_x
        self.fy = self.fy * factor_y
        
        self.cx = self.cx * factor_x
        self.cy = self.cy * factor_y
        
        self.ss = new_shape_struct
    
    def get_resized(self, new_shape_struct):
        resized = copy.deepcopy(self)
        
        if self.ss == new_shape_struct:
            return resized
        
        # Child class may overload the _resize() method.
        device = self.device
        resized._resize(new_shape_struct)
        resized.device = device
        
        return resized
    
    def get_valid_bounary(self, n_points=100):
        '''
        Get an array of pixel coordinates that represent the boundary of the valid region.
        The result array is ordered.
        '''
        raise NotImplementedError()
    
    def get_valid_mask(self, flatten=False, force_update=False):
        # NOTE: Potential bug if force_update is False and flatten althers between two calls.
        if self.valid_mask is not None and not force_update:
            return self.valid_mask
        
        # Get the pixel coordinates of the pixel centers.
        pixel_coordinates = self.pixel_coordinates( shift=0.5, normalized=False, flatten=True )
        
        # Get the valid mask.
        _, valid_mask = self.pixel_2_ray( pixel_coordinates )
        
        if not flatten:
            valid_mask = valid_mask.view( self.shape )
            
        return valid_mask
        
# Usenko, Vladyslav, Nikolaus Demmel, and Daniel Cremers. "The double sphere camera model." In 2018 International Conference on 3D Vision (3DV), pp. 552-560. IEEE, 2018.
@register(CAMERA_MODELS)
class DoubleSphere(CameraModel):
    def __init__(self, xi, alpha, fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(DoubleSphere, self).__init__(
            'double_sphere', fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.alpha = alpha
        self.xi = xi

        # w1 and w2 are defined in the origial paper.
        w1, w2 = self.get_w1_w2()
        self.w1 = w1
        self.w2 = w2

        self.r2_threshold = 1 / ( 2 * self.alpha - 1 )

    def get_w1_w2(self):
        # Refer to the original paper for more information.
        w1 = self.alpha / ( 1 - self.alpha ) \
            if self.alpha <= 0.5 \
            else ( 1 - self.alpha ) / self.alpha
        
        w2 = ( w1 + self.xi ) / math.sqrt( 2 * w1 * self.xi + self.xi**2.0 + 1 )

        return w1, w2

    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
        # Do nothing.

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates.
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        pixel_coor = self.in_wrap(pixel_coor)
        
        # mx and my becomes float64 if pixel_coor.dtype is integer type.
        mx = ( pixel_coor[..., 0, :] - self.cx ) / self.fx
        my = ( pixel_coor[..., 1, :] - self.cy ) / self.fy
        r2 = mx**2.0 + my**2.0

        if ( self.alpha <= 0.5 ):
            valid_mask = torch.full((mx.size,), True)
        else:
            valid_mask = r2 <= self.r2_threshold

        # Suppress the waring from the invalid values.
        r2[torch.logical_not(valid_mask)] = 0

        mz = \
            ( 1 - self.alpha**2.0 * r2 ) / \
            ( self.alpha * torch.sqrt( 1 - ( 2*self.alpha - 1 ) * r2 ) + 1 - self.alpha )

        mz2 = mz**2.0

        t = ( mz * self.xi + torch.sqrt( mz2 + ( 1 - self.xi**2.0 ) * r2 ) ) / ( mz2 + r2 )
        x = t * mx
        y = t * my
        z = t * mz - self.xi

        # Need to deal with batch dim
        ray = torch.stack( (x, y, z), dim=-2 )

        # Compute the norm of ray along column direction.
        # norm_ray = torch.linalg.norm( ray, ord=2, dim=0, keepdim=True ) # Non-batched version
        norm_ray = torch.linalg.norm( ray, ord=2, dim=-2, keepdim=True )
        zero_mask = norm_ray == 0
        norm_ray[zero_mask] = 1

        # Normalize ray.
        ray = ray / norm_ray

        # Filter by FOV.
        a = xyz_2_z_angle( x, y, z )
        valid_mask = torch.logical_and(
            valid_mask, 
            a <= self.fov_rad / 2.0
        )

        return self.out_wrap(ray), self.out_wrap(valid_mask)

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BXN if batched.
        '''

        point_3d = self.in_wrap(point_3d)

        # torch.split results in Bx1XN.
        x, y, z = torch.split( point_3d, 1, dim=-2 )

        # If Ftensor convert to Tensor.
        if isinstance(x, FTensor):
            x = x.tensor()
            y = y.tensor()
            z = z.tensor()

        x2 = x**2.0 # Note: this may promote x2 to torch.float64 if point_3d.dtype=torch.int. 
        y2 = y**2.0
        z2 = z**2.0

        d1 = torch.sqrt( x2 + y2 + z2 )
        d2 = torch.sqrt( x2 + y2 + ( self.xi * d1 + z )**2.0 )

        # Pixel coordinates. 
        t = self.alpha * d2 + ( 1 - self.alpha ) * ( self.xi * d1 + z )
        px = self.fx / t * x + self.cx
        py = self.fy / t * y + self.cy
        if normalized:
            # px = px / ( self.ss.W - 1 ) * 2 - 1
            # py = py / ( self.ss.H - 1 ) * 2 - 1
            # After changing to the new pixel coordinate definition.
            px = px / self.ss.W * 2 - 1
            py = py / self.ss.H * 2 - 1

        # pixel_coor = torch.stack( (px, py), dim=0 )
        pixel_coor = torch.cat( (px, py), dim=-2 )

        # Filter the invalid pixels.
        valid_mask = z > -self.w2 * d1

        # Filter by FOV.
        a = x2y2z_2_z_angle( x2, y2, z )
        valid_mask = torch.logical_and(
            valid_mask, 
            a <= self.fov_rad / 2.0
        )
        
        # This is for the batched dimension.
        valid_mask = valid_mask.squeeze(-2)

        return self.out_wrap(pixel_coor), self.out_wrap(valid_mask)
    
    def get_valid_bounary(self, n_points=1000):
        '''
        Get an array of pixel coordinates that represent the boundary of the valid region.
        The result array is ordered.
        '''
        
        # Unit length.
        unit_length = torch.ones( (1, n_points), device=self.device, dtype=torch.float32 )
        
        # Find the x, y, z coordinates.
        a_z = self.fov_rad / 2 # Angle w.r.t. the z-axis.
        z = unit_length * math.cos( a_z )
        
        # Projection of the unit length onto the xy-plane.
        r_xy = unit_length * math.sin( a_z )
        a_x = torch.linspace( -LOCAL_PI, LOCAL_PI, n_points ) # Angle w.r.t. the x-axis.
        x = r_xy * torch.cos( a_x )
        y = r_xy * torch.sin( a_x )
        
        # Create an array of 3D points at the unit sphere along the FOV.
        xyz = torch.cat( (x, y, z), dim=0 )
        
        pixel_coor, mask = self.point_3d_2_pixel(xyz)
        return self.out_wrap(pixel_coor)
    
    def __str__(self) -> str:
        return \
f'''{{
    "type": "{self.__class__.__name__}",
    "xi": {self.xi},
    "alpha": {self.alpha},
    "fx": {self.fx},
    "fy": {self.fy},
    "cx": {self.cx},
    "cy": {self.cy},
    "fov_degree": {self.fov_degree},
    "shape_struct": {self.ss},
    "in_to_tensor": {self.in_to_tensor},
    "out_to_numpy": {self.out_to_numpy}
}}'''

@register(CAMERA_MODELS)
class Equirectangular(CameraModel):
    # def __init__(self, cx, cy, shape_struct, lon_shift=0, open_span=False, in_to_tensor=False, out_to_numpy=False):
    def __init__(self, shape_struct, longitude_span=(-LOCAL_PI, LOCAL_PI), latitude_span=(-LOCAL_PI/2, LOCAL_PI/2), open_span=False, in_to_tensor=False, out_to_numpy=False):
        '''
        Used primarily for generating a panorama image from six pinhole images or ratating a
        panorama image for training.
        
        Since it is a camera model, the frame of the image is similar to other camera models.
        The z-axis is the optical axis and pointing forward. The x-axis is to the right. The 
        y-axis is downwards. The only difference is that to generate a panorama image similar
        to the ones generated by the Unreal Engine, we need to shift the longitude, normally 
        by -pi/2 angle. By "shift", we mean addition.
        
        NOTE: Currently, this is not the frame attached to the panorama obtained from Unreal Engine.
        When lon_shift is 0, the frame is the same as the normal definition, where z-axis is in the
        forward direction and the middle of the image is the zero longitude angle.
        '''

        # These two members are used during the call to set_members_by_shape_struct() and _resize().
        self.init_longitude_span = longitude_span
        self.init_latitude_span  = latitude_span
        
        self.open_span = open_span

        super(Equirectangular, self).__init__(
            'equirectangular', 1, 1, 0.5, 0.5, 360, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)
        # cx, cy will be updated.
        self.set_members_by_shape_struct(self.ss)

        # # Since lon_shift is applied by adding to the longitude span, the shifted frame has a measured
        # # rotation of -lon_shift, w.r.t. the original frame. Thus, the shifted frame has a rotation 
        # # that is measured in the original frame as
        # a = -self.lon_shift
        # self.R_ori_shifted = torch.Tensor(
        #     [ [ math.cos(a), -math.sin(a) ], 
        #       [ math.sin(a),  math.cos(a) ] ]
        #     ).to(dtype=torch.float32)
        
        # Override parent's variable.
        self.padding_mode_if_being_sampled = 'border'

    def set_members_by_shape_struct(self, shape_struct):
        # Full longitude span is [-pi, pi], with possible shift or crop.
        # Full latitude span is [-pi/2, pi/2], with possible crop. No shift.
        # The actual longitude span that all the pixels cover.
        self.lon_span_pixel = self.init_longitude_span[1] - self.init_longitude_span[0]
        assert self.lon_span_pixel <= 2 * LOCAL_PI, \
            f'logintude_span is over 2pi: {self.init_longitude_span}, longitude_span[1] - longitude_span[0] = {self.lon_span_pixel}. '
        
        # open_span is True means the last column of pixels do not have the same longitude angle as the first column.
        if self.open_span:
            # TODO: Potential bug if cx is not at the center of the image.
            # self.lon_span_pixel = 2*self.cx / ( 2*self.cx + 1 ) * self.lon_span_pixel
            self.lon_span_pixel = ( shape_struct.W - 1 ) / shape_struct.W * self.lon_span_pixel
        
        assert self.init_latitude_span[0] >= -LOCAL_PI / 2 and self.init_latitude_span[1] <= LOCAL_PI / 2, \
            f'latitude_span is wrong: {self.init_latitude_span}. '
        
        self.longitude_span = torch.Tensor( [ self.init_longitude_span[0], self.init_longitude_span[1] ] ).to(dtype=torch.float32)
        self.latitude_span  = torch.Tensor( [ self.init_latitude_span[0],  self.init_latitude_span[1]  ] ).to(dtype=torch.float32)

        # Figure out the virtual image center.
        self.cx = ( 0 - self.init_longitude_span[0] ) / self.lon_span_pixel * ( shape_struct.W - 1 )
        self.cy = ( 0 - self.init_latitude_span[0] ) / ( self.init_latitude_span[1] - self.init_latitude_span[0] ) * ( shape_struct.H - 1 )

    # TODO: Which is better: direct scale or calling set_members_by_shape_struct()?
    def _resize(self, new_shape_struct):
        self.set_members_by_shape_struct(new_shape_struct)
        self.ss = new_shape_struct

    @CameraModel.f.getter
    def f(self):
        print(f'Warning, the focal length of an {self.name} model has no meaning. ')
        return self._f()

    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
        
        self.longtitude_span = self.longitude_span.to(device=d)
        self.latitude_span   = self.latitude_span.to(device=d)
        # self.R_ori_shifted   = self.R_ori_shifted.to(dtype, device)

    def pixel_2_ray(self, pixel_coor):
        '''
        Assuming cx and cy is the center coordinates of the image. 
        Thus, the image shape is [ 2*cy + 1, 2*cx + 1 ]
        
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3XN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        pixel_coor = self.in_wrap(pixel_coor)
        
        # pixel_space_center = \
        #     torch.Tensor([ self.cx, self.cy ]).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        
        pixel_space_shape = \
            torch.Tensor([ self.ss.W, self.ss.H ]).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))

        angle_start = \
            torch.Tensor([ self.longitude_span[0], self.latitude_span[0] ]).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        
        angle_span = torch.Tensor(
            [ self.lon_span_pixel, self.latitude_span[1] - self.latitude_span[0] ]
        ).to(dtype=torch.float32, device=pixel_coor.device).view((2, 1))
        
        # lon_lat.dtype becomes torch.float64 if pixel_coor.dtype=torch.int.
        # TODO: Potential bug if pixel_space_center is not at the center of image.
        # lon_lat = pixel_coor / ( 2 * pixel_space_center ) * angle_span + angle_start
        # lon_lat = pixel_coor / ( pixel_space_shape - 1 ) * angle_span + angle_start # This is before changing the pixel coordinate definiton.
        lon_lat = pixel_coor / pixel_space_shape * angle_span + angle_start
        
        # Bx1xN after calling torch.split.
        longitute, latitute = torch.split( lon_lat, 1, dim=-2 )
        
        c = torch.cos(latitute)
        
        x = c * torch.sin(longitute)
        y =     torch.sin(latitute)
        z = c * torch.cos(longitute)
        
        # return self.out_wrap( torch.stack( (x, y, z), dim=0 )   ), \
        #        self.out_wrap( torch.ones_like(x).to(torch.bool) )
               
        return self.out_wrap( torch.cat( (x, y, z), dim=-2 )   ), \
               self.out_wrap( torch.ones_like(x.squeeze(-2)).to(torch.bool) )
    
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        point_3d = self.in_wrap(point_3d)
        
        show_sum(point_3d=point_3d, point_3d_abs=torch.abs(point_3d))
        
        # Input z and x coordinates.
        z_x_in = point_3d[ ..., [2, 0], : ]
        show_msg(f'z_x_in.dtype = {z_x_in.dtype}')
        
        # Compute latitude.
        # r = torch.linalg.norm(point_3d, dim=-2)
        # lat = torch.asin(point_3d[..., 1, :] / r)
        r   = torch.linalg.norm( z_x_in, dim=-2 )
        lat = torch.atan2( point_3d[..., 1, :], r )
        
        # Compute longitude.
        # z_x = self.R_ori_shifted @ point_3d[ ..., [2, 0], : ]
        z_x = z_x_in
        lon = torch.atan2( z_x[..., 1, :], z_x[..., 0, :] )
        
        latitude_range = self.latitude_span[1] - self.latitude_span[0]
        p_y = ( lat - self.latitude_span[0] ) / latitude_range # [ 0, 1 ]
        p_x = ( lon - self.longitude_span[0] ) / self.lon_span_pixel # [ 0, 1 ], closed span

        show_sum(r=r, lat=lat, lon=lon, lon_abs=torch.abs(lon), p_x=p_x, p_y=p_y)

        if normalized:
            # [-1, 1]
            p_y = p_y * 2 - 1
            p_x = p_x * 2 - 1
        else:
            # p_y = p_y * ( self.ss.H - 1 )
            # p_x = p_x * ( self.ss.W - 1 )
            # After changing the pixel coordinate definiton.
            p_y = p_y * self.ss.H
            p_x = p_x * self.ss.W
        
        show_sum(p_x=torch.abs(p_x), p_y=torch.abs(p_y))
        
        return self.out_wrap( torch.stack( (p_x, p_y), dim=-2 ) ), \
               self.out_wrap( torch.ones_like(p_x).to(torch.bool) )

    def __str__(self) -> str:
        return \
f'''{{
    "type": "{self.__class__.__name__}",
    "init_longitude_span": {self.init_longitude_span},
    "init_latitude_span": {self.init_latitude_span}
    "open_span": {self.open_span},
    "lon_span_pixel": {self.lon_span_pixel},
    "longitude_span": {self.longitude_span},
    "latitude_span": {self.latitude_span},
    "cx": {self.cx},
    "cy": {self.cy},
    "padding_mode_if_being_sampled": {self.padding_mode_if_being_sampled},
    "shape_struct": {self.ss},
    "in_to_tensor": {self.in_to_tensor},
    "out_to_numpy": {self.out_to_numpy} }}'''

@register(CAMERA_MODELS)
class Ocam(CameraModel):
    EPS = sys.float_info.epsilon
    
    def __init__(self, poly_coeff, inv_poly_coeff, cx, cy, affine_coeff, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        '''
        The implementation is mostly based on 
        https://github.com/hyu-cvlab/omnimvs-pytorch/blob/3016a5c01f55c27eff3c019be9aee02e34aaaade/utils/ocam.py#L15
        
        When reading values for poly_coeff and inv_poly_coeff, make sure that the coefficients of 
        higher order are listed first. If these values are read from the yaml file provided by omnimvs
        model, then we need to reverse the order of the original data (also skip the first value that is
        showing the total number of coefficients).
        
        The CIF (Camera Image Frame) defined by Davide Scaramuzza (see below) if different than ours. 
        The CIF here is originally defined as z-backward, y-right, and x-downward. So we need to convert
        between our CIF and the CIF used by Davide when dealing with coordinates.
        
        Note that we are not fliping the order of cx and cy internally, meaning the input arguments must have the 
        correct order (fliped outside this class) and we need to use self.cy for Davide's x-axis.
        
        The Ocam model is described by Davide Scaramuzza at
        https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
        '''
        
        super().__init__('Ocam', 1, 1, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)
        
        # Polynomial coefficients starting from the highest degree.
        self.poly_coeff     = torch.as_tensor(poly_coeff).to(dtype=torch.float32)     # Only contains the coefficients.
        self.inv_poly_coeff = torch.as_tensor(inv_poly_coeff).to(dtype=torch.float32) # Only contains the coefficients.
        self.affine_coeff   = affine_coeff   # c, d, e
    
    @CameraModel.f.getter
    def f(self):
        print(f'Warning, the focal length of an {self.name} model has no meaning. ')
        return self._f()
    
    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
    
        self.poly_coeff     = self.poly_coeff.to(device=d)
        self.inv_poly_coeff = self.inv_poly_coeff.to(device=d)
    
    def _resize(self, new_shape_struct):
        # Currently not supported.
        raise NotImplementedError()
    
    @staticmethod
    def poly_eval(poly_coeff, x):
        '''
        Evaluate the polynomial.
        '''
        # Exponent.
        p = torch.arange(len(poly_coeff)-1, -1, -1, device=x.device).view((-1, 1))
        
        # Change shapes.
        poly_coeff = poly_coeff.view((-1, 1))
        
        # Consider the batch dimension.
        # x = x.view((1, -1))
        # N -> 1xN, BxN -> Bx1xN
        x = x.unsqueeze(-2)
        
        # return torch.sum( poly_coeff * x ** p, dim=0 )
        return torch.sum( poly_coeff * x ** p, dim=-2 )
        
    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number. 
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''

        pixel_coor = self.in_wrap(pixel_coor).to(dtype=torch.float32)

        p = torch.zeros_like(pixel_coor, device=pixel_coor.device)
        
        # We need to use Davide's definition of the coordinate system.
        p[..., 0, :] = pixel_coor[..., 1, :] - self.cy
        p[..., 1, :] = pixel_coor[..., 0, :] - self.cx
        
        c, d, e = self.affine_coeff
        invdet = 1.0 / (c - d * e)
        
        A_inv = invdet * torch.Tensor( [
            [  1, -d ], 
            [ -e,  c ] ] ).to(dtype=pixel_coor.dtype, device=pixel_coor.device)
        
        # A_inv = invdet * torch.Tensor( [
        #     [ -d,  1 ], 
        #     [  c, -e ] ] ).to(dtype=pixel_coor.dtype, device=pixel_coor.device)

        p = A_inv @ p
        
        x = p[..., 0, :]
        y = p[..., 1, :]
        
        rho = torch.sqrt( x**2 + y**2 )

        z = Ocam.poly_eval( self.poly_coeff, rho )
        
        # theta is angle from the optical axis.
        theta = torch.atan2(rho, -z)
        
        # Convert back to our coordinate system.
        # out   = torch.stack((x, y, -z), dim=0)
        # out   = torch.stack((y, x, -z), dim=0)
        out   = torch.stack((y, x, -z), dim=-2)
        
        max_theta = self.fov_rad / 2.0
        valid_mask = theta <= max_theta

        return self.out_wrap( out ), \
               self.out_wrap( valid_mask )
        
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''   
        
        point_3d = self.in_wrap(point_3d)
        
        # torch.split() will reserve the dimension.
        x_3d = point_3d[..., 0, :]
        y_3d = point_3d[..., 1, :]
        z_3d = point_3d[..., 2, :]
        
        norm  = torch.sqrt( x_3d**2 + y_3d**2 ) + Ocam.EPS
        theta = torch.atan2( -z_3d, norm )
        rho   = Ocam.poly_eval( self.inv_poly_coeff, theta )
        
        # max_theta check : theta is the angle from xy-plane in ocam, 
        # thus add pi/2 to compute the angle from the optical axis.
        theta = theta + LOCAL_PI / 2
        
        c, d, e = self.affine_coeff
        
        # We need to use Davide's definition of the coordinate system.
        y = x_3d / norm * rho
        x = y_3d / norm * rho
        x2 = x * c + y * d + self.cy
        y2 = x * e + y     + self.cx
        
        # Convert back to our coordinate system.
        if normalized:
            # y2 = y2 / ( self.ss.W - 1 ) * 2 - 1
            # x2 = x2 / ( self.ss.H - 1 ) * 2 - 1
            # After changing the pixel coordinate definition.
            y2 = y2 / self.ss.W * 2 - 1
            x2 = x2 / self.ss.H * 2 - 1
        
        out = torch.stack( (y2, x2), dim=-2 )
        
        return self.out_wrap( out ), \
               self.out_wrap( theta <= self.fov_rad / 2.0 )
    
@register(CAMERA_MODELS)
class Pinhole(CameraModel):
    def __init__(self, fx, fy, cx, cy, shape_struct, in_to_tensor=False, out_to_numpy=False):
        
        # Compute the FoV from the specified parameters.
        shape_struct = SensorModel.make_shape_struct_from_repr(shape_struct)
        fov_degree = 2 * math.atan2(shape_struct.W, 2 * fx) * 180.0 / LOCAL_PI
        
        super().__init__('Pinhole', fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.set_members_by_shape_struct(shape_struct)

    def set_members_by_shape_struct(self, shape_struct):
        # FoV for both longitude (x and width) and latitude (y and height).
        self.fov_degree_longitude = 2 * math.atan2(shape_struct.W, 2 * self.fx) * 180.0 / LOCAL_PI
        self.fov_degree_latitude  = 2 * math.atan2(shape_struct.H, 2 * self.fy) * 180.0 / LOCAL_PI
        print(f"Created a new Pinhole camera model with lon/lat FoV of {self.fov_degree_longitude, self.fov_degree_latitude} degrees.")
        
        # The (inverse) intrinsics matrix is fixed throughout, keep a copy here.
        self.intrinsics = torch.tensor(
            [ [self.fx, 0      , self.cx ],
              [ 0,      self.fy, self.cy ],
              [ 0,      0      ,      1.0] ] ).to(dtype=torch.float32, device=self._device)

        self.inv_intrinsics = torch.tensor(
            [ [1.0/self.fx, 0      , -self.cx / self.fx],
              [ 0,      1.0/self.fy, -self.cy / self.fy],
              [ 0,      0      ,                    1.0] ] ).to(dtype=torch.float32, device=self._device)

    def _resize(self, new_shape_struct):
        super()._resize(new_shape_struct) # self.ss is updated.
        self.set_members_by_shape_struct(new_shape_struct)

    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
        self.inv_intrinsics = self.inv_intrinsics.to(device=d)
        self.intrinsics = self.intrinsics.to(device=d)

    def pixel_2_ray(self, uv):
        '''
        Arguments:
        uv (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch size.
        
        Returns:
        A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''

        # Warp to desired datatype.
        uv = self.in_wrap(uv).to(dtype=torch.float32)
        
        # Convert to honmogeneous coordinates.
        uv1 = F.pad(uv, (0, 0, 0, 1), value=1)

        # Convert to camera-frame (metric).
        xyz = self.inv_intrinsics @ uv1

        # Normalize rays to be unit length.
        xyz = xyz / torch.linalg.norm(xyz, dim = -2, keepdims= True)

        # Mask points that are out of field of view.
        # Currently returning a mask of only ones as all pixels are assumed to be with valid values, and the projection out of the image frame of all pixels is valid.
        # The following if-statements match the dimensionality of batched inputs.
        # NOTE(yoraish): why should there be any??
        if len(uv.shape) == 2:
            mask = torch.ones(uv.shape[1], device = self.device)
        if len(uv.shape) == 3:
            mask = torch.ones((uv.shape[0], uv.shape[2]), device = self.device)
        
        return self.out_wrap(xyz), \
               self.out_wrap(mask)
        
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number.
        
        Returns: 
        A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        point_3d = self.in_wrap(point_3d)

        # Pixel coordinates projected from the world points. 
        uv_unnormalized = self.intrinsics @ point_3d

        # Normalize the homogenous coordinate-points such that their z-value is 1. The expression uv_unnormalized[..., -1:, :] keeps the dimension of the tensor, which is required by the division operation since PyTorch has trouble to broadcast the operation. 
        uv = torch.div(uv_unnormalized, uv_unnormalized[..., -1:, :])

        # Do torch.split results in Bx1XN.
        px, py, _ = torch.split( uv, 1, dim=-2 )

        if normalized:
            # Using shape - 1 is the way for cv2.remap() and align_corners=True of torch.nn.functional.grid_sample().
            # px = px / ( self.ss.W - 1 ) * 2 - 1
            # py = py / ( self.ss.H - 1 ) * 2 - 1
            # Using shape is the way for torch.nn.functional.grid_sample() with align_corners=False.
            px = px / self.ss.W * 2 - 1
            py = py / self.ss.H * 2 - 1

        pixel_coor = torch.cat( (px, py), dim=-2 )

        # Filter the invalid pixels by the image size. Valid mask takes on shape [B] x N
        # If normalized, require the coordinates to be in the range [-1, 1].
        if normalized:
            valid_mask_px = torch.logical_and(px < 1, px > -1)
            valid_mask_py = torch.logical_and(py < 1, py > -1)
        
        # If not normalized, require the coordinates to be in the range [0, W] and [0, H].
        else:
            valid_mask_px = torch.logical_and(px < self.ss.W, px > 0)
            valid_mask_py = torch.logical_and(py < self.ss.H, py > 0)

        valid_mask = torch.logical_and(valid_mask_py, valid_mask_px)

        # This is for the batched dimension.
        valid_mask = valid_mask.squeeze(-2)

        return self.out_wrap(pixel_coor), self.out_wrap(valid_mask)
        
    
    def __repr__(self) -> str:
        return f'''An instance of Pinhole CameraModel
        Height : {self.ss.shape[0]}
        Width : {self.ss.shape[1]}
        fx : {self.fx}
        fy : {self.fy}
        cx : {self.cx}
        cy : {self.cy}
        FoV degrees (lon/lat, y/x, h/w): {self.fov_degree_longitude}, {self.fov_degree_latitude}
        device: {self._device}'''

    def __str__(self) -> str:
        return f'''Pinhole
        Shape : {self.ss.shape}
        FoV degrees (lon/lat, y/x, h/w): {self.fov_degree_longitude}, {self.fov_degree_latitude}'''

class LiDAR(SensorModel):
    def __init__(self, name, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super().__init__(
            name=name,
            shape_struct=shape_struct, 
            in_to_tensor=in_to_tensor, 
            out_to_numpy=out_to_numpy )
    
        # Sensor frame is the same as camera frame. z-axis forward, x-axis right, and y-axis down.
        # Rotation matrix between sensor and LiDAR frames measured in the sensor frame.
        self.R_sensor_lidar = f_eye(3, f0='sensor', f1='lidar', rotation=True, dtype=torch.float32, device=self.device)
        
        # The following angles are measured in the LiDAR frame.
        # Azimuth and elevation angles. 
        # 3D tensor. 2 x n_scanlines x n_points_per_scanline. In radians.
        self.az_el = None
    
    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
        
        self.R_sensor_lidar = self.R_sensor_lidar.to(device=self._device)
        self.az_el = self.az_el.to(device=self._device)
    
    def get_rays_wrt_lidar_frame(self):
        '''
        Returns a single ftensor of shape 3xN, where N is the number of points of all the scan lines.
        '''
        raise NotImplementedError()
    
    def get_rays_wrt_sensor_frame(self, shift=0.5):
        '''
        This function returns the rays shoting from the sensor and a valid mask.
        shift is for compatibility with the camera model.
        '''
        
        rays_lidar_frame = self.get_rays_wrt_lidar_frame()
        rays_sensor_frame = self.R_sensor_lidar @ rays_lidar_frame
        valid_mask = torch.ones( rays_sensor_frame.shape[1] )
        
        return self.out_wrap( rays_sensor_frame.tensor() ), self.out_wrap( valid_mask )

    def measure_wrt_lidar_frame(self, distance_measure):
        '''
        distance_measure (Tensor): A Bx1xN Tensor contains the measured distance value
        for every element in self.az_el. B is batch number.
        '''
        rays_lidar_frame = self.get_rays_wrt_lidar_frame().tensor()
        return rays_lidar_frame.unsqueeze(0).repeat(distance_measure.shape[0], 1, 1) * distance_measure

@register(LIDAR_MODELS)
class Velodyne(LiDAR):
    def __init__(self, description, in_to_tensor=False, out_to_numpy=False):
        '''
        A list of dictionaries. The keys in teh dictionary are: E, resA, and offA. 
        They are the elevation angle, the resolution of the azimuth angle, and the offset of the azimuth angle.
        '''

        # Assuming all scan lines have the same number of points.
        n_points_per_scanline = int( 360.0 / description[0]['resA'] )
        n_scanlines = len(description)
        
        ss = ShapeStruct(H=n_scanlines, W=n_points_per_scanline)
        
        super().__init__(
            name='Velodyne',
            shape_struct=ss, 
            in_to_tensor=in_to_tensor, 
            out_to_numpy=out_to_numpy )
        
        R_sensor_lidar = torch.tensor(
            [ [ -1,  0,  0], 
              [  0,  0, -1], 
              [  0,  1,  0] ], dtype=torch.float32, device=self.device )
        
        # Override the parent.
        self.R_sensor_lidar = FTensor( R_sensor_lidar, f0='sensor', f1='lidar', rotation=True)
        
        # Populate the azimuth and elevation angle arrayes.
        self.az_el = torch.zeros( ( 2, n_scanlines, n_points_per_scanline ), dtype=torch.float32, device=self.device )
        
        eps = 1e-6
        
        for i, d in enumerate(description):
            E     = deg2rad( d['E']    )
            res_a = deg2rad( d['resA'] )
            off_a = deg2rad( d['offA'] )
            
            self.az_el[0, i, :] = \
                torch.arange( 0, 2 * LOCAL_PI - eps, res_a, dtype=torch.float32, device=self.device ) + off_a
            
            self.az_el[1, i, :] = E
            
        self.desc = description
    
    # Override the parent's function.
    def get_rays_wrt_lidar_frame(self):
        '''
        Returns a single ftensor of shape 3xN, where N is the number of points of all the scan lines.
        This is the Velodyne definition, with y-axis forward, x-right, and z-up.
        '''
        
        cos_a = torch.cos( self.az_el[0, :, :] )
        sin_a = torch.sin( self.az_el[0, :, :] )
        cos_E = torch.cos( self.az_el[1, :, :] )
        sin_E = torch.sin( self.az_el[1, :, :] )
        
        x = cos_E * sin_a
        y = cos_E * cos_a
        z = sin_E
        
        return FTensor( torch.stack( (x, y, z), dim=0 ).reshape( (3, -1) ), f0='lidar' )


@register(CAMERA_MODELS)
class LinearSphere(CameraModel):
    def __init__(self, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(LinearSphere, self).__init__(
            'LinearSphere', 
            fx=shape_struct.W,
            fy=shape_struct.H,
            cx=shape_struct.W / 2,
            cy=shape_struct.H / 2,
            fov_degree= fov_degree, 
            shape_struct= shape_struct, 
            in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        assert self.ss.H == self.ss.W, 'The shape of the linear camera model must be a square.'

    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates.
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number.
        
        Returns:
        ray: A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        
        pixel_coor = self.in_wrap(pixel_coor) # Bx2xN
        
        # mx and my becomes float64 if pixel_coor.dtype is integer type.
        # Convert pixel to angle on the unit sphere.

        mx = ( pixel_coor[..., 0, :] - self.ss.W / 2)
        my = ( pixel_coor[..., 1, :] - self.ss.H / 2)
        r = (mx**2.0 + my**2.0)**0.5

        # Compute theta and phi. Theta is the angle of 3D points from the z-axis. Phi is the angle of 3D points from the x-axis on the x-y plane.
        th = torch.atan2( my, mx )
        ph = r * self.fov_rad / self.ss.W

        # Valid rays are only those within the field of view.
        valid_mask = torch.abs(ph) <= self.fov_rad / 2.0

        # Compute the 3D points. Those are of unit length.
        x = torch.sin(ph) * torch.cos(th)
        y = torch.sin(ph) * torch.sin(th)
        z = torch.cos(ph)

        # Need to deal with batch dim
        ray = torch.stack( (x, y, z), dim=-2 )

        return self.out_wrap(ray), self.out_wrap(valid_mask)

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number. 
        
        Returns: 
        pixel_coor: A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        valid_mask: A (N,) Tensor representing the valid mask. BXN if batched.
        '''

        point_3d = self.in_wrap(point_3d)

        # torch.split results in Bx1XN.
        x, y, z = torch.split( point_3d, 1, dim=-2 )

        x2 = x**2.0 # Note: this may promote x2 to torch.float64 if point_3d.dtype=torch.int. 
        y2 = y**2.0
        z2 = z**2.0

        # Magnitude of the 3D point.
        R = (x2 + y2 + z2)**0.5

        # Theta (th) is the angle between the ray and the z-axis.
        # Phi (ph) is the angle between the ray and the x-axis, in the x-y plane.
        ph = torch.acos( z / R )
        th = torch.atan2( y, x )


        # The radius in the image plane.
        r = ph * self.ss.W / self.fov_rad

        # The pixel coordinates.
        # If any are FTensors, then convert to Tensor.
        if isinstance(th, FTensor):
            th = th.tensor()
            ph = ph.tensor()


        u = r * torch.cos(th) + self.ss.W / 2
        v = r * torch.sin(th) + self.ss.H / 2

        # Mask out the invalid pixels.
        valid_mask = (u >= 0) & (u < self.ss.W) & (v >= 0) & (v < self.ss.H)
        
        # Pixel coordinates.
        if normalized:
            u = u / self.ss.W * 2 - 1
            v = v / self.ss.H * 2 - 1

        pixel_coor = torch.cat( (u, v), dim=-2 )

        
        # This is for the batched dimension.
        valid_mask = valid_mask.squeeze(-2)

        return self.out_wrap(pixel_coor), self.out_wrap(valid_mask)
    
    def get_valid_bounary(self, n_points=1000):
        '''
        Get an array of pixel coordinates that represent the boundary of the valid region.
        The result array is ordered.
        '''
        
        # Unit length.
        unit_length = torch.ones( (1, n_points), device=self.device, dtype=torch.float32 )
        
        # Find the x, y, z coordinates.
        a_z = self.fov_rad / 2 # Angle w.r.t. the z-axis.
        z = unit_length * math.cos( a_z )
        
        # Projection of the unit length onto the xy-plane.
        r_xy = unit_length * math.sin( a_z )
        a_x = torch.linspace( -LOCAL_PI, LOCAL_PI, n_points ) # Angle w.r.t. the x-axis.
        x = r_xy * torch.cos( a_x )
        y = r_xy * torch.sin( a_x )
        
        # Create an array of 3D points at the unit sphere along the FOV.
        xyz = torch.cat( (x, y, z), dim=0 )
        
        pixel_coor, mask = self.point_3d_2_pixel(xyz)
        return self.out_wrap(pixel_coor)
    
    def __str__(self) -> str:
        return \
f'''{{
    "type": "{self.__class__.__name__}",
    "fov_degree": {self.fov_degree},
    "shape_struct": {self.ss},
    "in_to_tensor": {self.in_to_tensor},
    "out_to_numpy": {self.out_to_numpy}
}}'''