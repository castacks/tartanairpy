
import math
from numba import (jit, cuda)
import numpy as np

from .six_images_common import OFFSETS

@jit(nopython=True)
def sample_coor(xyz, 
    valid_mask,
    offsets=OFFSETS):
    output = np.zeros( ( 2, xyz.shape[1] ), dtype=np.float32 )
    out_offsets = np.zeros_like(output)

    one_fourth_pi   = np.pi / 4
    half_pi         = np.pi / 2
    three_fourth_pi = one_fourth_pi + half_pi
    
    for i in range(xyz.shape[1]):
        # already rotated to the raw frame, z-forward, x-right, y-downwards.
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        a_y     = math.atan2(x, y) # Angle w.r.t. y+ axis projected to the x-y plane.
        a_z     = math.atan2(z, y) # Angle w.r.t. y+ axis projected to the y-z plane.
        azimuth = math.atan2(z, x) # Angle w.r.t. x+ axis projected to the z-x plane.

        if ( -one_fourth_pi < a_y and a_y < one_fourth_pi and \
             -one_fourth_pi < a_z and a_z < one_fourth_pi ):
            # Bottom.
            output[0, i] = min( max( ( 1 + x/y ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][2]
            out_offsets[1, i] = offsets[1][2]
        elif ( (three_fourth_pi < a_y or a_y < -three_fourth_pi) and \
               (three_fourth_pi < a_z or a_z < -three_fourth_pi) ):
            # Top.
            output[0, i] = min( max( ( 1 - x/y ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][4]
            out_offsets[1, i] = offsets[1][4]
        elif ( one_fourth_pi <= azimuth and azimuth < three_fourth_pi ):
            # Front.
            output[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 + y/z ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][0]
            out_offsets[1, i] = offsets[1][0]
        elif ( -one_fourth_pi <= azimuth and azimuth < one_fourth_pi ):
            # Right.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 + y/x ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][1]
            out_offsets[1, i] = offsets[1][1]
        elif ( -three_fourth_pi <= azimuth and azimuth < -one_fourth_pi ):
            # Back.
            output[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - y/z ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][5]
            out_offsets[1, i] = offsets[1][5]
        elif ( three_fourth_pi <= azimuth or azimuth < -three_fourth_pi ):
            # Left.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 )
            output[1, i] = min( max( ( 1 - y/x ) / 2, 0 ), 1 )
            out_offsets[0, i] = offsets[0][3]
            out_offsets[1, i] = offsets[1][3]
        else:
            raise Exception('xy invalid.')
            # raise Exception(f'x = {x}, y = {y}, z = {z}, a_y = {a_y}, a_z = {a_z}, one_fourth_pi = {one_fourth_pi}, half_pi = {half_pi}')

    output[:, np.logical_not(valid_mask)] = -1

    return output, out_offsets

@cuda.jit()
def k_sample_coor(
    output, out_offsets, xyz, offsets):
    # Prepare the index.
    x_idx    = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    x_stride = cuda.blockDim.x * cuda.gridDim.x

    # Constants.
    one_fourth_pi   = np.pi / 4
    half_pi         = np.pi / 2
    three_fourth_pi = one_fourth_pi + half_pi

    # Dimensionless image size.
    dls = 2 # 90 degrees of FOV.
    # dls = 1.9969 # 89.912 degrees of FOV.
    dls_half = dls / 2

    # Loop.
    for i in range( x_idx, xyz.shape[1], x_stride ):
        # already rotated to the raw frame, z-forward, x-right, y-downwards.
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        a_y     = math.atan2(x, y) # Angle w.r.t. y+ axis projected to the x-y plane.
        a_z     = math.atan2(z, y) # Angle w.r.t. y+ axis projected to the y-z plane.
        azimuth = math.atan2(z, x) # Angle w.r.t. x+ axis projected to the z-x plane.

        if ( -one_fourth_pi < a_y and a_y < one_fourth_pi and \
             -one_fourth_pi < a_z and a_z < one_fourth_pi ):
            # Bottom.
            output[0, i] = min( max( ( dls_half + x/y ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - z/y ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][2]
            out_offsets[1, i] = offsets[1][2]
        elif ( (three_fourth_pi < a_y or a_y < -three_fourth_pi) and \
               (three_fourth_pi < a_z or a_z < -three_fourth_pi) ):
            # Top.
            output[0, i] = min( max( ( dls_half - x/y ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - z/y ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][4]
            out_offsets[1, i] = offsets[1][4]
        elif ( one_fourth_pi <= azimuth and azimuth < three_fourth_pi ):
            # Front.
            output[0, i] = min( max( ( dls_half + x/z ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half + y/z ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][0]
            out_offsets[1, i] = offsets[1][0]
        elif ( -one_fourth_pi <= azimuth and azimuth < one_fourth_pi ):
            # Right.
            output[0, i] = min( max( ( dls_half - z/x ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half + y/x ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][1]
            out_offsets[1, i] = offsets[1][1]
        elif ( -three_fourth_pi <= azimuth and azimuth < -one_fourth_pi ):
            # Back.
            output[0, i] = min( max( ( dls_half + x/z ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - y/z ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][5]
            out_offsets[1, i] = offsets[1][5]
        elif ( three_fourth_pi <= azimuth or azimuth < -three_fourth_pi ):
            # Left.
            output[0, i] = min( max( ( dls_half - z/x ) / dls, 0 ), 1 )
            output[1, i] = min( max( ( dls_half - y/x ) / dls, 0 ), 1 )
            out_offsets[0, i] = offsets[0][3]
            out_offsets[1, i] = offsets[1][3]

def sample_coor_cuda( 
    xyz, 
    valid_mask,
    offsets=OFFSETS):

    output      = np.zeros((2, xyz.shape[1]), dtype=xyz.dtype)
    out_offsets = np.zeros_like(output)
    
    # Prepare the memory.
    d_xyz         = cuda.to_device(xyz)
    d_output      = cuda.to_device(output)
    d_out_offsets = cuda.to_device(out_offsets)
    d_offsets     = cuda.to_device(offsets)

    cuda.synchronize()
    k_sample_coor[(1024,1,1),(256,1,1)]( d_output, d_out_offsets, d_xyz, d_offsets )
    cuda.synchronize()

    output = d_output.copy_to_host()
    out_offsets = d_out_offsets.copy_to_host()
    # print(f'output.dtype = {output.dtype}')
    
    invalid_mask = np.logical_not(valid_mask)
    output[:, invalid_mask] = -1

    return output, out_offsets
