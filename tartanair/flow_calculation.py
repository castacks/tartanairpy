import numpy as np
import torch
import argparse
import json

import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

from .flow_utils import *

import time

FRONT_CAMERA = "front"

'''
    In this file, all transforms like B_T_A, B_R_A means the transform from A to B.

    i.e., you have a column vector v_A in frame A, then v_B = B_T_A @ v_A
'''



def interpolate_depth_pytorch(px_float, grid, depth_img):
    # TODO: change to blend interpolation

    px_x = (px_float[1, :] - 0.0).to(torch.int64)
    px_y = (px_float[0, :] - 0.0).to(torch.int64)

    mask = (0 <= px_x) & (px_x < depth_img.shape[0])
    mask &= (0 <= px_y) & (px_y < depth_img.shape[1])

    depth_output = torch.zeros_like(px_x, dtype=torch.float32)

    depth_output[mask] = depth_img[px_x[mask], px_y[mask]]
    
    return depth_output, mask

def sample_random_rotation_matrix():
    # Sample a random quaternion
    random_quaternion = R.random().as_quat()
    
    # Normalize the quaternion
    random_quaternion /= np.linalg.norm(random_quaternion)

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(random_quaternion).as_matrix()

    return torch.from_numpy(rotation_matrix).to(torch.float32)

def calculate_pairwise_flow(
    pose0, cam0_R_camsample0, depth0, mask0, camera_model_0, 
    pose1, cam1_R_camsample1, depth1, mask1, camera_model_1,
    device = "cuda"):
    
    # filter out erroneous depth value
    mask0 &= (depth0 != 0) # & (depth0 < 1e3)
    mask1 &= (depth1 != 0) # & (depth1 < 1e3)

    # calculate transform between cameras: from camera 0 to camera 1
    world_T_0 = np.eye(4)
    world_T_1 = np.eye(4)

    NED_R_cam = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    world_T_0[0:3, 0:3] = R.from_quat(pose0[3:]).as_matrix() @ NED_R_cam @ cam0_R_camsample0.numpy() # this accepts xyzw
    world_T_0[0:3, 3] = pose0[:3]

    world_T_1[0:3, 0:3] = R.from_quat(pose1[3:]).as_matrix() @ NED_R_cam @ cam1_R_camsample1.numpy() # this accepts xyzw
    world_T_1[0:3, 3] = pose1[:3]

    T_1_0 = torch.from_numpy(np.linalg.inv(world_T_1) @ world_T_0).to(dtype=torch.float32, device=device)

    # project the points to camera0's frame
    G0 = camera_model_0.pixel_coordinates(shift = 0.5, flatten = True)
    G0 = G0.to(device)

    valid_pixels_img0 = G0[:, mask0.reshape((-1,))]

    rays, rays_valid_mask = camera_model_0.pixel_2_ray(valid_pixels_img0) # rays is a 3xH*W tensor. 
    rays_valid_mask = rays_valid_mask.view((1, -1))

    rays = rays.to(device)
    rays_valid_mask = rays_valid_mask.to(device)

    assert rays_valid_mask.all() # mask0 should take care of this

    dist0 = depth0[mask0].view((1, -1))
    points0 = rays * dist0

    # project transform points0 into camera1's frame points1
    points1 = T_1_0[0:3,0:3] @ points0 + T_1_0[0:3,3:4]

    # project points1 into camera's pixel space according to camera model
    pixels1, projmask = camera_model_1.point_3d_2_pixel(points1)

    G1 = camera_model_1.pixel_coordinates(shift = 0.5, flatten = False)
    G1 = G1.to(device)

    valid_pixels_1 = pixels1[:, projmask]

    
    depth_value, mask = interpolate_depth_pytorch(valid_pixels_1, G1, depth1)

    # assert mask.all()

    depth_value_gt = torch.linalg.norm(points1[:, projmask], dim=0)


    # calculate depth reprojection error and final mask
    depth_error = torch.zeros(depth0.shape, dtype=torch.float32, device=device)
    mask_small = torch.zeros_like(depth_error[mask0], dtype=torch.float32)
    mask_small[projmask] = (depth_value_gt - depth_value)
    depth_error[mask0] = mask_small

    depth_error_mask = torch.zeros(mask0.shape, dtype=torch.bool, device=device)
    depth_error_mask[mask0] = projmask
    depth_error_mask &= mask0.to(device)

    # calculate valid flow
    valid_pixels_0 = G0[:, depth_error_mask.reshape((-1,))]

    # calculate flow image
    flow_image = torch.zeros(mask0.shape + (2,), dtype=torch.float32, device=device)
    flow_image[depth_error_mask, :] = (valid_pixels_1 - valid_pixels_0).T

    return depth_value_gt, depth_error, depth_error_mask, valid_pixels_0, valid_pixels_1, flow_image, world_T_0, world_T_1

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from scipy.ndimage import affine_transform, sobel
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.morphology import grey_erosion, grey_opening

import kornia
from kornia.filters import spatial_gradient
from kornia.morphology import opening

from einops import rearrange, reduce, repeat

def gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1, size // 2 + 1)
    y = torch.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / torch.sum(kernel)


def calculate_occlusion(
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    px0: torch.Tensor,
    px1: torch.Tensor,
    gt_depth: torch.Tensor,
    device = None,
    # 0.5 pixel offset
    pixel_05_offset: bool = True,
    # kernel size for morphological opening
    apply_morphological_opening: bool = True,
    kernel_size: int = 5,
    sigma: float = 1.0,
    # residual clipping
    max_residual: float = 0.5,
    # depth threshold for occlusion
    depth_start_threshold: float = 0.04,
    depth_temperature: float = 0.02,

    # relative error threshold
    apply_relative_error: bool = True,
    relative_error_tol: float = 0.001,
):
    
    if device is None:
        device = depth0.device

    # construct from torch
    depth0 = torch.from_numpy(depth0) if not isinstance(depth0, torch.Tensor) else depth0
    depth1 = torch.from_numpy(depth1) if not isinstance(depth1, torch.Tensor) else depth1
    px0 = torch.from_numpy(px0) if not isinstance(px0, torch.Tensor) else px0
    px1 = torch.from_numpy(px1) if not isinstance(px1, torch.Tensor) else px1
    gt_depth = torch.from_numpy(gt_depth) if not isinstance(gt_depth, torch.Tensor) else gt_depth

    # move everything to GPU
    depth0      =   depth0  .to(device, non_blocking=True)
    depth1      =   depth1  .to(device, non_blocking=True)
    px0         =   px0     .to(device, non_blocking=True)
    px1         =   px1     .to(device, non_blocking=True)
    gt_depth    =   gt_depth.to(device, non_blocking=True)

    if apply_morphological_opening:
        structuring_element = gaussian_kernel(kernel_size, sigma).to(device, non_blocking=True)

    # correct the 0.5 pixel offset
    if pixel_05_offset:
        px1_unoffset = px1 - 0.5
   
    # rearrange data into [B, C, H, W] format
    # depth0 = rearrange(depth0, 'h w -> () () h w')
    # depth1 = rearrange(depth1, 'h w -> () () h w')

    depth0 = depth0.unsqueeze(0).unsqueeze(0)
    depth1 = depth1.unsqueeze(0).unsqueeze(0)

    
    # 1. compute image gradient of depth1
    grad_img1 = spatial_gradient(depth1, normalized=False)
    grad_img1 = rearrange(grad_img1, 'b c d h w-> b (c d) h w')

    # 2. compute morphological opening of gradients
    if apply_morphological_opening: 
        grad_img1 = torch.sign(grad_img1) * opening(
            torch.abs(grad_img1),
            kernel=torch.ones_like(structuring_element), 
            structuring_element=structuring_element
        )

    
    # 3. interpolate depth at image 1
    _, _, H, W = depth1.shape

    # FIXME: do not know why, but this 0.5 is extremely important. without it the algorithm will not work as well. 
    sample_px1_normalized = (((px1_unoffset.T.reshape(1, 1, -1, 2) + 0.5) / torch.tensor([W, H], dtype=depth1.dtype, device=depth1.device)) * 2 - 1)

    depth_interp_1 = F.grid_sample(depth1, sample_px1_normalized, mode = 'bilinear', align_corners = False)
    depth_interp_1 = rearrange(depth_interp_1, 'b c x w -> b c x w')

    gaussian_blur = kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(1, 1))
    grad_img1 = gaussian_blur(grad_img1)

    # 4. interpolate gradients at px1 locations
    J = F.grid_sample(grad_img1, sample_px1_normalized, mode = 'bilinear', align_corners = False)
    J /= 8.0

    # 5. solve least squares problem
    r = 0.2 / (torch.sum(J ** 2, axis=0)) * J * (gt_depth - depth_interp_1)
    r = rearrange(r, 'b c x w -> b (c x) w')

    r_norm = torch.linalg.norm(r, axis=0)
    r[:, r_norm > max_residual] = r[:, r_norm > max_residual] / r_norm[r_norm > max_residual][None,:] * max_residual

    r = rearrange(r, 'b c w -> b () w c')

    # 6. obtain interpolation of the refined depth
    depth_interp_1 = F.grid_sample(depth1, sample_px1_normalized, mode = 'bilinear', align_corners = False)

    coeff = torch.tensor([[W / 2, H / 2]], dtype=depth1.dtype, device=depth1.device)
    depth_interp_1_refined = F.grid_sample(depth1, sample_px1_normalized + r / coeff, mode = 'bilinear', align_corners = False)
    
    # 7. compute occlusion
    depth_absolute_error = torch.abs(gt_depth - depth_interp_1)
    depth_absolute_error_refined = torch.abs(gt_depth - depth_interp_1_refined)

    depth_err = torch.minimum(depth_absolute_error, depth_absolute_error_refined)

    if apply_relative_error:
        occlusion = torch.exp(- torch.clip(
            depth_err - depth_start_threshold - relative_error_tol * gt_depth, 
            0, max=None
        ) / depth_temperature)
    else:
        occlusion = torch.exp(- torch.clip(depth_err - depth_start_threshold, 0, max=None) / depth_temperature)

    return occlusion


def sample_random_rotation_matrix():
    # Sample a random quaternion
    random_quaternion = R.random().as_quat()
    
    # Normalize the quaternion
    random_quaternion /= np.linalg.norm(random_quaternion)

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(random_quaternion).as_matrix()

    return torch.from_numpy(rotation_matrix).to(torch.float32)