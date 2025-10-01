import numpy as np
import torch
import argparse
import json

import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

from .flow_utils import *

import time
from functools import lru_cache

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



@lru_cache(maxsize=10)
def get_meshgrid_torch(W, H, device):
    u, v = torch.meshgrid(torch.arange(W, device=device).float(), torch.arange(H, device=device).float(), indexing="xy")

    uv = torch.stack((u, v), dim=-1)

    return uv

def flow_occlusion_post_processing(
    views, depth_error_threshold=0.1, depth_error_temperature=0.1, relative_depth_error_threshold=0.005, opt_iters=5
):
    """
    Generate flow supervision from pointmap, depthmap, intrinsics, and extrinsics.

    Args:
    - views (list[dict]): list of views, already batched by the dataloader
    """

    assert len(views) == 2, f"Expected 2 views, to compute flow to other view, got {len(views)} views"

    for view, other_view in zip(views, reversed(views)):

        if "flow" in view:
            assert "fov_mask" in view
            assert "depth_validity" in view
            assert "expected_normdepth" in view
            assert "norm_depthmap" in other_view

            B, H, W = view["fov_mask"].shape

            # print("Warning: flow already present in post processing, doing occlusion only")

            uv = get_meshgrid_torch(W, H, device=view["fov_mask"].device) + view["flow"].permute(0, 2, 3, 1)
            expected_norm_depth = view["expected_normdepth"][view["fov_mask"]]

            valid_mask = view["fov_mask"]
            norm_depth_in_other_view = other_view["norm_depthmap"]  # Cautious: this need to be normalized depth

        else:
            # project points from current view to other view
            # points are in a row-major format, so we need to transpose the last 2 dimensions
            B, H, W, C = view["pts3d"].shape

            world_to_other_camera = torch.linalg.inv(other_view["camera_pose"])

            current_points_in_other = (
                view["pts3d"].view(B, -1, C) @ world_to_other_camera[:, :3, :3].permute(0, 2, 1)
                + world_to_other_camera[:, :3, 3][:, None, :]
            )
            current_points_in_other = current_points_in_other.view(B, H, W, C)

            # project points to pixels
            uv, valid_mask = project_points_to_pixels_batched(current_points_in_other, other_view["camera_intrinsics"])

            # compute flow
            flow = uv - get_meshgrid_torch(W, H, device=uv.device)
            # flow[~valid_mask, :] = 0.0

            # compute occlusion based on depth reprojection error thresholding
            expected_norm_depth = torch.linalg.norm(current_points_in_other[valid_mask], dim=-1)
            norm_depth_in_other_view = z_depthmap_to_norm_depthmap_batched(
                other_view["depthmap"], other_view["camera_intrinsics"]
            )

            view["flow"] = flow.permute(0, 3, 1, 2)

            # compute correspondence validity
            view["fov_mask"] = valid_mask
            view["depth_validity"] = view["depthmap"] > 0

        error_threshold = (
            depth_error_threshold
            + relative_depth_error_threshold * expected_norm_depth
            - np.log(0.5) * depth_error_temperature
        )

        # to determine occlusion, we will threshold the error between the distance of projected point to the other camera center
        # v.s. the norm-depth value recorded in the otherview's depthmap at the projected pixel location. If they met, then the point
        # is the rendered point in the other view, and is not occluded. Otherwise, it is occluded.

        valid_uv = uv[valid_mask]
        view["valid_uv"] = valid_uv
        if (
            opt_iters > 0
        ):  # if opt_iters is 0, we will not optimize the uv_residual, and there are no need to create the optimizer and the residual tensor
            uv_residual = torch.zeros_like(
                valid_uv, requires_grad=True
            )  # we optimize uv_residual to estimate the lower bound of the depth error
            opt = torch.optim.Adam([uv_residual], lr=1e-1, weight_decay=1e-1)
            valid_uv = valid_uv + uv_residual
            opt.zero_grad()

        # select the possibly occluded pixels to check for non-occlusion
        possibly_occluded_mask = valid_mask.clone()
        possible_occlusion_in_valid_pixels = torch.ones(
            size=(valid_mask.sum(),), dtype=torch.bool, device=valid_mask.device
        )
        checked_uv = valid_uv  # [possible_occlusion_in_valid_pixels]
        checked_expected_norm_depth = expected_norm_depth  # [possible_occlusion_in_valid_pixels]
        checked_threshold = error_threshold  # [possible_occlusion_in_valid_pixels]

        opt_iteration = 0
        while True:
            # compute the reprojection error of the selected pixels and check if they are non-occluded
            reprojection_error = compute_reprojection_error(
                checked_uv, checked_expected_norm_depth, norm_depth_in_other_view, possibly_occluded_mask
            )

            occluded_selected_uv = reprojection_error >= checked_threshold

            # update the occlusion mask, uv_combined, and expected_norm_depth with the non_occluded_selected_uv
            possibly_occluded_mask_new = possibly_occluded_mask.clone()
            possibly_occluded_mask_new[possibly_occluded_mask] = occluded_selected_uv

            possible_occlusion_in_valid_pixels_new = possible_occlusion_in_valid_pixels.clone()
            possible_occlusion_in_valid_pixels_new[possible_occlusion_in_valid_pixels] = occluded_selected_uv

            possibly_occluded_mask = possibly_occluded_mask_new
            possible_occlusion_in_valid_pixels = possible_occlusion_in_valid_pixels_new

            if opt_iters == 0 or opt_iteration >= opt_iters:
                break

            # optimize the uv_residual
            loss = torch.sum(reprojection_error)
            loss.backward()
            opt.step()
            with torch.no_grad():
                uv_residual.clamp_(-0.707, 0.707)
            opt.zero_grad()

            opt_iteration += 1

            checked_uv = valid_uv[possible_occlusion_in_valid_pixels]
            checked_expected_norm_depth = expected_norm_depth[possible_occlusion_in_valid_pixels]
            checked_threshold = error_threshold[possible_occlusion_in_valid_pixels]

        # the non-occlsion mask is the invert of the possibly occluded mask
        non_occluded_mask = ~possibly_occluded_mask

        view["non_occluded_mask"] = non_occluded_mask & valid_mask

    # finally, account for depth invalidity in the other view
    for view, other_view in zip(views, reversed(views)):
        other_view_depth_validity = query_projected_mask(
            view["valid_uv"].detach(), other_view["depth_validity"], view["fov_mask"]
        )
        view["other_view_depth_validity"] = other_view_depth_validity

        # occlusion should be supervised at
        # 1. self depth is valid, once projected will land out of bound in the other view
        # OR
        # 2. self depth is valid, once projected will land in the bound of other view, landing position shows valid depth

        view["occlusion_supervision_mask"] = (view["depth_validity"] & (~view["fov_mask"])) | (
            view["fov_mask"] & other_view_depth_validity
        )

def query_projected_mask(uv, other_mask, uv_source_mask):
    """
    Compute reprojection error between the expected depth and the actual depthmap at the projected pixel location.

    Args:
    - uv (torch.Tensor): projected pixel locations
    - other_mask (torch.Tensor): boolean mask to query (B, H, W)
    - uv_source_mask (torch.Tensor): mask of pixels that corresponds to the uv pixels

    Returns:
    - torch.Tensor: reprojection error
    """

    B, H, W = other_mask.shape
    valid_pixels1_opt = uv.permute(1, 0) + 0.5  # since the pixel center is represented as 0.0

    # convert to normalized coordinates to apply grid sample
    shape_normalizer = torch.tensor([W, H], device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype).view(2, 1)
    valid_pixels1_opt_normalized = valid_pixels1_opt / shape_normalizer * 2 - 1

    valid_pixels1_opt_normalized = torch.clip(valid_pixels1_opt_normalized, -1, 1)

    # pad the queries to get uniform length for all images in batch
    pixels_in_each_example = torch.sum(uv_source_mask, dim=[1, 2])
    max_pixels = torch.max(pixels_in_each_example)
    sum_pixels = torch.cumsum(pixels_in_each_example, dim=0)

    padded_queries = torch.zeros(B, max_pixels, 2, device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype)
    valid_padded_mask = torch.arange(max_pixels, device=valid_pixels1_opt.device) < pixels_in_each_example[:, None]

    # fill the queries with the valid pixels
    padded_queries[valid_padded_mask] = valid_pixels1_opt_normalized.permute(1, 0)

    # apply grid sample
    sampled_mask = torch.nn.functional.grid_sample(
        other_mask.view(B, 1, H, W).float(),  # expand to BCHW
        grid=padded_queries.unsqueeze(1),
        # now grid have shape 1, 1, V, 2, in which V is the unrolled pixels
        # at which valid_mask is True. In other words, the results corresponds
        # to pixels True in valid_mask, unrolled row by row.
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[
        :, 0, 0, :
    ]  # output is BCHW, we only have the unrolled pixels in W dimension

    # select the non-padded values
    sampled_mask = sampled_mask[valid_padded_mask]

    output_mask = torch.zeros_like(uv_source_mask)
    output_mask[uv_source_mask] = sampled_mask.to(torch.bool)

    return output_mask

def compute_reprojection_error(uv, expected_depth, actual_depthmap, possibly_occluded_mask):
    """
    Compute reprojection error between the expected depth and the actual depthmap at the projected pixel location.

    Args:
    - uv (torch.Tensor): projected pixel locations
    - expected_depth (torch.Tensor): expected depth values for each uv
    - actual_depthmap (torch.Tensor): actual depthmap (B, H, W)
    - possibly_occluded_mask (torch.Tensor): mask of pixels that are possibly occluded

    Returns:
    - torch.Tensor: reprojection error
    """

    B, H, W = actual_depthmap.shape
    valid_pixels1_opt = uv.permute(1, 0) + 0.5  # since the pixel center is represented as 0.0

    # convert to normalized coordinates to apply grid sample
    shape_normalizer = torch.tensor([W, H], device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype).view(2, 1)
    valid_pixels1_opt_normalized = valid_pixels1_opt / shape_normalizer * 2 - 1

    valid_pixels1_opt_normalized = torch.clip(valid_pixels1_opt_normalized, -1, 1)

    # pad the queries to get uniform length for all images in batch
    pixels_in_each_example = torch.sum(possibly_occluded_mask, dim=[1, 2])
    max_pixels = torch.max(pixels_in_each_example)
    sum_pixels = torch.cumsum(pixels_in_each_example, dim=0)

    padded_queries = torch.zeros(B, max_pixels, 2, device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype)
    valid_padded_mask = torch.arange(max_pixels, device=valid_pixels1_opt.device) < pixels_in_each_example[:, None]

    # fill the queries with the valid pixels
    padded_queries[valid_padded_mask] = valid_pixels1_opt_normalized.permute(1, 0)

    # apply grid sample
    sampled_depth = torch.nn.functional.grid_sample(
        actual_depthmap.view(B, 1, H, W),  # expand to BCHW
        grid=padded_queries.unsqueeze(1),
        # now grid have shape 1, 1, V, 2, in which V is the unrolled pixels
        # at which valid_mask is True. In other words, the results corresponds
        # to pixels True in valid_mask, unrolled row by row.
        mode="bilinear",  # This is a very important design choice. Normally
        # we would not use bilinear interpolation for depth map, because it will
        # create non-existent points when interpolating between motion boundaries.
        # but here we are only using it to validate, which means its effect will not
        # propagate to the regression values. Using bilinear solves aliasing
        # at highly inclined angles.
        padding_mode="zeros",
        align_corners=False,
    )[
        :, 0, 0, :
    ]  # output is BCHW, we only have the unrolled pixels in W dimension

    # select the non-padded values
    sampled_depth = sampled_depth[valid_padded_mask]

    return torch.abs(sampled_depth - expected_depth)

def project_points_to_pixels_batched(pts_camera, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - pts_camera (BxHxWx3 torch.Tensor): points in camera coordinates
        - camera_intrinsics: a Bx3x3 torch.Tensor
    Returns:
        pixel coordinates (BxHxWx2 torch.Tensor), and a mask (BxHxW) specifying valid pixels.
    """
    camera_intrinsics = camera_intrinsics
    B, H, W, C = pts_camera.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert (camera_intrinsics[..., 0, 1] == 0.0).all()
    assert (camera_intrinsics[..., 1, 0] == 0.0).all()
    if pseudo_focal is None:
        fu = camera_intrinsics[..., 0, 0]
        fv = camera_intrinsics[..., 1, 1]
    else:
        assert pseudo_focal.shape == (B, H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[..., 0, 2]
    cv = camera_intrinsics[..., 1, 2]

    x, y, z = pts_camera[..., 0], pts_camera[..., 1], pts_camera[..., 2]

    uv = torch.zeros((B, H, W, 2), dtype=pts_camera.dtype, device=pts_camera.device)

    uv[..., 0] = fu.view(B, 1, 1) * x / z + cu.view(B, 1, 1)
    uv[..., 1] = fv.view(B, 1, 1) * y / z + cv.view(B, 1, 1)

    # Mask for valid coordinates
    valid_mask = (
        (z > 0.0) & (uv[..., 0] >= -0.5) & (uv[..., 0] < W - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] < H - 0.5)
    )
    # valid_mask = (z > 0.0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)

    return uv, valid_mask

def z_depthmap_to_norm_depthmap_batched(z_depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - z_depthmap (BxHxW array)
        - camera_intrinsics: a Bx3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """

    B, H, W = z_depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert (camera_intrinsics[..., 0, 1] == 0.0).all()
    assert (camera_intrinsics[..., 1, 0] == 0.0).all()
    if pseudo_focal is None:
        fu = camera_intrinsics[..., 0, 0]
        fv = camera_intrinsics[..., 1, 1]
    else:
        assert pseudo_focal.shape == (B, H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[..., 0, 2]
    cv = camera_intrinsics[..., 1, 2]

    rays = torch.ones((B, H, W, 3), dtype=z_depthmap.dtype, device=z_depthmap.device)

    uv = get_meshgrid_torch(W, H, device=z_depthmap.device)

    rays[..., 0] = (uv[..., 0].view(1, H, W) - cu.view(B, 1, 1)) / fu.view(B, 1, 1)
    rays[..., 1] = (uv[..., 1].view(1, H, W) - cv.view(B, 1, 1)) / fv.view(B, 1, 1)

    ray_norm = torch.linalg.norm(rays, axis=-1)

    return z_depthmap * ray_norm

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        # X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        X_world = X_cam @ (R_cam2world.T) + t_cam2world[None, None, :]

    return X_world, valid_mask

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = get_meshgrid(W, H)

    X_cam = np.zeros((H, W, 3), dtype=np.float32)

    X_cam[..., 0] = (u - cu) * depthmap / fu
    X_cam[..., 1] = (v - cv) * depthmap / fv
    X_cam[..., 2] = depthmap

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0

    return X_cam, valid_mask

@lru_cache(maxsize=10)
def get_meshgrid(W, H):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return u, v