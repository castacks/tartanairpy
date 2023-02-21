""""
Author: Yorai Shaoul
Date: 2023-01-07
Description: This is an example of how to use the GenericCameraModelSampler class to sample an optical flow image.
"""


# General imports.
import argparse
import os
from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch    
from scipy.spatial.transform import Rotation

# Local imports.
from ..generic_camera_model_sampler import GenericCameraModelSampler
from ...mvs_utils.camera_models import DoubleSphere, ShapeStruct, Pinhole
from .examples_utils.visualization_utils import flow_tensor_to_image

########################
# Visualization flag.
########################
parser = argparse.ArgumentParser(description='Visualize the optical flow.')
parser.add_argument('-v', '--vis', action='store_true', help='Visualize the optical flow.')
is_vis = parser.parse_args().vis

########################
# Load example.
########################
# Start with loading an example optical flow image.
# This is a 2-channel image, where the first channel is the x-component of the 
# optical flow, and the second channel is the y-component of the optical flow.
Fin = torch.tensor(torch.load('image_resampling/image_sampler/examples/data/flow_img.pt')).unsqueeze(0).permute(0,3,1,2).cuda() # Shape is (1, 2, H, W).

print(Fore.GREEN + "Loaded optical flow of shape {}".format(Fin.shape) + Style.RESET_ALL)   

# Show the flow to make sure it's okay.
if is_vis:
# Add a channel to the input.
    plt.figure(1)
    Fin_show = torch.cat([Fin, torch.ones_like(Fin[:,0:1,:,:])], dim=1).int()
    plt.imshow(flow_tensor_to_image(Fin.squeeze(0)).transpose(1,2,0))
    # plt.imshow(Fin_show.squeeze(0).cpu().numpy().transpose(1,2,0).astype(np.uint8) )
    plt.title("Optical Flow INPUT of shape {}".format(Fin_show.shape))

    # Draw arrows on the flow image.
    for i in range(0, Fin.shape[2], 20):
        for j in range(0, Fin.shape[3], 20):
            plt.arrow(j, i, Fin.cpu().numpy()[0,0,i,j], Fin.cpu().numpy()[0,1,i,j], color='r', head_width=5, head_length=3)


########################
# The camera parameters for the input and output.
########################
# The input camera parameters.
# The input camera is a pinhole.
camera_model_in = Pinhole(
    fx = 100,
    fy = 100,
    cx = 100,
    cy = 100,
    shape_struct= ShapeStruct(200, 200),
    # xi = -0.2,
    # alpha = 0.6,
    # fov_degree = 195
)

# The output camera parameters.
camera_model_out = DoubleSphere(
    fx = 80,
    fy = 80,
    cx = 100,
    cy = 100,
    shape_struct= ShapeStruct(200, 200),
    xi = -0.2,
    alpha = 0.6,
    fov_degree = 195
)

# Set device.
camera_model_in.device = 'cuda'
camera_model_out.device = 'cuda'

# The transformation between the cameras, if there is one.
R_out_in_in = torch.tensor(Rotation.from_euler('xyz', [0.2, 0, 0.5]).as_matrix()).to(dtype=torch.float32)
R_in_in_out = torch.tensor(Rotation.from_matrix(R_out_in_in.cpu().numpy()).inv().as_matrix()).to(dtype=torch.float32, device = 'cuda')


########################
# Create the pre- and post-processing functions.
########################
def create_preprocessing_flow(camera_model_in, R_in_in_out):
    def preprocessing(Fin):
        ########################
        # 1. Create the pixel coord list for the input image.
        ########################
        # Create the pixel coord grid for the input image.
        # This is of shape (2, N = H * W). Where the first channel is the x-coord (proxy column), and the second channel is the y-coord (proxy row) in pixels.
        # TODO(yoraish): is this shift correct?
        N = camera_model_in.ss.H * camera_model_in.ss.W
        Gin = camera_model_in.pixel_coordinates(shift = 0, normalized = False, flatten= True) # Shape is (2, N = H * W).

        ########################
        # 2. Create the pixel coord list given the applied flow.
        ########################
        Gin_f = Gin + Fin.view(2, N) # Shape is (2, N = H * W).

        ########################
        # 3. Transform the warped pixel coord list to the output image frame. This consists of rotating the rays that extend out of the flow-endpoints according to the specified rotation matrix.
        ########################
        # Get rays that go out of the flow endpoints. Disregard the valid mask since we also care here about endpoints that produce rays outside of the image frame.
        Gin_f_rays, Gin_f_rays_valid_mask = camera_model_in.pixel_2_ray(Gin_f)

        # Rotate the rays to be in the out image frame. We still mark this as Gin since the order of rays here corresponds to the pixels in the input camera model.
        RGin_f_rays = R_in_in_out @ Gin_f_rays # Shape is (3, N = H * W).

        ########################
        # 4. Sample the endpoints from the first image, that were transformed to the second image. Each pixel in the sampled image is an endpoint of the flow in normalized homogeneous coordinates.
        ########################
        # View the flow endpoint rotated rays in the shape of the input image. So each pixel holds the xyz position of its transformed flow endpoint.
        RGin_f_rays = RGin_f_rays.view(1, 3, camera_model_in.ss.H, camera_model_in.ss.W) # Shape is (1, 3, H, W).
    
        return RGin_f_rays
        
    return preprocessing


def create_postprocessing_flow(camera_model_out):
    def postprocessing(Gout_f_rays, valid_mask):
        ########################
        # 5. Project the warped pixel coord list to the output image as pixels.
        ########################
        # Convert the rays to pixel coordinates. These are the end pixel coordinates of each pixel. So pixel at (x=10, y=10) takes the value of [x,y] + Fout(x=10, y=10).
        # So Fout = Gout_f - Gout. With Gout being the pixel coordinates of the output image.
        Gout_f, Gout_f_valid_mask = camera_model_out.point_3d_2_pixel(Gout_f_rays.view( 3, camera_model_out.ss.W * camera_model_out.ss.H )) # Shape is (2, H * W).

        # Create a grid of pixel point coords in the output image.
        Gout = camera_model_out.pixel_coordinates(shift = 0, normalized = False, flatten= True) # Shape is (2, N = H * W).

        # Compute the flow.
        Fout = Gout_f - Gout # Shape is (2, N = H * W).

        # Apply the mask.
        Fout = Fout # * Gout_f_valid_mask # Shape is (2, N = H * W).
        Fout = Fout.view(1, 2, camera_model_out.ss.H, camera_model_out.ss.W) # Shape is (1, 2, H, W).

        return Fout, Gout_f_valid_mask
        
    return postprocessing

# Create the preprocessing function.
preprocessing_flow = create_preprocessing_flow(camera_model_in, R_in_in_out)

# Create the postprocessing function.
postprocessing_flow = create_postprocessing_flow(camera_model_out)


########################
# Create the sampler.
########################
# Create a sampler. Use fish as base in this example, but this should be a generic sampler later.
print(Fore.YELLOW + "Creating sampler..." + Style.RESET_ALL)
print(Fore.GREEN + "    Input camera model: {}".format(camera_model_in) + Style.RESET_ALL)
print(Fore.GREEN + "    Output camera model: {}".format(camera_model_out) + Style.RESET_ALL)
sampler = GenericCameraModelSampler(camera_model_in, camera_model_out, R_out_in_in, preprocessing_flow, postprocessing_flow)
sampler.device = 'cuda'

# Try sampler.
print(Fore.YELLOW + "Sampling..." + Style.RESET_ALL)
print(Fore.GREEN + "    Input shape: {}".format(Fin.shape) + Style.RESET_ALL)

# Show sampled. The input to the sampler is of shape (B, C, H, W).
sampled, valid_mask = sampler(Fin)
print(Fore.GREEN + "    Output shape: {}".format(sampled.shape) + Style.RESET_ALL)

if is_vis:
    plt.figure(3)
    plt.imshow(flow_tensor_to_image(sampled.squeeze(0)).transpose(1,2,0))
    # plt.imshow(sampled.squeeze(0).cpu().numpy().transpose(1,2,0))
    plt.title("Optical Flow SAMPLED of shape {}".format(sampled.shape))
    

########################
# Start the optical flow flow. Hehe.
########################
Fout, Fout_mask = sampler(Fin, interpolation = 'linear', invalid_pixel_value = 0) # Shape is (1, 2, H, W). TODO(yoraish): is this interpolation okay?


# Plot the flow.
if is_vis:
    plt.figure(5)
    plt.imshow(flow_tensor_to_image(Fout.squeeze(0)).transpose(1,2,0))
    Fout_show = torch.cat([Fout, torch.ones_like(Fout[:,0:1,:,:])], dim=1).int()
    # plt.imshow(Fout_show.squeeze(0).cpu().numpy().transpose(1,2,0).astype(np.uint8))
    plt.title("Optical Flow Output of shape {}".format(Fout.shape))\
    # Draw arrows on the flow image.
    for i in range(0, Fin.shape[2], 20):
        for j in range(0, Fin.shape[3], 20):
            plt.arrow(j, i, Fout.cpu().numpy()[0,0,i,j], Fout.cpu().numpy()[0,1,i,j], color='r', head_width=5, head_length=3)
    plt.show()


