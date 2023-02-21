# A test file for LinearSphere camera model from six images.

from colorama import Fore, Style
import torch
import numpy as np
import matplotlib.pyplot as plt

from ...image_sampler.camera_model_sampler import CameraModelRotation
from ...image_sampler.six_images import SixPlanarTorch
from ..camera_models import DoubleSphere, ShapeStruct, Pinhole, LinearSphere

# Load the six images.
img_gps = {
    'front': "image_resampling/mvs_utils/examples/data/000000_lcam_front.png",
    'left': "image_resampling/mvs_utils/examples/data/000000_lcam_left.png",
    'back': "image_resampling/mvs_utils/examples/data/000000_lcam_back.png",
    'right': "image_resampling/mvs_utils/examples/data/000000_lcam_right.png",
    'top': "image_resampling/mvs_utils/examples/data/000000_lcam_top.png",
    'bottom': "image_resampling/mvs_utils/examples/data/000000_lcam_bottom.png"}

img_dict = {k: plt.imread(v) for k, v in img_gps.items()}

# Camera model for the fisheye.
fish_model = DoubleSphere(
    xi = -0.2, 
    alpha = 0.6, 
    fx = 235, 
    fy = 235, 
    cx = 500, 
    cy = 500, 
    fov_degree = 195, 
    shape_struct = ShapeStruct(1000, 1000)
)

pinhole_model = Pinhole(
    fx = 235,
    fy = 235,
    cx = 250,
    cy = 250,
    shape_struct = ShapeStruct(500, 500)
    )

six_image_sampler = SixPlanarTorch(195, fish_model, R_raw_fisheye= torch.eye(3))
fish_img, _ = six_image_sampler(img_dict)

# Camera model for the linear.
linear_model = LinearSphere(fov_degree=195, shape_struct=ShapeStruct(1000, 1000)) 

# Camera model sampler.
linear_model_sampler = SixPlanarTorch( 195, linear_model, R_raw_fisheye= torch.eye(3))
linear_img, valid_mask = linear_model_sampler(img_dict)
# Zero out the invalid pixels.
linear_img[~valid_mask] = 0

# Plot the images.
fig, ax = plt.subplots(1, 2)
# Set a heading for the figure.
fig.suptitle("LinearSphere and DoubleSphere camera models", fontsize=16)
ax[0].imshow(linear_img)
ax[0].set_title("LinearSphere")
ax[1].imshow(fish_img)
ax[1].set_title("DoubleSphere")

# Plot the images.
fig, ax = plt.subplots(3, 2)
fig.suptitle("LinearSphere and DoubleSphere comparison", fontsize=16)
ax[0, 0].imshow(linear_img)
ax[0, 0].set_title("LinearSphere")
ax[0, 1].imshow(fish_img)
ax[0, 1].set_title("DoubleSphere")

# Overlay the images.
ax[1, 0].imshow(linear_img)
ax[1, 0].imshow(fish_img, alpha=0.5)
ax[1, 0].set_title("LinearSphere")
ax[1, 1].imshow(fish_img)
ax[1, 1].imshow(linear_img, alpha=0.5)
ax[1, 1].set_title("DoubleSphere")

# Show the error between the images.
ax[2, 0].imshow(linear_img - fish_img)
ax[2, 0].set_title("LinearSphere - DoubleSphere")
ax[2, 1].imshow(fish_img - linear_img)
ax[2, 1].set_title("DoubleSphere - LinearSphere")


#####################
# Test rays and projections.
#####################

# Create a sampler.
linear_model_sampler = CameraModelRotation(linear_model, linear_model, R_raw_fisheye= torch.eye(3))
linear_model_sampler.device = 'cuda'

# Sample the image.
resampled, _ = linear_model_sampler(linear_img)

# Plot the images.
fig, ax = plt.subplots(2, 2)
fig.suptitle("LinearSphere camera model resampling test", fontsize=16)
ax[0, 0].imshow(linear_img)
ax[0, 0].set_title("LinearSphere")
ax[0, 1].imshow(resampled)
ax[0, 1].set_title("Resampled")

# Error of the images.
ax[1, 0].imshow(linear_img - resampled)
ax[1, 0].set_title("LinearSphere - Resampled")
ax[1, 1].imshow(resampled - linear_img)
ax[1, 1].set_title("Resampled - LinearSphere. Error is {}".format(np.mean(np.abs(linear_img - resampled))))

plt.show()


# Test batch.
batch_size = 10
coord_batch = torch.rand((batch_size, 2, 1000)).to('cuda')
print(Fore.GREEN + "Testing batch of size {}".format(batch_size) + Style.RESET_ALL)

# Get rays for these coordinates.
rays, _ = linear_model.pixel_2_ray(coord_batch)
print(Fore.GREEN + "Rays shape: {}".format(rays.shape) + Style.RESET_ALL)

# Project the rays.
pix, _ = linear_model.point_3d_2_pixel(rays)
print(Fore.GREEN + "Proj pix shape: {}".format(pix.shape) + Style.RESET_ALL)

# Get rays.



######################333



# Create a sampler.
pinhole_model_sampler = CameraModelRotation(linear_model, pinhole_model, R_raw_fisheye= torch.eye(3))
pinhole_model_sampler.device = 'cuda'
# Sample the image.
resampled, _ = pinhole_model_sampler(linear_img)

pinhole_model_sampler2 = CameraModelRotation(fish_model, pinhole_model, R_raw_fisheye= torch.eye(3))
pinhole_model_sampler2.device = 'cuda'
# Sample the image.
resampled2, _ = pinhole_model_sampler2(fish_img)

# Plot the images.
fig, ax = plt.subplots(3, 2)
ax[0, 0].imshow(linear_img)
ax[0, 0].set_title("LinearSphere")
ax[0, 1].imshow(resampled)
ax[0, 1].set_title("Resampled")

ax[1, 0].imshow(fish_img)  
ax[1, 0].set_title("DoubleSphere")
ax[1, 1].imshow(resampled2)
ax[1, 1].set_title("Resampled")

# Error of the images.
ax[2, 1].imshow(resampled - resampled2)
ax[2, 1].set_title("Resampled - Resampled2, Error is {}".format(np.mean(np.abs(resampled - resampled2))))


plt.show()