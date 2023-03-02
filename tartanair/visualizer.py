'''
Author: Yorai Shaoul
Date: 2023-02-28

This file contains the visualizer class, which visualizes data from local tartanair data.
'''
# General imports.
import os
from colorama import Fore, Style
import cv2
import numpy as np

# Local imports.
from .tartanair_module import TartanAirModule
from .iterator import TartanAirIterator

class TartanAirVisualizer(TartanAirModule):
    def __init__(self, tartanair_data_root, azure_token = None):
        super().__init__(tartanair_data_root)

        # Modality mapped to a reader.
        self.modality_to_vis_func = {'image': self.visimage, 'depth': self.visdepth, 'seg': self.visseg}

    def visualize(self, env, difficulty = ['easy'], trajectory_id = ['P000'], modality = [], camera_name = []):
        """
        Visualizes a trajectory from the TartanAir dataset. A trajectory includes a set of images and a corresponding trajectory text file describing the motion.

        Args:
            env (str or list): The environment to visualize the trajectory from. 
            difficulty (str or list): The difficulty of the trajectory. Valid difficulties are: easy, medium, hard.
            trajectory_id (int or list): The id of the trajectory to visualize.
            modality (str or list): The modality to visualize. Valid modalities are: rgb, depth, seg. Default is rgb.
        """
        
        # Create an iterator for the data.
        iterator_obj = TartanAirIterator(self.tartanair_data_root)
        iterator = iterator_obj.get_iterator(env, difficulty, trajectory_id, modality, camera_name)

        # The images of this sample. Of shape (string with camname and modality, image).

        for ix, sample in enumerate(iterator):
            sample_images = []
            sample_image_names = []
            for cam_name in sample.keys():
                for modality in sample[cam_name]:
                    if modality == 'motion':
                        continue
                    vis_img = self.modality_to_vis_func[modality](sample[cam_name][modality])
                    vis_img_name = str(ix) + " " + cam_name + " " + modality
                    sample_images.append( vis_img )
                    sample_image_names.append(vis_img_name)

            #############################
            # Visualize the images.
            #############################
            # Visualize the images side by side. Max images per row is 5.
            max_images_per_row = 5
            num_images = len(sample_images)
            num_rows = int(num_images / max_images_per_row) + 1
            num_cols = min(num_images, max_images_per_row)

            # Create a window to display the images.
            window_name = 'TartanAir Visualizer'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Create a black image to display the images on.
            img_height = sample_images[0].shape[0]
            img_width = sample_images[0].shape[1]
            black_image = 255 * np.ones((num_rows * img_height, num_cols * img_width, 3), np.uint8)

            # Display the images.
            for i in range(num_images):
                # Get the image.
                image = sample_images[i]
                # Convert to bgr.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Overlay the name of the image onto the image itself.
                name = sample_image_names[i] if sample_image_names is not None else str(i)
                # Bar on top.
                cv2.rectangle(image, (0, 0), (image.shape[1], 30 ), (0, 0, 0), -1)
                # Text on top.
                cv2.putText(image, name, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Small frame around image.
                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), 3)

                # Get the row and column of the image.
                row = int(i / max_images_per_row)
                col = i % max_images_per_row

                # Get the image position.
                x = col * img_width
                y = row * img_height

                # Copy the image to the black image.
                black_image[y:y+img_height, x:x+img_width] = image

            # Display the image.
            cv2.imshow(window_name, black_image)

            # Wait for the user to press a key.
            cv2.waitKey(5)


    def visimage(self, image):
        """
        Visualizes an image.

        Args:
            image (np.array): The image to visualize. Shape h,w,c. Type uint8.

        Returns:
            The image to visualize. Shape h,w,c. Type uint8.
        """
        return image


    def visdepth(self, depth):
        depthvis = np.clip(400/depth ,0 ,255)
        depthvis = depthvis.astype(np.uint8)
        depthvis = cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)

        return depthvis


    def visseg(self, seg):
        return seg