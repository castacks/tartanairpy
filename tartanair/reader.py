'''
Reading images from TartanAir dataset files.
'''

import os
import cv2
import numpy as np
import cv2

# Local imports.
from .tartanair_module import TartanAirModule

class TartanAirImageReader():
    '''
    Read images from files, 
    Return numpy array
    or return visualizable data 
    '''
    def __init__(self, ):
        # Some parameters for converting depth to distance.
        self.depth_shape = (0,0)
        self.conv_matrix = None

    def read_bgr(self, imgpath, scale = 1):
        img = cv2.imread(imgpath)
        if img is None or img.size==0:
            return None
        if scale != 1:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img

    def read_rgb(self, imgpath, scale = 1):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None or img.size==0:
            return None
        if scale != 1:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img

    def depth_rgba_float32(self, depth_rgba):
        depth = depth_rgba.view("<f4")
        return np.squeeze(depth, axis=-1)

    def flow16to32(self, flow16):
        '''
        flow_32b (float32) [-512.0, 511.984375]
        flow_16b (uint16) [0 - 65535]
        flow_32b = (flow16 -32768) / 64
        '''
        flow32 = flow16[:,:,:2].astype(np.float32)
        flow32 = (flow32 - 32768) / 64.0

        mask8 = flow16[:,:,2].astype(np.uint8)
        return flow32, mask8
    
    def read_depth(self, depthpath):
        if depthpath.endswith('npy'):
            depth = np.load(depthpath)
        else:
            depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
            if depth_rgba is None:
                return None
            depth = self.depth_rgba_float32(depth_rgba)
        return depth

    def read_disparity(self, depthpath, p = 80.0):
        depth = self.read_depth(depthpath)
        return p/(depth+1e-6)

    def read_seg(self, segpath):
        if segpath.endswith('npy'):
            seg = np.load(segpath)
        else:
            seg = cv2.imread(segpath, cv2.IMREAD_UNCHANGED)
        return seg

    # def read_flow(self, flowpath, scale = 1):
    #     if flowpath.endswith('npy'):
    #         flownp = np.load(flowpath)
    #     else:
    #         flow16 = cv2.imread(flowpath, cv2.IMREAD_UNCHANGED)
    #         if flow16 is None:
    #             return None
    #         flownp, _ = self.flow16to32(flow16)

    #     return flownp

    def read_flow(self, flowpath, scale=1, direction="fwd"):
        if flowpath.endswith(".npz"):
            # only npz supports backward direction
            assert direction in ("fwd", "bwd"), \
                f"Invalid direction '{direction}', must be 'fwd' or 'bwd' for npz"

            dir_key = "flow_fwd" if direction == "fwd" else "flow_bwd"
            with np.load(flowpath) as data:
                if dir_key not in data:
                    raise KeyError(
                        f"'{dir_key}' not found in {flowpath}. "
                        f"Available keys: {list(data.keys())}"
                    )
                flownp = data[dir_key]
                flownp = flownp[0].transpose(1, 2, 0)  # CHW to HWC

        elif flowpath.endswith(".npy"):
            # npy is always forward
            assert direction == "fwd", \
                f"direction='{direction}' is not supported for npy (only 'fwd')"
            flownp = np.load(flowpath)

        else: # png
            # image encoding, also only forward
            assert direction == "fwd", \
                f"direction='{direction}' is not supported for image inputs (only 'fwd')"
            flow16 = cv2.imread(flowpath, cv2.IMREAD_UNCHANGED)
            if flow16 is None:
                return None
            flownp, _ = self.flow16to32(flow16)

        if scale != 1:
            flownp = flownp * scale
        return flownp

    def read_dist(self, fn ): # read a depth image and convert it to distance
        depth = self.read_depth(fn)
        return self.depth_to_dist(depth)

    def depth_to_dist(self, depth):
        '''
        assume: fov = 90 on both x and y axes, and optical center is at image center.
        '''
        if self.depth_shape is None or \
            depth.shape != self.depth_shape or \
            self.conv_matrix is None: # only calculate once if the depth shape has not changed
            hh, ww = depth.shape
            f = ww/2
            wIdx = np.linspace(0, ww - 1, ww, dtype=np.float32) + 0.5 - ww/2 # put the optical center at the middle of the image
            hIdx = np.linspace(0, hh - 1, hh, dtype=np.float32) + 0.5 - hh/2 # put the optical center at the middle of the image
            u, v = np.meshgrid(wIdx, hIdx)
            dd = np.sqrt(u * u + v * v + f * f)/f
            self.conv_matrix = dd
        self.depth_shape = depth.shape
        disp = self.conv_matrix * depth
        return disp


    ###############################
    # Data reading, writing, and processing.
    ###############################
    def depth_to_dist(self, depth):
        '''
        assume: fov = 90 on both x and y axes, and optical center is at image center.
        '''
        # import ipdb;ipdb.set_trace()
        if self.depth_shape is None or \
            depth.shape != self.depth_shape or \
            self.conv_matrix is None: # only calculate once if the depth shape has not changed
            hh, ww = depth.shape
            f = ww/2
            wIdx = np.linspace(0, ww - 1, ww, dtype=np.float32) + 0.5 - ww/2 # put the optical center at the middle of the image
            hIdx = np.linspace(0, hh - 1, hh, dtype=np.float32) + 0.5 - hh/2 # put the optical center at the middle of the image
            u, v = np.meshgrid(wIdx, hIdx)
            dd = np.sqrt(u * u + v * v + f * f)/f
            self.conv_matrix = dd
        self.depth_shape = depth.shape
        disp = self.conv_matrix * depth
        return disp

    # def ocv_read(self, fn ):
    #     image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    #     assert image is not None, \
    #         f'{fn} read error. '
    #     return image

    # def read_rgb(self, fn ):
    #     return self.ocv_read(fn)

    # def read_dist(self, fn ): # read a depth image and convert it to distance
    #     depth = self.read_dep(fn)
    #     return self.depth_to_dist(depth)

    # def read_dep(self, fn ):
    #     image = self.ocv_read(fn)
    #     depth = np.squeeze( image.view('<f4'), axis=-1 )
    #     return depth

    # def vis_dep(self, fn):
    #     depth = self.read_dep(fn)
    #     depthvis = np.clip(1/depth * 400, 0, 255).astype(np.uint8)
    #     return cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)

    # def read_seg(self, fn ):
    #     image = self.ocv_read(fn)
    #     return image.astype(np.uint8)

    def ocv_write(self, fn, image ):
        cv2.imwrite(fn, image)
    
    def write_as_is(self, fn, image ):
        self.ocv_write( fn, image )
    
    def write_float_compressed(self, fn, image):
        assert(image.ndim == 2), 'image.ndim = {}'.format(image.ndim)
    
        # Check if the input array is contiguous.
        if ( not image.flags['C_CONTIGUOUS'] ):
            image = np.ascontiguousarray(image)

        dummy = np.expand_dims( image, 2 )
        self.ocv_write( fn, dummy )
        
    def write_float_depth(self, fn, image):
        if len(image.shape) == 2:
            image = image[...,np.newaxis]
        depth_rgba = image.view("<u1")
        self.ocv_write(fn, depth_rgba)

class TartanAirTrajectoryReader(TartanAirModule):
    '''
    Load a trajectory from TartanAir dataset.
    '''
    def __init__(self, tartanair_data_root):
        super(TartanAirTrajectoryReader, self).__init__(tartanair_data_root)
        self.tartanair_data_root = tartanair_data_root

    def get_traj_np(self, env, difficulty, trajectory_id, camera_name):
        '''
        Get a trajectory.
        '''
        # Construct the path to the trajectory file.
        difficulty = "Data_{}".format(difficulty)
        traj_path = os.path.join(self.tartanair_data_root, env, difficulty, trajectory_id, 'pose_' + camera_name + '.txt')
        # Read the trajectory file.
        traj_np = np.loadtxt(traj_path, delimiter=' ')

        return traj_np