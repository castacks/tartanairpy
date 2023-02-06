'''
Reading images from TartanAir dataset files.
'''

import cv2
import numpy as np
import cv2

class TartanAirImageReader():
    '''
    Read images from files, 
    Return numpy array
    or return visualizable data 
    '''
    def __init__(self, ):
        pass

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

    def read_flow(self, flowpath, scale = 1):
        if flowpath.endswith('npy'):
            flownp = np.load(flowpath)
        else:
            flow16 = cv2.imread(flowpath, cv2.IMREAD_UNCHANGED)
            if flow16 is None:
                return None
            flownp, _ = self.flow16to32(flow16)

        return flownp
