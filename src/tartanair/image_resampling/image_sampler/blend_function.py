
import cv2
import numpy as np

import torch
import kornia

class BlendBy2ndOrderGradient(object):
    def __init__(self, threshold_scaling_factor) -> None:
        super().__init__()
        
        self.threshold_scaling_factor = threshold_scaling_factor
        self.adaptive_threshold_max = 10
    
    def __call__(self, img):
        return self.blend_func(img)
    
class BlendBy2ndOrderGradTorch(BlendBy2ndOrderGradient):
    def __init__(self, threshold_scaling_factor) -> None:
        super().__init__(threshold_scaling_factor)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (B, C, H, W), g is (B, C, 2, H, W).
        g = kornia.filters.spatial_gradient(img.to(torch.float32), mode='sobel', order=2)
        
        # Sum (norm) the result along the channel dimension.
        s = torch.linalg.norm( g, dim=-3, keepdim=False )
        s = torch.linalg.norm( s, dim=-3, keepdim=True )
        
        # adaptive threshold wrt the depth value
        adaptive_thresh = torch.clamp(img.float() * self.threshold_scaling_factor, self.threshold_scaling_factor, self.adaptive_threshold_max)

        # Find the over-threshold ones.
        m = s > adaptive_thresh
        m = m.to(torch.float32)
        
        # Add some dilation.
        # m = kornia.morphology.dilation(m, torch.ones((3, 3), device=s.device), border_type='geodesic', border_value=0.0)
        # Add some erosion.
        m = kornia.morphology.erosion(m, torch.ones((3, 3), device=s.device), border_type='geodesic', border_value=0.0)
        
        return m
    
class BlendBy2ndOrderGradOcv(object):
    def __init__(self, threshold_scaling_factor) -> None:
        super().__init__(threshold_scaling_factor)
        
    def blend_func(self, img):
        # Compute the Laplacian.
        # img is (H, W, ...), g is (H, W, ...).
        # g = cv2.Laplacian(img, cv2.CV_32F, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(img, cv2.CV_32FC1, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, cv2.CV_32FC1, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT)
        
        # Sum (norm) the results.
        s = np.sqrt( gx**2 + gy**2 )

        # adaptive threshold wrt the depth value
        adaptive_thresh = np.clip(img.astype(np.float32) * self.threshold_scaling_factor, self.threshold_scaling_factor, self.adaptive_threshold_max)

        # Find the over-threshold ones.
        m = s > adaptive_thresh
        m = m.astype(np.float32)
        
        # Add some dilation.
        # m = cv2.dilate( m, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0 )
        # Add some erosion.
        m = cv2.erode(m, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0)
        
        return m

if __name__=="__main__":
    thresh_f_list = [0.02, 0.05, 0.1, 0.5, 1, 2]
    visfile = "/home/amigo/tmp/test_root/test_sample_downtown2/Data_easy/P000/image_lcam_back/000000_lcam_back.png"
    imgfile = "/home/amigo/tmp/test_root/test_sample_downtown2/Data_easy/P000/depth_lcam_back/000000_lcam_back_depth.png"
    vis = cv2.imread(visfile)
    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
    img = np.squeeze(img.view("<f4"), axis=-1)
    img = torch.from_numpy(img).view(1,1,640,640)
    cv2.imshow('vis', vis)
    imglist = []
    for thresh_f in thresh_f_list:
        bb = BlendBy2ndOrderGradTorch(thresh_f)
        yy = bb.blend_func(img)
        yy = yy.squeeze().numpy()
        imglist.append(yy)
    disp = np.concatenate((np.concatenate(imglist[:3],axis=1),np.concatenate(imglist[3:],axis=1)),axis=0)
    cv2.imshow('img',disp)
    cv2.waitKey(0)
