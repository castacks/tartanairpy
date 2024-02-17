'''
Adapted by: Yorai Shaoul 
Date: December 2022

Modified by Wenshan Wang
Modified by Raul Mur-Artal
Automatically compute the optimal scale factor for monocular VO/SLAM.

Software License Agreement (BSD License)

Copyright (c) 2013, Juergen Sturm, TUM
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of TUM nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Compute the absolute trajectory error (ATE) for a monocular VO/SLAM system. ATE is computed as the mean of the translational error over all frames. The translational error is computed as the Euclidean distance between the ground truth and the estimated trajectory. Both the ground truth trajectory and the estimatedd trajectory are specified in the same reference frame. 
'''

# General imports.
from colorama import Fore, Style
import argparse
import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local imports.
from .trajectory_evaluator_base import TrajectoryEvaluatorBase

class TrajectoryEvaluatorATE(TrajectoryEvaluatorBase):        

    def __init__(self, gt_traj=None, est_traj=None, plot=False, plot_gfp=None, do_scale=True, do_align=True):
        """Initialize the trajectory evaluator."""
        super().__init__(gt_traj, est_traj, plot, plot_gfp, do_scale, do_align)

        # Placeholder for estimated trajectory aligned to the ground truth and scaled.
        self.est_traj_aligned = None


    def compute_ate(self, gt_traj = None, est_traj = None, do_scale=False, do_align=False):
        """Compute the absolute trajectory error (ATE).
            self.gt_traj (np.ndarray , (N,8): stamp, xyz xyzw): Ground truth trajectory.
            self.est_traj (np.ndarray, (N,8): stamp, xyz xyzw): Estimated trajectory.

            returns:
                ate (float): Absolute trajectory error.

        """
        # Use the member variables if an input is not provided.
        if gt_traj is None:
            gt_traj = self.gt_traj
        if est_traj is None:
            est_traj = self.est_traj
        if do_scale is None:
            do_scale = self.do_scale
    

        # Align the estimated trajectory to the ground truth trajectory, get the alignment transformation, and the scale if requested.
        if do_align: # TODO(yoraish): Should this be if (do_align or do_scale)?
            est_traj_aligned, s = self.align_and_scale_traj_to_gt(gt_traj, est_traj, do_scale)
            print(Fore.GREEN + "ATE Scale factor: " + str(s) + Style.RESET_ALL)
        else:
            est_traj_aligned = est_traj
        
        self.est_traj_aligned = est_traj_aligned
        ate = np.sqrt( (np.sum( np.linalg.norm(self.est_traj_aligned[:, 1:4] - gt_traj[:, 1:4], axis = 1)**2 ) / len(est_traj_aligned)  ) )
        self.ate = ate

        if self.plot:
            title_text = "ATE_{:.3f} m".format(ate)
            self.visualize(gt_traj, est_traj_aligned, title_text, self.plot_gfp)
            self.visualize_2d_projection(gt_traj, est_traj_aligned, title_text, self.plot_gfp)

        # Return the ATE, the ground truth trajectory with its timestamps, and the aligned estimated trajectory.
        return ate, gt_traj, est_traj_aligned
