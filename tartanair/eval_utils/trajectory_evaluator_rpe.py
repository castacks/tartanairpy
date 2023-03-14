'''
Adapted by: Yorai Shaoul 
Date: December 2022

Modified by Wenshan Wang
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


Description: Compute the relative pose error (RPE) for a monocular VO/SLAM system. RPE is computed as the mean of the translational error over all pairs of consecutive frames. The translational error is computed as the Euclidean distance between the ground truth and the estimated relative pose. Both the ground truth and the estimated relative poses are specified in the same reference frame.

'''

# General imports.
import random
from colorama import Fore, Style
import argparse
import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local imports.
from .trajectory_evaluator_base import TrajectoryEvaluatorBase

class TrajectoryEvaluatorRPE(TrajectoryEvaluatorBase):

    def __init__(self, gt_traj=None, est_traj=None, plot=False, plot_gfp=None, do_scale=True, do_align=True):

        # Instantiate super class.
        super().__init__(gt_traj, est_traj, plot, plot_gfp, do_scale, do_align)

    def compute_rpe(self, gt_traj = None, est_traj = None, delta = 1, do_scale = False, do_align = False):
        # Use the member variables if an input is not provided.
        if gt_traj is None:
            gt_traj = self.gt_traj
        if est_traj is None:
            est_traj = self.est_traj
        if do_scale is None:
            do_scale = self.do_scale

        # Align the estimated trajectory to the ground truth trajectory, get the alignment transformation, and the scale if requested.
        if do_align:
            est_traj_aligned, s = self.align_and_scale_traj_to_gt(gt_traj, est_traj, do_scale)
            print(Fore.GREEN + "RPE Scale: " + str(s) + Style.RESET_ALL)

        else:
            est_traj_aligned = est_traj
            s = 1.0
        
        # Store in a member variable.
        self.est_traj_aligned = est_traj_aligned

        # Turn the trajectories into a list of 4x4 matrices.
        gt_traj_SEs = self.pos_quats_to_SE_matrices(gt_traj[:, 1:])
        est_traj_SEs = self.pos_quats_to_SE_matrices(est_traj_aligned[:, 1:])

        # Compute the relative pose error.
        pairs = []
        max_pairs = 100000
        for i in range(len(est_traj)):
            j = i + delta
            if j < len(est_traj): 
                pairs.append((i,j))
        if(max_pairs!=0 and len(pairs) > max_pairs):
            pairs = random.sample(pairs, max_pairs)
            
        result = []
        for i,j in pairs:
            error44 = self.ominus(  self.ominus( est_traj_SEs[j], est_traj_SEs[i] ),
                            self.ominus( gt_traj_SEs[j], gt_traj_SEs[i] ) )
            
            trans = self.compute_distance(error44)
            rot = self.compute_angle(error44)
            
            result.append([i,j,trans,rot])
            
        if len(result)<2:
            raise Exception("Couldn't find pairs between groundtruth and estimated trajectory!")
            
        trans_error = np.array(result)[:,2]
        rot_error = np.array(result)[:,3]

        trans_error_mean = np.mean(trans_error)
        rot_error_mean = np.mean(rot_error)

        if self.plot:
            title_text = "RPE_" + "{:.5f}".format(trans_error_mean) + "_t_" + "{:.5f}".format(rot_error_mean) + "_R"
            self.visualize_2d_projection(gt_traj, est_traj_aligned, title_text, self.plot_gfp)
        return (trans_error_mean, rot_error_mean), gt_traj, est_traj_aligned
