'''
Author: Yorai Shaoul
Date: 2023-03-01

This file contains the evaluator class, which evaluated estimated trajectories against ground truth.
'''

# General imports.
import os
from scipy.spatial.transform import Rotation
from colorama import Fore, Style
import numpy as np


# Local imports.
from .tartanair_module import TartanAirModule
from .reader import TartanAirTrajectoryReader
from .eval_utils.trajectory_evaluator_ate import TrajectoryEvaluatorATE
from .eval_utils.trajectory_evaluator_rpe import TrajectoryEvaluatorRPE

class TartanAirEvaluator(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)
        self.tartanair_data_root = tartanair_data_root
        self.reader = TartanAirTrajectoryReader(tartanair_data_root)

    def evaluate_traj(self, 
                        est_traj,
                        gt_traj = None,
                        env = None, 
                        difficulty = None, 
                        trajectory_id = None, 
                        camera_name = None, 
                        enforce_length = True, 
                        plot = False, 
                        plot_out_path = None, 
                        do_scale = True, 
                        do_align = True):
        '''
        Evaluate estimated trajectory against ground truth.
        '''
        # Make sure that we were passed a trajectory specification or a ground truth trajectory.
        if (env is None) or (difficulty is None) or (trajectory_id is None) or (camera_name is None):
            if gt_traj is None:
                raise ValueError("Please pass a ground truth trajectory or a trajectory specification (env, difficulty, trajectory_id, camera_name) to the evaluation method.")


        # Make sure that the input is a numpy array with a correct shape.
        if isinstance(est_traj, list):
            print(Fore.CYAN + "Warning: Converting the estimated trajectory (list) to a numpy array." + Style.RESET_ALL)
            est_traj = np.array(est_traj)
        
        # Get the ground truth trajectory, if one was not passed.
        if gt_traj is None:
            gt_traj = self.reader.get_traj_np(env, difficulty, trajectory_id, camera_name)

        # Make sure that the estimated trajectory is the same length as the ground truth trajectory.
        if enforce_length:
            if est_traj.shape[0] != gt_traj.shape[0]:
                raise ValueError("The estimated trajectory has {} entries, while the ground truth trajectory has {} frames. If you'd like to evaluate this setup, set enforce_length to False".format(est_traj.shape[0], gt_traj.shape[0]))

        else:
            # Make sure that the estimated trajectory is at least as long as the ground truth trajectory.
            if est_traj.shape[0] < gt_traj.shape[0]:
                gt_traj = gt_traj[0:est_traj.shape[0], :]
        
        # Check the shape of the input trajectory.
        if est_traj.shape[1] != 7:
            raise ValueError("The estimated trajectory should have 7 columns (xyz xyzw), but it has {} columns.".format(est_traj.shape[1]))

        # Compute the ATE and the RPE.
        ate_evaluator = TrajectoryEvaluatorATE(gt_traj = gt_traj.copy(), 
                                            est_traj = est_traj.copy(), 
                                            plot= plot_out_path is not None,
                                            plot_gfp = plot_out_path, 
                                            do_scale=do_scale, 
                                            do_align=do_align)

        ate, gt_traj, est_traj_aligned   = ate_evaluator.compute_ate(do_scale=do_scale,
                                        do_align=do_align)
        print(f'---> ATE: {ate} m.')

        rpe_evaluator = TrajectoryEvaluatorRPE(gt_traj = gt_traj.copy(),
                                                est_traj = est_traj.copy(),    
                                                    plot= False,
                                                    plot_gfp = plot_out_path,
                                                    do_scale=do_scale,
                                                    do_align=do_align)

        rpe, gt_traj, est_traj_aligned = rpe_evaluator.compute_rpe(do_scale=do_scale,
                                        do_align=do_align)
        print(f'---> RPE: {rpe}.')

        output = {
            'ate': ate,
            'rpe': rpe,
            'gt_traj': gt_traj[:, 1:], # Removing the index column.
            'est_traj': est_traj_aligned[:, 1:] # Removing the index column.
        }
        return output 