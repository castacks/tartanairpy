'''
Author: Yorai Shaoul
Date: 2023-03-14. Happy Pi Day!

Example script for evaluating using the TartanAir dataset toolbox.
'''

# General imports.
import sys
import numpy as np

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
 
# Initialize the toolbox.
ta.init(tartanair_data_root)

# Create an example trajectory. This is a noisy version of the ground truth trajectory.
env = 'AbandonedCableExposure'
difficulty = 'easy'
trajectory_id = 'P002'
camera_name = 'lcam_front'
gt_traj = ta.get_traj_np(env, difficulty, trajectory_id, camera_name)
est_traj = np.zeros_like(gt_traj)
est_traj[:, :3] = gt_traj[:, :3] + np.random.normal(0, 0.2, gt_traj[:, :3].shape)  
est_traj[:, 3:] = gt_traj[:, 3:] + np.random.normal(0, 0.01, gt_traj[:, 3:].shape)

# Get the evaluation results.
plot_out_path = "evaluator_example.png"
results = ta.evaluate_traj(est_traj, env, difficulty, trajectory_id, camera_name, enforce_length = True, plot = True, plot_out_path = plot_out_path, do_scale = True, do_align = True)

# Optionally pass the ground truth trajectory directly to the evaluation function.
results = ta.evaluate_traj(est_traj, gt_traj = gt_traj, enforce_length = True, plot = True, plot_out_path = plot_out_path, do_scale = True, do_align = True)