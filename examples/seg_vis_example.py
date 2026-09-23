import sys
# Local imports.
sys.path.append('..')
import tartanair as ta

if __name__ == '__main__':
    # Create a TartanAir object.
    tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'

    # Initialize the toolbox.
    ta.init(tartanair_data_root)

    # Create an example trajectory. This is a noisy version of the ground truth trajectory.
    env = 'CoalMine'
    difficulty = 'easy'
    trajectory_id = 'P004'
    camera_name = 'lcam_front'

    # List available trajectories.
    ta.visualize(env,
                difficulty=difficulty,
                trajectory_id = trajectory_id,
                modality = ['image', 'depth', 'seg'],
                camera_name = ['lcam_front'],
                show_seg_palette = True)
