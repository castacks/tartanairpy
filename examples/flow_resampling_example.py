'''
Author: Yuchen Zhang
Date: 2024-11-19

Example script for sample new correspondence from the TartanAir dataset.
'''
import tartanair as ta

# Initialize TartanAir.
tartanair_data_root = 'your/path/to_tav2'
ta.init(tartanair_data_root)

# Customize the dataset.
ta.customize_flow(env = 'ArchVizTinyHouseDay',
             difficulty = 'easy',
             trajectory_id = ['P000'],
             cam_sides=["left", "right"],
             num_workers = 4,
             frame_sep = 1,
             device = "cuda") # or cpu