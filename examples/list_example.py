'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
 
ta.init(tartanair_data_root)

# List available trajectories.
available_envs = ta.list_envs() # Returns a dictionary with the available environments. Of form {'local': ['env1', 'env2', ...], 'remote': ['env1', 'env2', ...]}
print(available_envs)