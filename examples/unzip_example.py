'''
Author: Wenshan Wang    
Date: 2025-11-01

If you haven't unzipped the files while downloading, you can use this script to unzip them.
This script also allows you to unzip the files into a different directory.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/my/path/to/root/folder/for/tartanair-v2'
unzip_target_dir = '/my/path/to/root/folder/for/tartanair-v2/unzip'

ta.init(tartanair_data_root)

ta.unzip(output_dir = unzip_target_dir, num_workers = 4)
