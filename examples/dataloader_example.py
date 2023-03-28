'''
An example for using the dataloader. Note that under the hood the dataloader employs a powerful parallelized systems that simulatanuously loads data and serves mini-batches. So training and loading can happen concurrently.

Author: Yorai Shaoul
Date: 2023-03-28
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/media/yoraish/overflow/data/tartanair-v2'
ta.init(tartanair_data_root)


# Specify the environments, difficulties, and trajectory ids to load.
envs = ['HQWesternSaloonExposure']
difficulties = ['easy', 'hard']
trajectory_ids = []#['P000', 'P001']

# Specify the modalities to load.
modalities = ['image', 'depth', 'pose']
camnames = ['lcam_front', 'lcam_left', 'lcam_right']

# Specify dataloader configuration. The dataloader operates in the following way:
# 1. It loads a subset of the dataset to RAM.
# 2. It serves mini-batches from this subset. The mini-batches are of data-sequences. So for example, if the sequence length is 2, then the mini-batch will have samples of 2 frames each. A batch size of 16 means that the mini-batch will contain 16 pairs of images, for the example of images.
# 3. The dataloader will load a new subset of the dataset to RAM while the mini-batches are loaded from the first subset, and switch the subsets when the first subset is exhausted. If the first subset is exhausted before the mini-batches are loaded, then the dataloader will keep loading mini-batches from the first subset until the second subset is loaded.

new_image_shape_hw = [640, 640] # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
subset_framenum = 10 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
seq_length = {'image': 2, 'depth': 1} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
batch_size = 8 # This is the number of data-sequences in a mini-batch.
num_workers = 4 # This is the number of workers to use for loading the data.
shuffle = False # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)\

# Create a dataloader object.
dataloader = ta.dataloader(env = envs, 
            difficulty = difficulties, 
            trajectory_id = trajectory_ids, 
            modality = modalities, 
            camname = camnames, 
            new_image_shape_hw = new_image_shape_hw, 
            subset_framenum = subset_framenum, 
            frame_skip = frame_skip, 
            seq_length = seq_length, 
            seq_stride = seq_stride, 
            batch_size = batch_size, 
            num_workers = num_workers, 
            shuffle = shuffle)


# Iterate over the dataloader and visualize the output.

# Iterate over the batches.
while True:
    
        # Get the next batch.
        batch = next(dataloader)
        # Check if the batch is None.
        if batch is None:
            break

        # Get the data-sequences.
        seqs = batch['seqs']
    
        # Get the number of data-sequences in the batch.
        batch_size = len(seqs)
    
        # Iterate over the data-sequences.
        for seq_idx in range(batch_size):
    
            # Get the data-sequence.
            seq = seqs[seq_idx]
    
            # Get the number of frames in the data-sequence.
            seq_length = len(seq)
    
            # Iterate over the frames in the data-sequence.
            for frame_idx in range(seq_length):
    
                # Get the frame.
                frame = seq[frame_idx]
    
                # Get the modalities.
                images = frame['image']
                depths = frame['depth']
                segs = frame['seg']
                flows = frame['flow']
                poses = frame['pose']
    
                # Get the number of modalities.
                num_modalities = len(images)
    
                # Iterate over the modalities.
                for modality_idx in range(num_modalities):
    
                    # Get the modality.
                    image = images[modality_idx]
                    depth = depths[modality_idx]
                    seg = segs[modality_idx]
                    flow = flows[modality_idx]
                    pose = poses[modality_idx]
    
                    # Get the camera name.
                    camname = camnames[modality_idx]
    
                    # Visualize the frame.
                    ta.visualize(image, depth, seg, flow, pose, camname)
    
            # Wait for a key press.
            ta.wait_for_key_press()
    
        # Print the progress.
        print('Batch {}/{}'.format(batch_idx + 1, num_batches))