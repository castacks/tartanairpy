# ftensor
Add reference frames to PyTorch Tensor objects. Reduce the headache of ambiguous definitions of frames and transforms between frames.

# Introduction

Tensor with reference frame definition. Designed for handling vector coordinates and transformation between frames. Dedicated to 2-, 3-, and 4-dimensional spaces. Try to recall how many times you have trouble understanding other people when they say "The pose of the robot is XYZ", "T is a transform between frame A and B", and "M is the extrinsic matrix of camera C". The problem is that when we, casually, say something about coordiantes and the transformation between them, we are skipping some very important pieces of information. The information that is usually missing is what frame is used for defining coordinates and transforms. Furthermore, when we start implementing these things, there are rarely any mechanisms that can watch our back and make sure we are not messing up these things.
        
The core goal for FTensor is dealing with poor and ambiguous definitions of transforms and preventing wrong transformations from happening between coordinate frames. For an FTensor, there are two frames, f0 and f1.

- For vectors, f0 is the frame of the vector coordinates. f1 should always be None.
- For transform matrices, such as rotation matrix and 4x4 transform matrix, f0 is the frame of reference and f1 is the target. To be more specific, when we say that a transform matrix represents the pose of frame f1 w.r.t. frame f0, we mean that this transform matrix records the orientation and position of f1 w.r.t. f0 measured in f0. So the coordinates of a point vector under f1 can be transformed to f0 by p_f0 = T_f0_f1 @ p_f1, where T_f0_f1 is the transform matrix.
        
Note that for vectors represented by FTensor, they are always column vectors, meaning the shape variable of a FTensor vector is (3, 1).
        
Note that for an array of vectors, we support a row of column vectors, meaning that the valid shape of the array should be 2xN, 3xN, or 4xN, where N is the total number of vectors. FTensor also supports higher-dimensional formats as torch.Tensor does. Just make sure that the last two dimensions have the correct shape.
        
FTensor is designed to support 2D to 4D spaces. However, at the moment, we do not have a good way to enforce this restriction. Please use it with caution.
        
FTensor supports all the operations on a normal torch.Tensor. However, some operations might have counterintuitive results. For more details, please refer to the documentation at `xxxx`.
        
Special note: the rotation flag is not used for enforcing a valid SO(3) or 2D rotation, it is used for quickly accessing the sub-matrices of a general 4x4 or 3x3 transformation matrix.

# Prerequisites

FTensor depends on `PyTorch`. 

To use `frame_graph`, we need `networkx`.

# Run the test script.

Due to the limitation of test scripts reside in a Python package, we need to jump outside the package folder and run the test scripts as module-name. Sadly, autocomplete is not available.

```bash
python3 -m ftensor.test_ftensor --disable-amp
python3 -m ftensor.test_frame_graph
```
