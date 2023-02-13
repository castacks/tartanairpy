
# Author
# ======
# 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# 
# Data
# ====
#
# Date: 2022-07-12
#

from typing import Any

import torch
# from torch.utils._pytree import tree_map
from .compatible_pytree import tree_map

DEFAULT_FRAME = 'ori'

# A dict for saving the torch functions that are also implemented in FTensor.
IMPLEMENTATIONS={}

# Decorator for adding a function to the IMPLEMENTATIONS dict.
def implemented(name):
    def dec_fn(impl):
        IMPLEMENTATIONS[name] = impl
        def _inner_fn(*args, **kwargs):
            return impl(*args, **kwargs)
        return _inner_fn
    return dec_fn

class FTensor(torch.Tensor):
    
    @staticmethod
    def __new__(cls, *data, f0:str=DEFAULT_FRAME, f1:str=DEFAULT_FRAME, rotation=False):
        # This is the same way as pypose does.
        tensor = data[0] if isinstance(data[0], torch.Tensor) else torch.Tensor(*data)
        return torch.Tensor.as_subclass(tensor, FTensor)

    def __init__(self, *data, f0:str=DEFAULT_FRAME, f1:str=None, rotation=False):
        '''
        Special note: the rotation flag is not used for enforcing a valid SO(3) or 
        2D rotation, it is used for quickly accessing the sub-matrices of a general 
        4x4 or 3x3 transformation matrix.
        
        Arguments:
        ==========
        data: the appropriate values for constructing a torch.Tensor.
        f0: the string name of the first frame. 
        f1: the string name of the second frame.
        rotation: whether the FTensor is a rotation matrix. It is False if the matrix is a general transform.
        '''
        # Do not need to call super().__init__() here.
        self.f0 = f0
        self.f1 = f1
        assert rotation in (True, False), f'rotation must be True, False, or None. Got {rotation}.'
        self._is_rotation = rotation
        
        # # d0, d1 = self.shape[-2:] # This causes errors when doing slicing.
        # d0 = self.shape[-1]
        # if self.is_transform:
        #     if rotation == True:
        #         assert d0 == 2 or d0 == 3, f'Rotation matrix should be 2x2 or 3x3, but got {self.shape}'
        #     self._is_rotation = rotation
        # else:
        #     self._is_rotation = False
        
    def __repr__(self):
        return f'{self.__class__.__name__}, f0 = {self.f0}, f1 = {self.f1}, rotation = {self._is_rotation}: \n{super().__repr__()}'
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # if func.__name__ in IMPLEMENTATIONS:
        #     return IMPLEMENTATIONS[func.__name__](*args, **kwargs)
        
        unwrapped_types = [
            torch.Tensor
            if t is FTensor 
            else t
            for t in types
        ]

        # Check if all the FTensors has the same frame.
        t_f0 = None
        t_f1 = None
        r    = True

        def first_ftensor(x):
            nonlocal t_f0, t_f1, r
            if t_f0 is not None:
                return x
            
            if isinstance(x, FTensor):
                t_f0 = x.f0
                t_f1 = x.f1
                r    = x._is_rotation

            return x
        
        args   = tree_map(first_ftensor, args)
        kwargs = tree_map(first_ftensor, kwargs)
        
        # This is the same way as pypose does.
        out = torch.Tensor.__torch_function__(func, unwrapped_types, args, kwargs)
        
        # This seems not needed. Because we are forwarding all the function calls to the respective
        # functions of torch.Tensor. And PyTorch guarantees that the returned value is a subclass object.
        # That is, x is always an FTensor object.
        def wrap(x):
            nonlocal t_f0, t_f1, r
            return FTensor(x, f0=t_f0, f1=t_f1, rotation=r) if isinstance(x, torch.Tensor) and not isinstance(x, FTensor) else x
        
        return tree_map(wrap, out)

    def __deepcopy__(self, memo):
        # TODO: Check if this naive implementation is enough.
        return self.detach().clone()

    def tensor(self) -> torch.Tensor:
        '''
        Return the underlying torch.Tensor.
        '''
        return torch.Tensor.as_subclass(self, torch.Tensor)

    def f_to_str(self):
        '''
        Return a string representation of the frames.
        '''
        return f'({self.f0}, {self.f1})'

    def have_same_frames(self, other):
        '''
        Return True if the two FTensors have the same frames.
        '''
        return self.f0 == other.f0 and self.f1 == other.f1

    def have_different_frames(self, other):
        '''
        Return True if the two FTensors have different frames.
        '''
        return self.f0 != other.f0 or self.f1 != other.f1

    @implemented('__eq__')
    def __eq__(self, other):
        '''
        Return False if the two FTensors have different frames.
        Continue checking the equality as torch.Tensor if the frames are the same.
        '''
        if isinstance(other, FTensor):
            if self.have_different_frames(other):
                return False
            if self._is_rotation != other._is_rotation:
                return False
            return super().__eq__(other.tensor())
        else:
            return super().__eq__(other)

    @implemented('__ne__')
    def __ne__(self, other):
        '''
        Return False if the tow FTensors have the same frames.
        Continue checking the inequality as torch.Tensor if the frames are different.
        '''
        return not self.__eq__(other)

    @staticmethod
    def check_frames_equality(t0, t1):
        '''
        Raise an exception if the two FTensors have different frames.
        '''
        assert t0.have_same_frames(t1), \
            f'The frames of two FTensors are not equal. Got {t0.f0} and {t0.f1}, {t1.f0} and {t1.f1}'

    @property
    def is_vector(self):
        return self.f0 is not None and self.f1 is None

    @staticmethod
    def are_vectors(t0, t1):
        return t0.is_vector and t1.is_vector

    @property
    def is_transform(self):
        return self.f0 is not None and self.f1 is not None
    
    @staticmethod
    def are_transforms(t0, t1):
        return t0.is_transform and t1.is_transform

    @staticmethod
    def check_frames_compatibility(t0, t1):
        '''
        For vectors or array of vectors, raise an exception if the t0 and t1 have different f1 values.
        Fro matrices, raise an exception if t0.f1 != t1.f0. Note that, this operation is not commutative.
        (t0, t1) is compatible does NOT imply (t1, t0) is also compatible. 
        '''
        if FTensor.are_vectors(t0, t1):
            assert t0.f0 == t1.f0, f'Inconsistent frames between two vectors. t0.f0 = {t0.f0}, t1.f0 = {t1.f0}. '
        else:
            assert t0.f1 == t1.f0, f'Inconsistent frames: t0.f1 = {t0.f1}, t1.f0 = {t1.f0}'

    @implemented('matmul')
    def __matmul__(self, other):
        '''
        Matrix multiplication between two FTensor matrices or between a matrix and a (array of) vector(s).
        '''
        if isinstance(other, FTensor):
            if self.f1 is None and other.f1 is None:
                raise Exception(f'Cannot perform matrix multiplication between two (arrays of) column vectors. self: {self.f_to_str()}, other: {other.f_to_str()}. ')
            
            FTensor.check_frames_compatibility( self, other )
            out = super().__matmul__( other.tensor() )
            # out = torch.Tensor.as_subclass(out, FTensor)
            out.f0 = self.f0
            out.f1 = other.f1
        else:
            out = super().__matmul__( other )
            # out = torch.Tensor.as_subclass(out, FTensor)
            out.f0 = self.f0
            out.f1 = self.f1
        
        out._is_rotation = self._is_rotation if out.is_transform else False
        
        if out.f0 is None and out.f1 is None:
            return torch.Tensor.as_subclass(out, torch.Tensor)
        else:
            return out

    @staticmethod
    def check_addition_compatibility(t0, t1):
        if FTensor.are_vectors(t0, t1):
            assert FTensor.have_same_frames(t0, t1), \
                f'Vectors must have the same frames. Got {t0.f_to_str()} and {t1.f_to_str()}'
            return True
        return False

    @implemented('add')
    def __add__(self, other):
        '''
        Addition between two FTensor vectors is allowed if they have the same frames.
        '''
        if isinstance(other, FTensor):
            FTensor.check_addition_compatibility( self, other )
            out = super().__add__( other.tensor() )
            # f1 = other.f1 # I need more thoughts on this.
            f1 = self.f1
        else:
            out = super().__add__( other )
            f1 = self.f1

        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f0
        out.f1 = f1
        
        out._is_rotation = self._is_rotation

        return out

    @implemented('radd')
    def __radd__(self, other):
        return self.__add__(other)

    @implemented('sub')
    def __sub__(self, other):
        '''
        Subtract other from self.
        Subtractiong between two FTensor vectors is allowed if they have the same frames.
        '''
        if isinstance(other, FTensor):
            FTensor.check_addition_compatibility( self, other )
            out = super().__sub__( other.tensor() )
        else:
            out = super().__sub__( other )

        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f0
        out.f1 = self.f1
        
        out._is_rotation = self._is_rotation

        return out

    @implemented('rsub')
    def __rsub__(self, other):
        if isinstance(other, FTensor):
            FTensor.check_addition_compatibility( self, other )
            out = super().__rsub__( other.tensor() )
        else:
            out = super().__rsub__( other )

        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f0
        out.f1 = self.f1
        
        out._is_rotation = self._is_rotation

        return out

    @implemented('mul')
    def __mul__(self, other):
        if isinstance(other, FTensor):
            raise Exception(f'Elementwise-multiplication between FTensors is not not allowed. ')
    
        out = super().__mul__( other )
        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f0
        out.f1 = self.f1
        
        out._is_rotation = self._is_rotation

        return out

    @implemented('rmul')
    def __rmul__(self, other):
        if isinstance(other, FTensor):
            raise Exception(f'Elementwise-multiplication between FTensors is not not allowed. ')
    
        out = super().__rmul__( other )
        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f0
        out.f1 = self.f1
        
        out._is_rotation = self._is_rotation

        return out

    @implemented('__setitem__')
    def __setitem__(self, key, newvalue):
        '''
        Used by slice operations.
        '''
        if isinstance(newvalue, FTensor):
            FTensor.check_frames_equality(self, newvalue)
            
        return super().__setitem__(key, newvalue)

    @implemented('transpose')
    def transpose(self, dim0=-2, dim1=-1):
        '''
        Be really careful with FTensor vectors. After transpose, f0 is None and f1 is not None.
        
        The only operation allowed with a vector that has f0 == None and f1 != None 
        (a.k.a. the transposed, row version of a) is multiplication with a matrix or a 
        column vector. And the transposed vector must be on the right side of the multiplication. 
        The result will have f0 == None and even f1 == None, depending on the other operant. 
        Leaving the result as having f0 == None and f1 != None is problematic in later operations.
        '''
        out = super().transpose(dim0, dim1)
        # out = torch.Tensor.as_subclass(out, FTensor)
        out.f0 = self.f1
        out.f1 = self.f0
        
        out._is_rotation = self._is_rotation

        return out

    def check_transformation_matrix(self):
        assert self.is_transform, f'FTensor must be a transformation matrix. self: {self.f_to_str()}'
        assert self._is_rotation == False, f'FTensor must be a non-rotation matrix. rotation: {self._is_rotation}'
        assert self.shape[-2] == self.shape[-1] == 3 or self.shape[-2] == self.shape[-1] == 4, \
            f'The last two dimensions of the FTensor must be 3 or 4. Got {self.shape[-2]} and {self.shape[-1]}. '

    @property
    def translation(self):
        '''
        Return the translation part of the transformation matrix. 
        The result is a FTensor vector with f1 = None.
        
        Note: this function is only valid for transformation matrices.
        Note: this function does not enforce the last row to be [0, 0, ..., 1].
        '''
        self.check_transformation_matrix()
        N = self.shape[-1]
        t = self[..., :N-1, N-1]
        t.f1 = None
        t._is_rotation = False
        return t

    @translation.setter
    def translation(self, t):
        '''
        Set the translation part of the transformation matrix. 
        t can be a plain torch.Tensor.
        
        Note: this function is only valid for transformation matrices.
        Note: this function does not enforce the last row to be [0, 0, ..., 1].
        '''
        self.check_transformation_matrix()
        
        if isinstance(t, FTensor):
            # Override the frame equality check.
            assert t.f0 == self.f0 and t.f1 is None, \
                f'Translation vector must be in frame f0. self: {self.f_to_str()}, t: {t.f_to_str()}. '
            t = t.tensor()
        
        N = self.shape[-1]
        self[..., :N-1, N-1] = t

    @property
    def rotation(self):
        '''
        Return the rotation part of the transformation matrix.
        
        Note: this function is only valid for transformation matrices.
        Note: this function does not garantee that the rotation is a valid SO(3) or 2D rotation.
        Note: this function does not enforce the last row to be [0, 0, ..., 1].
        '''
        self.check_transformation_matrix()
        N = self.shape[-1]
        R = self[..., :N-1, :N-1]
        R._is_rotation = True
        return R

    @rotation.setter
    def rotation(self, R):
        '''
        Set the rotation part of the transformation matrix.
        
        Note: this function is only valid for transformation matrices.
        Note: this function does not check if R is a valid SO(3) or 2D rotation.
        Note: this function does not enforce the last row to be [0, 0, ..., 1].
        '''
        self.check_transformation_matrix()
        # if isinstance(R, FTensor):
        #     assert R.is_rotation, f'R must be a rotation matrix. R: {R}'
        N = self.shape[-1]
        self[..., :N-1, :N-1] = R

    @property
    def is_rotation(self):
        '''
        Note: this function is only for working with translation(), rotation(), and invers().
        A True value does not nessarily mean that the FTensor is a valid SO(3) or 2D rotation.
        '''
        assert self.is_transform, f'Calling is_rotation on a vector is not allowed. self: {self.f_to_str()}. '
        return self._is_rotation
    
    @is_rotation.setter
    def is_rotation(self, value):
        '''
        Note: this function is only for working with translation(), rotation(), and invers().
        Settting True does not nessarily enforce that the FTensor to be a valid SO(3) or 2D rotation.
        '''
        assert self.is_transform, f'Calling is_rotation on a vector is not allowed. self: {self.f_to_str()}. '
        assert value in (True, False), f'value is not a boolean. value = {value}. '
        
        if value:
            if self.shape[-2] == 1 or self.shape[-2] >= 4:
                raise Exception(f'Cannot convert a (presumable) 4x4 transformation matrix to a rotation matrix. self.shape = {self.shape}.')
        
        self._is_rotation = value

    def inverse(self):
        '''
        Only works for rotation and transformation matrices.
        Return the inverse of the transformation.
        
        Note: this function DOES enforce the last row to be [0, 0, ..., 1].
        '''
        assert self.is_transform, f'Inverse is only valid for transformation matrices. self: {self.f_to_str()}. '
        d0, d1 = self.shape[-2:]
        assert d0 == d1 == 2 or d0 == d1 == 3 or d0 == d1 == 4, \
            f'Only supports 2x2, 3x3 or 4x4 matrices. self.shape = {self.shape}. '
        
        N = self.ndim
        if self.is_rotation:
            return self.transpose( N-2, N-1 )
        else:
            base_tensor = torch.eye(d0, dtype=self.dtype, device=self.device)
            if N > 2:
                base_tensor = base_tensor.repeat( *self.shape[:-2], 1, 1 )    
            inv = FTensor( base_tensor, f0=self.f1, f1=self.f0 )
            inv.rotation = self.rotation.transpose(0, 1)
            inv.translation = -inv.rotation @ self.translation
            return inv

def f_eye(N, f0, f1, rotation=False, dtype=torch.float32, device=None):
    '''
    Convinient function to create an FTensor with an identity matrix.
    '''
    ft = torch.Tensor.as_subclass(torch.eye(N, dtype=dtype, device=device), FTensor)
    ft.f0 = f0
    ft.f1 = f1
    ft._is_rotation = False
    return ft

def f_ones(shape, f0, f1, dtype=torch.float32, device=None):
    '''
    Convinient function to create an FTensor with ones.
    '''
    ft = torch.Tensor.as_subclass(torch.ones(shape, dtype=dtype, device=device), FTensor)
    ft.f0 = f0
    ft.f1 = f1
    ft._is_rotation = False
    return ft

def f_zeros(shape, f0, f1, dtype=torch.float32, device=None):
    '''
    Convinient function to create a zero FTensor.
    '''
    ft = torch.Tensor.as_subclass(torch.zeros(shape, dtype=dtype, device=device), FTensor)
    ft.f0 = f0
    ft.f1 = f1
    ft._is_rotation = False
    return ft

if __name__ == '__main__':
    ft = FTensor([ [0], [1], [2] ], f0='cam0', f1=None)
    print(ft)
    
    print(ft.dtype)
    print(ft.shape)
    
    # ===========================================================================
    # ===== There is a dedicated test script now called test_frame_point.py =====
    # ===========================================================================
