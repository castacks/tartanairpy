
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from .ftensor import ( FTensor, f_eye )

class TransStruct(object):
    def __init__(self, x, y, z, f0):
        super().__init__()
        self._ft = FTensor(
            torch.Tensor( [ x, y, z ] ).to(dtype=torch.float32),
            f0=f0)
    
    @property
    def x(self):
        return self._ft[0]
    
    @property
    def y(self):
        return self._ft[0]
    
    @property
    def z(self):
        return self._ft[0]
    
    @x.setter
    def x(self, value):
        self._ft[0] = value
        
    @y.setter
    def y(self, value):
        self._ft[1] = value
    
    @z.setter
    def z(self, value):
        self._ft[2] = value
        
    @property
    def ft(self):
        return self._ft

class RotMatStruct(object):
    def __init__(self, array, f0, f1):
        super().__init__()
        
        assert array.shape == (3, 3), \
            f'array must be (3, 3), but got {array.shape}'
            
        self._ft = FTensor(
            torch.Tensor(array).to(dtype=torch.float32),
            f0=f0,
            f1=f1,
            rotation=True)
    
    # Only provide the property function to protect the internal data.
    @property
    def ft(self):
        return self._ft

class RotQuatStruct(object):
    def __init__(self, x, y, z, w, f0, f1):
        super().__init__()
        # The SciPy ordering.
        self.array = np.array([x, y, z, w], dtype=np.float32)
        
        # The FTensor.
        self._ft = FTensor(
            torch.from_numpy( R.from_quat(self.array) ),
            f0=f0,
            f1=f1,
            rotation=True)
    
    # Only provide the property function to protect the internal data.
    @property
    def x(self):
        return self.array[0]
    
    @property
    def y(self):
        return self.array[1]
    
    @property
    def z(self):
        return self.array[2]
    
    @property
    def w(self):
        return self.array[3]
    
    @property
    def ft(self):
        return self._ft

class PoseStruct(object):
    def __init__(self, t, r, f0, f1):
        super().__init__()
        
        # Translation part.
        if isinstance( t, np.array ):
            self._t = TransStruct( *t, f0 )
        elif isinstance( t, TransStruct ):
            assert t.tf.f0 == f0, \
                f't is not consistent with f0. \nt = {t}, f0 = {f0}'
            self._t = t
        else:
            raise Exception(f'Must be np.array or TransStruct. Got {type(t)}')

        # Rotation part.
        if isinstance( r, RotMatStruct ):
            assert r.ft.f0 == f0 and r.ft.f1 == f1, \
                f'r is not consistent with f0 and f1. \nr = {r}\nf0 = {f0}, f1 = {f1}'
            self._rot_mat = r
            q = R.from_matrix(r.ft.tensor().cpu().numpy()).as_quat()
            self._q = RotQuatStruct( *q, f0, f1 )
        elif  isinstance( r, RotQuatStruct ):
            assert r.ft.f0 == f0 and r.ft.f1 == f1, \
                f'r is not consistent with f0 and f1. \nr = {r}\nf0 = {f0}, f1 = {f1}'
            self._rot_mat = RotMatStruct( r.ft.tensor().cpu().numpy(), f0, f1 )
            self._q = r
        else:
            raise Exception(f'Expected to be RotMatStruct or RotQuatStruct. Got {type(r)}')
        
        # Transform matrix as ftensor.
        self._T = f_eye(4, f0, f1)
        self._T.rotation = self._rot_mat.ft
        self._T.translation = self._t
    
    # Only provide property functions to protect the internal values.
    @property
    def t(self):
        return self._t
    
    @property
    def rot_mat(self):
        return self._rot_mat
    
    @property
    def q(self):
        return self._q
    
    @property
    def trans_ft(self):
        return self._T