
import numpy as np

class ShapeStruct(object):
    def __init__(self, H, W, C=-1, **kwargs):
        super().__init__()
        
        self.H = H
        self.W = W
        self._C = C
        
    @property
    def shape(self):
        '''
        This funtion is meant to be used with NumPy, PyTorch, etc.
        '''
        return (self.H, self.W)
    
    @property
    def size(self):
        '''
        This function is meant to be used with OpenCV APIs.
        '''
        return (self.W, self.H)
    
    @property
    def shape_numpy(self):
        return np.array( [ self.H, self.W ], dtype=np.int32 )
    
    @staticmethod
    def read_shape_struct(dict_like):
        '''
        Read shape information from a dict-like object.
        '''
        return ShapeStruct( **dict_like ) \
            if not isinstance(dict_like, ShapeStruct) \
            else dict_like

    @property
    def C(self):
        return self._C

    def __str__(self) -> str:
        return f'{{ "H": {self.H}, "W": {self.W}, "C": {self.C} }}'

    def __repr__(self) -> str:
        return f'ShapeStruct(H={self.H}, W={self.W})'
    
    def __eq__(self, other):
        return self.H == other.H and self.W == other.W and self.C == other.C