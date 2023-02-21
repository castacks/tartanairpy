
# The following is for compatibility issue with PyTorch 1.8 on 
# Jetson Xavier NX with Jetpack 4.6, Python 3.6.

import re
import torch

# Test torch.meshgrid()
try:
    torch.meshgrid( torch.rand((3,)), torch.rand((3,)), indexing='ij' )
    
    # No error.
    def torch_meshgrid(*args, indexing='xy'):
        res = torch.meshgrid(*args, indexing=indexing)
        return [ r.contiguous() for r in res ]
    
except TypeError as exc:
    print('meshgrid() compatibility issue detected. The exception is: ')
    print(exc)
    
    # exc must mention the word `indexing`.
    _m = re.search(r'indexing', str(exc))
    assert _m is not None, \
        f'The exception is not the expected one, which should contain the key word "indexing". '
    
    print('Use a customized version of torch.meshgrid(). ')

    def meshgrid_ij(*args):
        res = torch.meshgrid(*args)
        return [ r.contiguous() for r in res ]

    def meshgrid_xy(*args):
        res = torch.meshgrid(*args[::-1])
        return [ r.contiguous() for r in res[::-1] ]

    def torch_meshgrid(*args, indexing='xy'):
        if indexing == 'xy':
            return meshgrid_xy(*args)
        elif indexing == 'ij':
            return meshgrid_ij(*args)
        else:
            raise Exception(f'Expect indexing to be either "xy" or "ij". Got {indexing}. ')
