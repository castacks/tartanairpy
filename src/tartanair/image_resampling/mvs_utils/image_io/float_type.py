
import numpy as np

def c4_uint8_as_float(img):
    assert img.ndim == 3 and img.shape[2] == 4, f'img.shape = {img.shape}. Expecting image.shape[2] == 4. '
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<f4')

def float_as_c4_uint8(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif img.ndim == 3 and img.shape[2] != 1:
        raise Exception(f'img.shape = {img.shape}. Expecting image.shape[2] == 1. ')
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<u1')
