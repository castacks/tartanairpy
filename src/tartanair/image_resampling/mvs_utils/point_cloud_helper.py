
import copy
import numpy as np
from plyfile import PlyData, PlyElement

def convert_2D_array_2_1D_list(a):
    """
    Return a 2D NumPy array into a list of tuples.
    """

    res = []

    for i in range(a.shape[0]):
        t = tuple( a[i,:].tolist() )
        res.append( t )
    
    return res

def write_PLY(fn, coor, color=None, binary=True):
    """
    fn: The output filename.
    coor (NumPy array): (3, N)
    color (NumPy array): The color vector. The vector could be (N,) or (C, N).
           C could be 1 or 3. If color==None, no color properties 
           will be in the output PLY file.
    binary: Set True to write a binary format PLY file. Set False for
            a ASCII version.
    """

    # Need a table-like array.
    coor = coor.astype(np.float32).transpose()

    # Handle color.
    if ( color is not None ):
        if ( 1 == color.ndim ):
            color = np.stack([ color, color, color ], axis=0)
        
        # Need a table-like array.
        color = color.transpose()
        
        # Clip and convert to uint8.
        color = np.clip( color, 0, 255 ).astype(np.uint8)

        # Concatenate.
        vertex = np.concatenate([coor, color], axis=1)

        # Create finial vetex array.
        vertex = convert_2D_array_2_1D_list(vertex)
        vertex = np.array( vertex, dtype=[\
            ( "x", "f4" ), \
            ( "y", "f4" ), \
            ( "z", "f4" ), \
            ( "red", "u1" ), \
            ( "green", "u1" ), \
            ( "blue", "u1" ) \
            ] )
    else:
        coor = convert_2D_array_2_1D_list(coor)
        vertex = np.array( coor, dtype=[\
            ( "x", "f4" ), \
            ( "y", "f4" ), \
            ( "z", "f4" ) \
            ] )
    
    # Save the PLY file.
    el = PlyElement.describe(vertex, "vertex")

    PlyData([el], text= (not binary) ).write(fn)
    