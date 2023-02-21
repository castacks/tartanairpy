
import json
import networkx as nx
import numpy as np
from pyquaternion import Quaternion
import torch

from .ftensor import frame_graph as fg
from .ftensor import f_eye

def rot_mat_from_quat(x, y, z, w):
    rot_mat = Quaternion(w=w, x=x, y=y, z=z).rotation_matrix
    return torch.from_numpy(rot_mat).to(dtype=fg.FLOAT_TYPE)

def rot_mat_from_1d_array(a):
    na = np.array(a, dtype=np.float32).reshape((3, 3))
    return torch.from_numpy(na).to(dtype=fg.FLOAT_TYPE)

def parse_orientation( orientation_dict ):
    '''
    orientation_dict is a dictionary representing a rotation.
    There are two types of keys:
    
    "type": "rotation_matrix",
    "data": [ rotation matrix represented as 1D array ]
    
    or
    
    "type": "quaternion:,
    "data": {"x": x, "y": y, "z": z, "w": w}
    '''
    t = orientation_dict['type']
    if t == 'rotation_matrix':
        return rot_mat_from_1d_array( orientation_dict['data'] )
    elif t == 'quaternion':
        return rot_mat_from_quat( orientation_dict['data']['x'],
                                  orientation_dict['data']['y'],
                                  orientation_dict['data']['z'],
                                  orientation_dict['data']['w'] )
    else:
        raise Exception(f'Unsupported type of orientation. type = {t}. ')

def parse_position_orientation( pose_dict ):
    position = torch.Tensor( pose_dict['position'] ).to(dtype=fg.FLOAT_TYPE)
    orientation = parse_orientation( pose_dict['orientation'] )
    return position, orientation

def parse_frames( json_obj, G ):
    '''
    Parse the frames in the JSON object and add them to G.
    '''
    
    # Get the element of "frames".
    frames = json_obj['frames']
    
    # Loop over all the frames.
    for entry in frames:
        G.add_frame( fg.RefFrame( entry['name'], entry['comment'] ) )

def parse_typical_transforms( json_obj ):
    # Get the "typical_poses" element.
    typical_poses = json_obj['typical_poses']
    
    # Loop over all the typical poses and create the transforms.
    typical_transforms = dict()
    for key, value in typical_poses.items():
        # Value is also a dictionary.
        position, orientation = parse_position_orientation( value )
        
        # Add the pose as a transform to typical_transforms.
        typical_transforms[key] = {
            'position': position,
            'orientation': orientation
        }
    
    return typical_transforms

def parse_frame_graph( json_obj, G, typical_transforms ):
    # Get the "transforms" element.
    transforms = json_obj['transforms']
    
    # Loop over all the entries.
    for entry in transforms:
        # Initialize an FTensor.
        ft = f_eye(4, f0=entry['f0'], f1=entry['f1'])
        
        # Parse the "pose" element.
        pose_element = entry['pose']
        pose_type = pose_element['type']
        
        if pose_type == 'create':
            # Create a new transform.
            position, orientation = parse_position_orientation(pose_element)
        elif pose_type == 'reference':
            # Copy from typical_transforms.
            key = pose_element['key']
            position = typical_transforms[key]['position']
            orientation = typical_transforms[key]['orientation']
        else:
            raise Exception(f'Unsupported pose type. type = {pose_type}. ')
        
        # Update ft.
        ft.translation = position
        ft.rotation = orientation
        
        # Add ft to the graph.
        G.add_or_update_pose_edge(ft)

def read_frame_graph(fn, additional_info=False):
    '''
    Read a frame graph from a JSON file.
    '''
    
    # Read the JSON file.
    with open(fn, 'r') as fp:
        json_obj = json.load(fp)
    
    # Creat the frame graph.
    G = fg.FrameGraph()
    
    # Parse the frames.
    parse_frames( json_obj, G )
    
    # Parse the typical transforms.
    typical_transforms = parse_typical_transforms( json_obj )
    
    # Parse the transforms between the frames and add edges to the graph.
    parse_frame_graph( json_obj, G, typical_transforms )
    
    if not additional_info:
        return G
    
    additional_info_dict = dict()
    for key, value in json_obj.items():
        if key not in ('frames', 'typical_poses', 'transforms'):
            additional_info_dict[key] = value
            
    return G, additional_info_dict

def node_2_dict(n):
    return {
        'name': n['data'].name,
        'comment': n['data'].comment
    }

def edge_2_dict(e):
    '''
    Return the dictionary representation of an edge.
    '''
    ft = e['pose']
    position_array = ft.translation.cpu().numpy().reshape((-1,))
    rotation_array = ft.rotation.cpu().numpy().reshape((-1,))
    
    return {
        'f0': ft.f0,
        'f1': ft.f1,
        'pose': {
            'type': 'create',
            'position': position_array,
            'orientation': {
                'type': 'rotation_matrix',
                'data': rotation_array
            }
        }
    }