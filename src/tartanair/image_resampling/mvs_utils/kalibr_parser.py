
import argparse
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

import torch

from .ftensor import ( FTensor, RefFrame, FrameGraph, compose )
from .frame_io import ( read_frame_graph, node_2_dict, edge_2_dict )

from .file_sys import test_directory_by_filename
from .pretty_dict import ( 
    PrettyDict, PlainPrinter, DictPrinter, ListPrinter, NumPyPrinter, NumPyLineBreakPrinter )

KEY_KLB_FRAME = 'kalibr_frame'
KEY_MVS_FRAME = 'mvs_frame'
KEY_FT_K_C    = 'FT_K_C'

def read_customized_kalibr_yaml(fn):
    global KEY_KLB_FRAME, KEY_MVS_FRAME, KEY_FT_K_C
    
    with open(fn, 'r') as file:
        yobj = yaml.safe_load(file)
        
    # Check if the special keys are present.
    for key, value in yobj.items():
        assert \
            KEY_KLB_FRAME in value and \
            KEY_MVS_FRAME in value and \
            KEY_FT_K_C in value, \
            f'Key {KEY_KLB_FRAME, KEY_MVS_FRAME, KEY_FT_K_C} are not all in the {key} section. '
            
    return OrderedDict( sorted( yobj.items() ) )

def build_kalibr_frame_graph(yobj):
    '''
    Assume that yobj is an OrderedDict.
    '''
    
    G = FrameGraph()
    
    # Add frames as nodes and create edges.
    first_node = True
    frame_dict_list = []
    transform_dict_list = []
    mvs_frame_2_kalibr_map = dict()
    for key, value in yobj.items():
        # Add the Kalibr frame as a graph node.
        kalibr_frame_name = value[KEY_KLB_FRAME]
        G.add_frame( RefFrame( kalibr_frame_name, f'Kalibr Frame {kalibr_frame_name}') )
        
        # Get a dict representation.
        frame_dict_list.append( node_2_dict( G.g.nodes[kalibr_frame_name] ) )
        
        # Add the camera image frame as a graph node.
        mvs_frame_name = value[KEY_MVS_FRAME]
        G.add_frame( RefFrame( mvs_frame_name, f'Camera Image Frame {mvs_frame_name} from Kalibr') )
        mvs_frame_2_kalibr_map[mvs_frame_name] = key
        
        # Get a dict representation.
        frame_dict_list.append( node_2_dict( G.g.nodes[mvs_frame_name] ) )
        
        # Add en edge between the Kalibr frame and the MVS frame.
        FT_K_C = np.array( value['FT_K_C']['T'], dtype=np.float32 )
        ft_k_c = FTensor( torch.from_numpy(FT_K_C), f0=value['FT_K_C']['f0'], f1=value['FT_K_C']['f1'] )
        G.add_or_update_pose_edge(ft_k_c)
        
        # Get a dict representation of the edge between the Kalibr frame and the camera image frame.
        transform_dict_list.append( edge_2_dict( G.g[kalibr_frame_name][mvs_frame_name] ) )
        
        # Prepare for the edge between Kalibr frames.
        if first_node:
            first_node = False
            f1 = kalibr_frame_name
            continue
        
        f0 = kalibr_frame_name
        
        # Get the T_cn_cnm1 element.
        T_cn_cnm1 = np.array( value['T_cn_cnm1'], dtype=np.float32 )
        
        # Kalibr uses the inverse definition compared with ours.
        ft = FTensor( torch.from_numpy(T_cn_cnm1), f0=f0, f1=f1 )
        
        # Add the edge to the graph.
        G.add_or_update_pose_edge(ft)
        
        # Get a dict representation of the edge between the Kalibr frames.
        transform_dict_list.append( edge_2_dict( G.g[f0][f1] ) )
        
        # Overwrite kalibr_f0.
        f1 = f0

    return G, mvs_frame_2_kalibr_map, frame_dict_list, transform_dict_list

def write_frame_graph_components(fn, frame_dict_list, transform_dict_list):
    pd = PrettyDict()
    
    # Populate the values of pd.
    pd['frames'] = frame_dict_list
    pd['transforms'] = transform_dict_list
    
    # Assign printers automatically.
    pd.auto_update_printer()
    
    # Override the printer for the orientation represented as rotation_matrix.
    for i, transform in enumerate(pd['transforms']):
        if transform['pose']['type'] == 'create':
            if transform['pose']['orientation']['type'] == 'rotation_matrix':
                pd.p['transforms'][i]['pose']['orientation']['data'] = NumPyLineBreakPrinter(shape=(3, 3))
                
    # Visualize.
    s = pd.make_str()
    print(s)

    # Write.
    with open(fn, 'w') as fp:
        fp.write( s )

def visualize_frames(G, frame_names, base_frame_name):
    tm = TransformManager()
    for frame in frame_names:
        # Find the transform between frame and base_frame
        # measured in base_frame.
        ft = G.query_transform(f0=base_frame_name, f1=frame)
        
        # Convert the FTensor to NumPy array.
        R = ft.rotation.tensor().numpy()
        t = ft.translation.tensor().numpy()
        
        # Create a pytransform3d transform.
        tr = pt.transform_from( R=R, p=t )
        tm.add_transform( frame, base_frame_name, tr )
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax = tm.plot_frames_in(base_frame_name, ax=ax, s=0.1)
    
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.2, 0.2)
    plt.show()

def parse_for_frame_graph(args, yobj):
    # Build the graph for the Kalibr cameras.
    G_kalibr, mvs_frame_2_kalibr_map, \
    frame_dict_list, transform_dict_list = \
        build_kalibr_frame_graph(yobj)
    
    # Generate the part of a JSON formated Frame Graph for all the Kalibr frames
    # and camera image frames.
    test_directory_by_filename(args.out_graph_components)
    write_frame_graph_components(args.out_graph_components, frame_dict_list, transform_dict_list)
    
    # Read the frame graph from the JSON file.
    # additional_info is a dict that contains a key of calib_cam_frame, which is the
    # anchor frame of all the cameras in the Kalibr calibration YAML file. calib_cam_frame
    # must be present in the YAML file and has a key of KEY_MVS_FRAME.
    frame_graph, additional_info = read_frame_graph(args.graphfile, additional_info=True)
    mvs_frames = [ key for key in mvs_frame_2_kalibr_map.keys() ]
    
    assert additional_info['calib_cam_frame'] in mvs_frame_2_kalibr_map, \
        f'Kalibr results do not contain a {KEY_MVS_FRAME} with the value of {additional_info["calib_cam_frame"]}'
        
    # Compose the two graphs.
    composed = compose(frame_graph, G_kalibr)
    
    # Draw the graph for visualization.
    vis_pos = nx.spring_layout(composed.g)
    nx.draw_networkx_nodes(composed.g, vis_pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(composed.g, vis_pos)
    nx.draw_networkx_edges(composed.g, vis_pos, edge_color='r', arrows=True)
    plt.show()
    
    # Visualize the image frames in 3D.
    visualize_frames(composed, mvs_frames, additional_info['calib_cam_frame'])
    
    return mvs_frame_2_kalibr_map

def write_camera_model_repr_dict(fn, camera_model_repr_dict):
    pd = PrettyDict()
    
    # Populate the values of pd.
    pd['camera_models'] = camera_model_repr_dict
    
    # Assign printers automatically.
    pd.auto_update_printer()
    
    # Visualize.
    s = pd.make_str()
    print(s)

    # Write.
    with open(fn, 'w') as fp:
        fp.write( s )

def parse_for_manifest(args, yobj, mvs_frame_2_kalibr_map):
    # Create a dict of camra model representations.
    camera_model_repr_dict = dict()
    for mvs_key, kalibr_key in mvs_frame_2_kalibr_map.items():
        # Get the Kalibr dict.
        kalibr_cam_dict = yobj[kalibr_key]
        
        assert kalibr_cam_dict['camera_model'] == 'ds', \
            f'Only supports Double Sphere model (ds). Got {kalibr_cam_dict["camera_model"]}'
        
        intrinsics = kalibr_cam_dict['intrinsics']
        resolution = kalibr_cam_dict['resolution']
        
        camera_model_repr_dict[f'cam_model_{mvs_key}'] = {
            'type': 'DoubleSphere',
            'xi': intrinsics[0],
            'alpha': intrinsics[1],
            'fx': intrinsics[2],
            'fy': intrinsics[3],
            'cx': intrinsics[4],
            'cy': intrinsics[5],
            'fov_degree': kalibr_cam_dict['fov_degree'],
            'shape_struct': { 
                'H': resolution[1], 
                'W': resolution[0] 
            },
            'in_to_tensor': True,
            'out_to_numpy': False
        }
    
    # Prepare the output directory.
    test_directory_by_filename(args.out_camera_models)
    
    # Save the dict.
    write_camera_model_repr_dict(args.out_camera_models, camera_model_repr_dict)

def handle_args():
    parser = argparse.ArgumentParser(description='Parse Kalibr YAML file for MVS. ')
    
    parser.add_argument('infile', type=str,
                        help='The YAML file from Kalibr. ')
    parser.add_argument('graphfile', type=str,
                        help='The JSON file describing a graph that has one camera of Kalibr and a frame defined by MVS. ')
    parser.add_argument('--out-graph-components', type=str, default='./frame_graph_components.json',
                        help='The output filename for the frame graph components generated from infile and graphfile. ')
    parser.add_argument('--out-camera-models', type=str, default='./camera_models.json',
                        help='The output filename for the camera models. ')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = handle_args()
    
    # Read the Kalibr YAML file.
    yobj = read_customized_kalibr_yaml(args.infile)
    print(yobj)
    
    mvs_frame_2_kalibr_map = parse_for_frame_graph(args, yobj)
    parse_for_manifest(args, yobj, mvs_frame_2_kalibr_map)
    