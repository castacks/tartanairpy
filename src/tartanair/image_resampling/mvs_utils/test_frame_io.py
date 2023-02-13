
import os
import sys

# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

import matplotlib.pyplot as plt
import networkx as nx

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from .frame_io import read_frame_graph

def visualize_camera_body_frames(G):
    # The transforms for visualization.
    transforms = [
        { 'f0': 'rbf', 'f1': 'cbf0' },
        { 'f0': 'rbf', 'f1': 'cbf1' },
        { 'f0': 'rbf', 'f1': 'cbf2' },
        { 'f0': 'rbf', 'f1': 'cbf3' },
    ]
    
    # Gather the transforms needed by pytransform3d.
    transforms_for_pyt3d = {}
    for t in transforms:
        f0 = t['f0']
        f1 = t['f1']
        
        # Query the graph for the transform.
        tf = G.query_transform(f0, f1)
        
        # Save.
        transforms_for_pyt3d[f1] = tf
            
    tm = TransformManager()
    for _, value in transforms_for_pyt3d.items():
        # value is an FTensor.
        # Conver the FTensor to pytransform3d format.
        tf = pt.transform_from(
                R=value.rotation.cpu().numpy(), 
                p=value.translation.cpu().numpy())
        
        tm.add_transform(value.f1, value.f0, tf)
        
    ax = tm.plot_frames_in('rbf', s=0.1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    plt.show()

if __name__ == '__main__':
    in_json_fn = os.path.join( _CURRENT_PATH, 'test_data', 'frame_graph.json' )
    
    G = read_frame_graph( in_json_fn )
    
    # Draw the graph for visualization.
    vis_pos = nx.spring_layout(G.g)
    nx.draw_networkx_nodes(G.g, vis_pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G.g, vis_pos)
    nx.draw_networkx_edges(G.g, vis_pos, edge_color='r', arrows=True)
    plt.show()
    
    # Visualize in 3D.
    visualize_camera_body_frames(G)
    
