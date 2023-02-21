import matplotlib.pyplot as plt
import networkx as nx
import torch

from .ftensor import f_eye
from .frame_graph import RefFrame, FrameGraph

if __name__ == '__main__':
    # Create a directed graph.
    G = FrameGraph()

    # ========== Add nodes. ==========
    awf = RefFrame('awf', 'AirSim World NED Frame. ')
    rbf = RefFrame('rbf', 'Rig Body Frame. ')
    rpf = RefFrame('rpf', 'Rig Panorama Frame. ')
    cbf = RefFrame('cbf', 'Camera Body Frame. ')
    cpf = RefFrame('cpf', 'Camera Panorama Frame. ')
    cif = RefFrame('cif', 'Camera Image Frame. ')

    G.add_frame(awf)
    G.add_frame(rbf)
    G.add_frame(rpf)
    G.add_frame(cbf)
    G.add_frame(cpf)
    G.add_frame(cif)

    # ========== Add edges. ==========

    # RBF.
    T_awf_rbf = f_eye(4, 'awf', 'rbf')
    T_awf_rbf.translation = torch.as_tensor([ 0, 1, 0 ])
    T_awf_rbf = T_awf_rbf.to('cuda')
    G.add_or_update_pose_edge( T_awf_rbf )

    # RPF.
    T_rbf_rpf = f_eye(4, 'rbf', 'rpf')
    T_rbf_rpf.rotation = torch.as_tensor([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0]
    ])
    T_rbf_rpf = T_rbf_rpf.to('cuda')
    G.add_or_update_pose_edge( T_rbf_rpf )

    # A CBF.
    T_rbf_cbf = f_eye(4, 'rbf', 'cbf')
    T_rbf_cbf.translation = torch.as_tensor([ 0, 1, 0 ])
    T_rbf_cbf = T_rbf_cbf.to('cuda')
    G.add_or_update_pose_edge( T_rbf_cbf )

    # A CPF.
    T_cbf_cpf = f_eye(4, 'cbf', 'cpf')
    T_cbf_cpf.rotation = torch.as_tensor([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0]
    ])
    T_cbf_cpf = T_cbf_cpf.to('cuda')
    G.add_or_update_pose_edge( T_cbf_cpf )

    # A CIF.
    T_cbf_cif = f_eye(4, 'cbf', 'cif')
    T_cbf_cif.rotation = torch.as_tensor([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    T_cbf_cif = T_cbf_cif.to('cuda')
    G.add_or_update_pose_edge( T_cbf_cif )

    # ========== End of adding edges. ==========

    # Query a transform.
    tf = G.query_transform(f0='cif', f1='awf', print_path=True)
    print(tf)

    # Should be
    # [[ 0.,  1.,  0., -2.], 
    #  [ 0.,  0.,  1.,  0.], 
    #  [ 1.,  0.,  0.,  0.], 
    #  [ 0.,  0.,  0.,  1.]]

    # Draw the graph for visualization.
    vis_pos = nx.spring_layout(G.g)
    nx.draw_networkx_nodes(G.g, vis_pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G.g, vis_pos)
    nx.draw_networkx_edges(G.g, vis_pos, edge_color='r', arrows=True)
    plt.show()
    