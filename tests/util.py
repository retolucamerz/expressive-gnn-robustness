import torch
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Data

from datasets.util import paired_edges_order

id = 0

def rdm_graph(
    num_nodes: int,
    edge_prob: float,
    num_node_features: int,
    num_edge_features: int,
    directed: bool = False,
    feature_type: str = "real",  # other option: discrete
):
    edge_index = erdos_renyi_graph(num_nodes, edge_prob, directed)
    order = paired_edges_order(edge_index)
    edge_index = edge_index[:, order]
    num_edges = edge_index.shape[1]
    if feature_type == "discrete":
        x = torch.randint(low=0, high=10, size=(num_nodes, num_node_features))
        edge_attr = torch.randint(low=0, high=10, size=(num_edges, num_edge_features))
    elif feature_type == "real":
        x = torch.rand((num_nodes, num_node_features))
        edge_attr = torch.rand((num_edges, num_edge_features))
    if not directed:
        edge_attr[1::2] = edge_attr[0::2]

    y = torch.randn(1)
    global id
    id += 1
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=id)
