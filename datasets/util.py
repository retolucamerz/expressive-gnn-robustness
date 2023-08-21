import torch
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import Embedding

REGRESSION_DATASETS = ["ZINC"]
OGB_DATASETS = ["ogbg-molhiv", "ogbg-molhiv-sm", "ogbg-molpcba"]


def paired_edges_order(edge_index):
    """provides a new order for edges such that the directed edges at indices
    2i and 2i+1 belong to the same undirected edge"""
    e = edge_index
    edge_ids = (
        e.min(dim=0)[0] * (e.max() + 1) + e.max(dim=0)[0]
    )  # assigns each edge a unique id
    return torch.sort(edge_ids)[1]

