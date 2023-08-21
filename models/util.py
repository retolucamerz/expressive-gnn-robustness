import copy
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Data, Batch
import numpy as np
import random


def to_dense_data(
    data, x=None, mask=None, adj=None, edge_attr=None, max_num_nodes=None
):
    if hasattr(data, "batch"):
        batch = data.batch
    else:
        batch = None

    if x is None:
        x, mask = to_dense_batch(data.x, batch)  # batch, N, num_node_feats
    elif mask is None:
        _, mask = to_dense_batch(data.x, batch)  # batch, N, num_node_feats
    if adj is None:
        adj = to_dense_adj(data.edge_index, batch, max_num_nodes=max_num_nodes)
    if edge_attr is None:
        edge_attr = to_dense_adj(
            data.edge_index, batch, data.edge_attr, max_num_nodes=max_num_nodes
        )  # batch, N, N, num_edge_attrs
    return x, mask, adj, edge_attr


def to_sparse_batch(x, adj, edge_attr, y, node_mask=None):
    B, n, _ = adj.shape

    if node_mask is None:
        # only consider nodes that have degree > 0
        node_mask = adj.sum(dim=-1) > 0

    node_number = torch.cumsum(node_mask.to(dtype=torch.int32), dim=-1) - 1

    first_node = node_number.unsqueeze(-1).expand(-1, -1, n)
    second_node = node_number.unsqueeze(-2).expand(-1, n, -1)
    edges = torch.stack((first_node, second_node), dim=-1)

    graphs = []
    for i in range(B):
        x_ = x[i][node_mask[i]].clone().squeeze()
        edge_index_ = edges[i][adj[i] > 0].t().clone()
        edge_attr_ = edge_attr[i][adj[i] > 0].clone().squeeze()
        y_ = y[i].clone()
        graphs.append(Data(x=x_, edge_index=edge_index_, edge_attr=edge_attr_, y=y_))

    return Batch.from_data_list(graphs)


def seed_from_param(initial_seed, *args):
    """Creates a seed based on an initial seed and parameters for
    an experiment. The intended use is to get a fixed seed per parameter
    set, which is different from seeds for other parameter sets."""

    seed = initial_seed
    for arg in args:
        seed = (17 * seed + 19463 * hash(arg)) % 101173 + initial_seed
    return seed


from functools import reduce
import operator
import math

prod = lambda ls: reduce(operator.mul, ls, 1)


def convert_size(size_bytes):
    # from https://stackoverflow.com/a/14822210
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def size_of(tensor):
    return convert_size(tensor.element_size() * prod(tensor.shape))


def count_parameters(model):
    """copied from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_to_leaky_relu(net, negative_slope):
    net = copy.deepcopy(net)  # potential problem: deepcopy on every level of recursion
    for child_name, child in net.named_children():
        if isinstance(child, nn.ReLU):
            setattr(net, child_name, nn.LeakyReLU(negative_slope=negative_slope))
        else:
            convert_to_leaky_relu(child, negative_slope)
    return net

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
