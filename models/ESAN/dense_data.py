import torch
import math
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


class DenseGraph2Subgraph:
    def __init__(self, pbar=None):
        self.pbar = pbar

    def __call__(self, *args, **kwargs):
        if self.pbar is not None:
            next(self.pbar)
        return self.to_subgraphs(*args, **kwargs)

    def to_subgraphs(self, data):
        raise NotImplementedError


class DenseEdgeDeleted(DenseGraph2Subgraph):
    def to_subgraphs(self, x, adj, edge_attr, mask):
        adj_upper = torch.tril(adj, diagonal=-1)
        indices = adj_upper.nonzero()
        i = indices[:, 0]
        j = indices[:, 1]
        M = indices.shape[0]

        subgraph_x = x.unsqueeze(0).repeat(M, *[1] * len(x.shape))
        subgraph_adj = torch.zeros(M, *adj.shape)
        edge_attr.unsqueeze(0).repeat(M, *[1] * len(edge_attr.shape))
        subgraph_edge_attr = (
            edge_attr.unsqueeze(0).repeat(M, *[1] * len(edge_attr.shape)).clone()
        )
        subgraph_mask = mask.unsqueeze(0).repeat(M, *[1] * len(mask.shape))
        subgraph_idx = torch.arange(M, device=x.device)

        # remove edges and attributes
        subgraph_adj[range(M), i, j] = 0
        subgraph_adj[range(M), j, i] = 0
        subgraph_edge_attr[range(M), i, j] = 0
        subgraph_edge_attr[range(M), j, i] = 0

        return subgraph_x, subgraph_adj, subgraph_edge_attr, subgraph_mask, subgraph_idx


class DenseNodeDeleted(DenseGraph2Subgraph):
    def to_subgraphs(self, x, adj, edge_attr, mask):
        N = mask.sum().item()
        subgraph_x = x.unsqueeze(0).repeat(N, *[1] * len(x.shape))
        subgraph_adj = torch.zeros(N, *adj.shape, device=adj.device)
        edge_attr.unsqueeze(0).repeat(N, *[1] * len(edge_attr.shape))
        subgraph_edge_attr = (
            edge_attr.unsqueeze(0).repeat(N, *[1] * len(edge_attr.shape)).clone()
        )
        subgraph_mask = mask.unsqueeze(0).repeat(N, *[1] * len(mask.shape))
        subgraph_idx = torch.arange(N, device=x.device)

        # remove nodes and adjacent edges
        r = list(range(N))
        nodes = mask.nonzero().squeeze()
        subgraph_x[r, nodes] = 0
        subgraph_adj[r, nodes] = 0
        subgraph_adj.transpose(1, 2)[r, nodes] = 0
        subgraph_edge_attr[r, nodes] = 0
        subgraph_edge_attr.transpose(1, 2)[r, nodes] = 0

        return subgraph_x, subgraph_adj, subgraph_edge_attr, subgraph_mask, subgraph_idx


def k_hop_subgraphs_dense(A, start_nodes, num_hops):
    n = A.shape[0]
    batch_size = len(start_nodes)
    A_ = A + torch.eye(n, device=A.device)

    reachable_nodes = torch.zeros((batch_size, n), device=A.device)
    reachable_nodes[range(batch_size), start_nodes] = 1
    reachable_nodes = reachable_nodes @ torch.matrix_power(A_, num_hops)
    mask = reachable_nodes>0 + 0
    mask = torch.einsum('bi,bj->bij', (mask, mask))
    ego_A = A.repeat(batch_size, 1, 1).clone()
    return mask*ego_A



class DenseEgoNets(DenseGraph2Subgraph):
    def __init__(
        self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None
    ):
        super().__init__(pbar=pbar)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def n_subgraphs(self, data):
        return data.num_nodes

    def create_subgraph(self, data, subgraph_idx):
        pass

    def to_subgraphs(self, x, adj, edge_attr, mask, directed=False, sample_fraction=1, subgraph_indices=None):
        # sample concrete adjaceny matrix
        discrete_adj = torch.bernoulli(adj).detach()
        if not directed:
            discrete_adj = torch.tril(discrete_adj, diagonal=-1)
            discrete_adj = discrete_adj + discrete_adj.transpose(-2, -1)

        # sample subgraphs
        num_nodes = mask.sum().item()
        if subgraph_indices is None:
            num_subgraphs = math.ceil(sample_fraction * num_nodes)
            node_indices = mask.nonzero().squeeze()
            subgraph_indices = node_indices[torch.randperm(num_nodes)][:num_subgraphs].tolist()
        else:
            num_subgraphs = len(subgraph_indices)

        subgraph_x = x.unsqueeze(0).repeat(num_subgraphs, *[1] * len(x.shape))
        subgraph_mask = mask.unsqueeze(0).repeat(num_subgraphs, *[1] * len(mask.shape))
        subgraph_idx = torch.arange(num_subgraphs, device=x.device)

        subgraph_adj = adj.unsqueeze(0) * k_hop_subgraphs_dense(discrete_adj, subgraph_indices, self.num_hops)
        subgraph_edge_attr = edge_attr.unsqueeze(0).repeat(num_subgraphs, *[1] * len(edge_attr.shape))
        subgraph_edge_attr = subgraph_edge_attr * subgraph_adj.unsqueeze(-1)

        if self.add_node_idx:
            ids = (
                torch.arange(2, dtype=torch.int32, device=x.device)
                .unsqueeze(0)
                .repeat(num_subgraphs, num_nodes, 1)
            )
            r = list(range(num_subgraphs))
            ids[r, subgraph_indices] = torch.stack((ids[r, subgraph_indices, 1], ids[r, subgraph_indices, 0]), dim=-1)

            ids_full = torch.zeros(
                (*subgraph_x.shape[:-1], 2), dtype=torch.int32, device=x.device
            )
            nodes = mask.nonzero().squeeze()
            ids_full[:, nodes] = ids
            subgraph_x = torch.cat((subgraph_x, ids_full), dim=-1)

        return subgraph_x, subgraph_adj, subgraph_edge_attr, subgraph_mask, subgraph_idx


def densepolicy2transform(
    policy: str, num_hops, sample_fraction, process_subgraphs=lambda x: x, pbar=None
):
    if policy == "edge_deleted":
        generate_subgraphs = DenseEdgeDeleted(
            process_subgraphs=process_subgraphs, pbar=pbar
        )
    elif policy == "node_deleted":
        generate_subgraphs = DenseNodeDeleted(
            process_subgraphs=process_subgraphs, pbar=pbar
        )
    elif policy == "ego_nets":
        generate_subgraphs = DenseEgoNets(
            num_hops, process_subgraphs=process_subgraphs, pbar=pbar
        )
    elif policy == "ego_nets_plus":
        generate_subgraphs = DenseEgoNets(
            num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar
        )
    else:
        raise ValueError("Invalid subgraph policy type")

    def transform(x, adj, edge_attr, mask, id, subgraph_indices=None):
        batch_size = x.shape[0]
        transformed_x = []
        transformed_adj = []
        transformed_edge_attr = []
        transformed_mask = []
        transformed_idx = []
        graph_idx = []
        for i in range(batch_size):
            args = x[i], adj[i], edge_attr[i], mask[i]
            (
                subgraph_x,
                subgraph_adj,
                subgraph_edge_attr,
                subgraph_mask,
                subgraph_idx,
            ) = generate_subgraphs(*args, subgraph_indices=subgraph_indices)

            transformed_x.append(subgraph_x)
            transformed_adj.append(subgraph_adj)
            transformed_edge_attr.append(subgraph_edge_attr)
            transformed_mask.append(subgraph_mask)
            graph_idx.append(i * torch.ones(subgraph_x.shape[0], device=x.device, dtype=torch.int64))
            transformed_idx.append(subgraph_idx)

        x_shape = list(x.shape[1:])
        if policy == "ego_nets_plus":
            x_shape[-1] += 2
        _x = torch.concat(transformed_x, dim=0).view(-1, *x_shape)
        _adj = torch.concat(transformed_adj, dim=0).view(-1, *adj.shape[1:])
        _edge_attr = torch.concat(transformed_edge_attr, dim=0).view(
            -1, *edge_attr.shape[1:]
        )
        _mask = torch.concat(transformed_mask, dim=0).view(-1, *mask.shape[1:])
        graph_idx = torch.concat(graph_idx, dim=0).view(-1)
        subgraph_idx = torch.concat(transformed_idx, dim=0).view(-1)

        return Data(
            x=_x,
            adj=_adj,
            edge_attr=_edge_attr,
            mask=_mask,
            graph_idx=graph_idx,
            subgraph_idx=subgraph_idx,
            original_adj=adj,
            original_mask=mask,
            original_edge_attr=edge_attr,
            id=id,
        )

    return transform
