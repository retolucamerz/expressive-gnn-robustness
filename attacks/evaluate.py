import numpy as np
import torch
from models.util import to_dense_data
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from networkx import simple_cycles, number_connected_components


# TARGETS
def classflip_target(
    model, data, x=None, adj=None, edge_attr=None, encode=True, **kwargs
):
    # maximize 'classflip_target' to flip class
    pred = model(
        data, x=x, adj=adj, edge_attr=edge_attr, encode=encode, **kwargs
    ).view(-1)
    y = data.y.view(-1)
    return (1-2*y)*pred


def untargeted(model, data, x=None, adj=None, edge_attr=None, encode=True, **kwargs):
    # maximize 'untargeted' to increase MAE
    pred = model.predict(
        data, x=x, adj=adj, edge_attr=edge_attr, encode=encode, **kwargs
    ).view(-1)
    y = data.y.view(-1)
    return (pred - y).abs()

def untargeted_multiclass(model, data, x=None, adj=None, edge_attr=None, encode=True, **kwargs):
    pred = model(
        data, x=x, adj=adj, edge_attr=edge_attr, encode=encode, **kwargs
    )
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[range(pred.shape[0]),data.y] = False
    return pred[mask].view(pred.shape[0], -1).max(dim=-1)[0] - pred[~mask]


def incr_target(model, data, x=None, adj=None, edge_attr=None, encode=True, **kwargs):
    # maximize 'incr_target' to increase predictions
    pred = model.predict(
        data, x=x, adj=adj, edge_attr=edge_attr, encode=encode, **kwargs
    ).view(-1)
    return pred


def decr_target(model, data, x=None, adj=None, edge_attr=None, encode=True, **kwargs):
    # maximize 'decr_target' to decrease predictions
    pred = model.predict(
        data, x=x, adj=adj, edge_attr=edge_attr, encode=encode, **kwargs
    ).view(-1)
    return -pred


# EVALUATION
def max_degree(adj, mask):
    """computes the per-graph maximum node degree
    adj: a dense (B, N, N) shaped adjacency matrix,
    mask: (B, N) shaped node mask as returend by to_dense_adj"""
    return adj.sum(dim=-1).max(dim=-1)[0]


def min_degree(adj, mask):
    """computes the per-graph minimum node degree
    adj: a dense (B, N, N) shaped adjacency matrix,
    mask: (B, N) shaped node mask as returend by to_dense_adj"""
    adj_ = adj.clone()
    adj_[~mask] = float("inf")  # ignore non-existant nodes
    return adj_.sum(dim=-1).min(dim=-1)[0]


def is_connected(adj, mask):
    """computes which of the graphs are connected
    adj: a dense (B, N, N) shaped adjacency matrix,
    mask: (B, N) shaped node mask as returend by to_dense_adj"""
    B, N, _ = adj.shape
    adj_ = adj.clone()
    adj_[:, range(N), range(N)] = 1
    walks = torch.linalg.matrix_power(adj_, N - 1)

    # ignore edges between non-existant nodes
    walks[~mask] = 1
    walks.transpose(-2, -1)[~mask] = 1

    return (walks > 0.001).all(dim=-1).all(dim=-1)


def is_undirected(sparse_batch):
    batch_size = len(sparse_batch.y)
    is_undirected = [sparse_batch[i].is_undirected() for i in range(batch_size)]
    return torch.tensor(is_undirected) + 0


def count_cycles(sparse_batch):
    num_cycles = []
    num_large_cycles = []
    batch_size = len(sparse_batch.y)
    for i in range(batch_size):
        graph = sparse_batch[i]
        cycles = simple_cycles(to_networkx(graph))
        cycle_lengths = [len(c) for c in cycles]
        num_cycles.append(len([n for n in cycle_lengths if n > 2]))
        num_large_cycles.append(len([n for n in cycle_lengths if n >= 6]))

    num_cycles = torch.tensor(num_cycles) / 2
    num_large_cycles = torch.tensor(num_large_cycles) / 2
    return num_cycles, num_large_cycles


def count_connected_components(sparse_batch):
    batch_size = len(sparse_batch.y)
    graphs = [
        to_networkx(sparse_batch[i], to_undirected=True) for i in range(batch_size)
    ]
    return torch.tensor([number_connected_components(graph) for graph in graphs])


def count_graph_size(sparse_batch):
    num_nodes = []
    num_edges = []
    batch_size = len(sparse_batch.y)
    for i in range(batch_size):
        graph = sparse_batch[i]
        num_nodes.append(graph.num_nodes)
        num_edges.append(graph.num_edges)

    num_nodes = torch.tensor(num_nodes)
    num_edges = torch.tensor(num_edges)
    return num_nodes, num_edges


def find_single_node_attr_change(x, changes):
    B, _, _ = x.shape
    batch_mask = changes.sum(dim=(-2, -1)) == 1
    idx = changes[batch_mask].nonzero()

    mod_x_node = -1 * torch.ones(B, dtype=torch.long, device=x.device)
    node_idx = idx[:, 1]
    mod_x_node[batch_mask] = node_idx

    mod_x_attr = -1 * torch.ones(B, dtype=torch.long, device=x.device)
    attr_idx = idx[:, 2]
    mod_x_attr[batch_mask] = attr_idx

    mod_x_val = -1 * torch.ones(B, device=x.device, dtype=x.dtype)
    batch_idx = batch_mask.nonzero().squeeze()
    mod_x_val[batch_mask] = x[batch_idx, node_idx, attr_idx]

    return mod_x_node, mod_x_attr, mod_x_val


def find_single_edge_change(tensor, directed=False):
    B, N, _ = tensor.shape
    changes = tensor.sum(dim=(-2, -1))
    if not directed:
        changes = changes / 2

    changes_edge_0 = -1 * torch.ones(B, dtype=torch.long, device=tensor.device)
    changes_edge_1 = -1 * torch.ones(B, dtype=torch.long, device=tensor.device)
    batch_mask = changes == 1
    changes_edges = tensor[batch_mask].nonzero()[:, 1:]
    if not directed:
        changes_edges = changes_edges[changes_edges[:, 0] < changes_edges[:, 1]]
    changes_edge_0[batch_mask] = changes_edges[:, 0]
    changes_edge_1[batch_mask] = changes_edges[:, 1]
    return changes_edge_0, changes_edge_1


def find_single_edge_attr_change(edge_attr, diff_edge_attr, directed=False):
    B, N, _, _ = edge_attr.shape

    changes = diff_edge_attr.sum(dim=(-3, -2, -1))
    if not directed:
        changes = changes / 2
    batch_mask = changes == 1
    idx = diff_edge_attr[batch_mask].nonzero()

    if not directed:
        idx = idx[idx[:, 1] < idx[:, 2]]

    mod_edgeattr_edge0 = -1 * torch.ones(B, dtype=torch.long, device=edge_attr.device)
    mod_edge_0 = idx[:, 1]
    mod_edgeattr_edge0[batch_mask] = mod_edge_0
    mod_edgeattr_edge1 = -1 * torch.ones(B, dtype=torch.long, device=edge_attr.device)
    mode_edge_1 = idx[:, 2]
    mod_edgeattr_edge1[batch_mask] = mode_edge_1

    mod_edgeattr_attr = -1 * torch.ones(B, dtype=torch.long, device=edge_attr.device)
    mod_edge_attr = idx[:, 3]
    mod_edgeattr_attr[batch_mask] = mod_edge_attr

    mod_edgeattr_val = -1 * torch.ones(B, device=edge_attr.device, dtype=edge_attr.dtype)
    batch_idx = batch_mask.nonzero().squeeze()
    mod_edgeattr_val[batch_mask] = edge_attr[
        batch_idx, mod_edge_0, mode_edge_1, mod_edge_attr
    ]

    return mod_edgeattr_edge0, mod_edgeattr_edge1, mod_edgeattr_attr, mod_edgeattr_val


def eval_attacked_graphs(
    model,
    sparse_batch,
    directed=False,
    id=None,
    init_adj=None,  # (B, N, N)
    init_x=None,  # (B, N, D)
    init_edge_attr=None,  # (B, N, N, D)
    x=None,
    adj=None,
    edge_attr=None,
    mask=None,
    encode=True,
    compute_cycle_count=False,
    find_node_changes=True,
    find_edge_changes=True,
    transform=None,
):
    with torch.no_grad():
        results = {}

        model.eval()
        pred = model.predict(
            sparse_batch,
            encode=encode,
        )
        if len(pred.shape)==2 and pred.shape[-1]>1:
            pred = F.softmax(pred, dim=-1)
            for i in range(pred.shape[-1]):
                results[f"pred_{i}"] = pred[:,i]
        else:
            results["pred"] = pred.view(-1)
        results["y"] = sparse_batch.y.view(-1)

        if id is not None:
            results["id"] = id

        x, mask, adj, edge_attr = to_dense_data(
            sparse_batch, x=x, adj=adj, edge_attr=edge_attr, mask=mask
        )

        if init_x is not None:
            if len(x.shape) < 3:
                x = x.unsqueeze(-1)
            if len(init_x.shape) < 3:
                init_x = init_x.unsqueeze(-1)

            different_x = init_x != x
            results["node_attr_changes"] = different_x.sum(dim=(-2, -1))
            if find_node_changes:
                (
                    results["mod_x_node"],
                    results["mod_x_attr"],
                    results["mod_x_val"],
                ) = find_single_node_attr_change(x, different_x)

        factor = 1 if directed else 1 / 2
        if init_adj is not None:
            new_edges = adj > init_adj
            results["edges_added"] = (factor * new_edges.sum(dim=(-2, -1))).to(
                dtype=torch.int32
            )
            if find_edge_changes:
                results["add_edge_0"], results["add_edge_1"] = find_single_edge_change(
                    new_edges, directed=directed
                )

            removed_edges = adj < init_adj
            results["edges_removed"] = (factor * removed_edges.sum(dim=(-2, -1))).to(
                dtype=torch.int32
            )
            if find_edge_changes:
                (
                    results["drop_edge_0"],
                    results["drop_edge_1"],
                ) = find_single_edge_change(removed_edges, directed=directed)

            if (results["edges_added"] + results["edges_removed"] > 5).any():
                # to prevent long running time for computing many cycles
                compute_cycle_count = False

        if init_edge_attr is not None:
            if len(edge_attr.shape) < 4:
                edge_attr = edge_attr.unsqueeze(-1)
            if len(init_edge_attr.shape) < 4:
                init_edge_attr = init_edge_attr.unsqueeze(-1)

            in_both = (adj * init_adj).unsqueeze(-1)
            edge_attr_ = in_both * edge_attr
            init_edge_attr_ = in_both * init_edge_attr

            # diff_edge_attr = edge_attr != init_edge_attr
            diff_edge_attr = edge_attr_ != init_edge_attr_
            results["edge_attr_changes"] = (
                factor * diff_edge_attr.sum(dim=(-3, -2, -1))
            ).to(dtype=torch.int32)
            if find_edge_changes:
                (
                    results["mod_edgeattr_edge0"],
                    results["mod_edgeattr_edge1"],
                    results["mod_edgeattr_attr"],
                    results["mod_edgeattr_val"],
                ) = find_single_edge_attr_change(
                    edge_attr, diff_edge_attr, directed=directed
                )

        if init_x is not None and init_adj is not None and init_edge_attr is not None:
            results["abs_budget_"] = results["edges_added"] + results["edges_removed"] + results["node_attr_changes"] + results["edge_attr_changes"]

        results["num_nodes"], results["num_edges"] = count_graph_size(sparse_batch)
        results["num_edges"] = factor * results["num_edges"]
        results["min_degree"] = min_degree(adj, mask)
        results["max_degree"] = max_degree(adj, mask)
        results["connected"] = is_connected(adj, mask)
        if compute_cycle_count:
            results["num_cycles"], results["num_large_cycles"] = count_cycles(
                sparse_batch
            )
        else:
            results["num_cycles"] = -torch.ones_like(results["num_nodes"])
            results["num_large_cycles"] = -torch.ones_like(results["num_nodes"])
        results["connected_components"] = count_connected_components(sparse_batch)
        results["is_undirected"] = is_undirected(sparse_batch)

        for key, value in results.items():
            results[key] = value.detach().cpu().numpy()

    if transform is not None:
        transformed_batch = transform(sparse_batch)
        transformed_results = eval_attacked_graphs(
            model,
            transformed_batch,
            directed=directed,
            id=id,
            init_adj=init_adj,
            init_x=init_x,
            init_edge_attr=init_edge_attr,
            x=None,
            adj=None,
            edge_attr=None,
            mask=mask,
            encode=encode,
            compute_cycle_count=compute_cycle_count,
            find_node_changes=find_node_changes,
            find_edge_changes=find_edge_changes,
            transform=None,
        )

        n = len(transformed_results["y"])
        transformed_results["transformed"] = np.ones((n,))

        results["transformed"] = np.zeros((n,))

        for key, value in results.items():
            results[key] = np.concatenate((results[key], transformed_results[key]))

    return results
