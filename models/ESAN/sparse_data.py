import torch
import random
import math
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, k_hop_subgraph, subgraph
from multiprocessing.pool import ThreadPool as Pool


ORIG_EDGE_INDEX_KEY = "original_edge_index"


class SubgraphData(Data):
    def __inc__(self, key, value, store):
        # for correct mini-batching
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value)


class Graph2Subgraph:
    # need to apply this to every graph
    def __init__(self, process_subgraphs=lambda x: x, pbar=None):
        self.cache = {}

        self.process_subgraphs = process_subgraphs
        self.pbar = pbar

    def __call__(self, data, sample_fraction=1, sample_idx=None, enable_caching=True):
        if sample_idx is None:
            n_subgraphs = self.n_subgraphs(data)
            count = math.ceil(sample_fraction * n_subgraphs)
            sample_idx = random.sample(list(range(n_subgraphs)), count)

        subgraphs = {idx: None for idx in sample_idx}

        # fetch subgraphs from cache
        if enable_caching and hasattr(data, "id") and data.id.item() in self.cache:
            id = data.id.item()
            all_subgraphs = self.cache[id]
            for idx in sample_idx:
                if idx in all_subgraphs:
                    subgraphs[idx] = all_subgraphs[idx].clone().to(data.x.device)

        # compute remaining subgraphs
        for idx in subgraphs.keys():
            if subgraphs[idx] is None:
                subgraph = self.create_subgraph(data, idx).to(data.x.device)
                subgraphs[idx] = self.process_subgraphs(subgraph)

        # write back to cache
        if enable_caching:
            id = data.id.item()
            if not id in self.cache:
                self.cache[id] = {}
            for key, value in subgraphs.items():
                if key not in self.cache[id]:
                    self.cache[id][key] = value.detach().clone().to("cpu")

        subgraphs = [subgraphs[idx] for idx in sample_idx]
        batch = Batch.from_data_list(subgraphs)

        if self.pbar is not None:
            next(self.pbar)

        subgraph_data = SubgraphData(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            subgraph_batch=batch.batch,
            y=data.y,
            subgraph_idx=batch.subgraph_idx,
            subgraph_node_idx=batch.subgraph_node_idx,
            num_subgraphs=len(subgraphs),
            num_nodes_per_subgraph=data.num_nodes,
            original_edge_index=data.edge_index,
            original_edge_attr=data.edge_attr,
        )

        return subgraph_data

    def to_subgraphs(self, data):
        raise NotImplementedError
    
    def n_subgraphs(self, data):
        raise NotImplementedError
    
    def create_subgraph(self, data, subgraph_idx):
        raise NotImplementedError


class EdgeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        # remove one of the bidirectional index
        if data.edge_attr is not None and len(data.edge_attr.shape) == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = (
            data.edge_attr[keep_edge, :]
            if data.edge_attr is not None
            else data.edge_attr
        )

        subgraphs = []

        for i in range(edge_index.size(1)):
            subgraph_edge_index = torch.hstack(
                [edge_index[:, :i], edge_index[:, i + 1 :]]
            )
            subgraph_edge_attr = (
                torch.vstack([edge_attr[:i], edge_attr[i + 1 :]])
                if data.edge_attr is not None
                else data.edge_attr
            )

            if data.edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(
                    subgraph_edge_index, subgraph_edge_attr, num_nodes=data.num_nodes
                )
            else:
                subgraph_edge_index = to_undirected(
                    subgraph_edge_index, subgraph_edge_attr, num_nodes=data.num_nodes
                )

            subgraphs.append(
                Data(
                    x=data.x,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        if len(subgraphs) == 0:
            subgraphs = [
                Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    subgraph_idx=torch.tensor(0),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            ]
        return subgraphs


class NodeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        subgraphs = []
        all_nodes = torch.arange(data.num_nodes, device=data.x.device)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1 :]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(
                subset,
                data.edge_index,
                data.edge_attr,
                relabel_nodes=False,
                num_nodes=data.num_nodes,
            )

            subgraphs.append(
                Data(
                    x=data.x,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs
    
    def create_subgraph(self, data, subgraph_idx):
        all_nodes = torch.arange(data.num_nodes, device=data.x.device)

        subset = torch.cat([all_nodes[:subgraph_idx], all_nodes[subgraph_idx + 1 :]])
        subgraph_edge_index, subgraph_edge_attr = subgraph(
            subset,
            data.edge_index,
            data.edge_attr,
            relabel_nodes=False,
            num_nodes=data.num_nodes,
        )

        return Data(
            x=data.x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_idx=torch.tensor(subgraph_idx),
            subgraph_node_idx=torch.arange(data.num_nodes),
            num_nodes=data.num_nodes,
        )


class EgoNets(Graph2Subgraph):
    def __init__(
        self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None, **kwargs
    ):
        super().__init__(process_subgraphs=process_subgraphs, pbar=pbar, **kwargs)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def n_subgraphs(self, data):
        return data.num_nodes

    def to_subgraphs(self, data):
        subgraphs = []

        for i in range(data.num_nodes):
            subgraph.append(self.create_subgraph(data, i))
        return subgraphs

    def create_subgraph(self, data, subgraph_idx):
        _, _, _, edge_mask = k_hop_subgraph(
            subgraph_idx,
            self.num_hops,
            data.edge_index,
            relabel_nodes=False,
            num_nodes=data.num_nodes,
        )
        subgraph_edge_index = data.edge_index[:, edge_mask]
        subgraph_edge_attr = (
            data.edge_attr[edge_mask]
            if data.edge_attr is not None
            else data.edge_attr
        )

        x = data.x
        if self.add_node_idx:
            # prepend a feature [0, 1] for all non-central nodes
            # a feature [1, 0] for the central node
            ids = torch.arange(2, device=x.device).repeat(data.num_nodes, 1)
            ids[subgraph_idx] = torch.tensor([ids[subgraph_idx, 1], ids[subgraph_idx, 0]])

            # change from original code: concat additional features to the already encoded x
            x = torch.cat((x, ids), dim=1)

        return Data(
            x=x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_idx=torch.tensor(subgraph_idx),
            subgraph_node_idx=torch.arange(data.num_nodes),
            num_nodes=data.num_nodes,
        )


def unbatch_subgraphs(data):
    device = data.x.device
    subgraphs = []
    num_nodes = data.num_nodes_per_subgraph
    for i in range(data.num_subgraphs):
        edge_index, edge_attr = subgraph(
            torch.arange(num_nodes, device=device) + (i * num_nodes),
            data.edge_index,
            data.edge_attr,
            relabel_nodes=False,
            num_nodes=data.x.size(0),
        )
        subgraphs.append(
            Data(
                x=data.x[i * num_nodes : (i + 1) * num_nodes, :],
                edge_index=edge_index - (i * num_nodes),
                edge_attr=edge_attr,
                subgraph_idx=torch.tensor(0),
                subgraph_node_idx=torch.arange(num_nodes),
                num_nodes=num_nodes,
            )
        )

    original_edge_attr = (
        data.original_edge_attr if data.edge_attr is not None else data.edge_attr
    )
    return Data(
        x=subgraphs[0].x,
        edge_index=data.original_edge_index,
        edge_attr=original_edge_attr,
        y=data.y,
        subgraphs=subgraphs,
    )


def policy2transform(
    policy: str, num_hops, sample_fraction, process_subgraphs=lambda x: x, pbar=None, enable_caching=True
):
    if policy == "edge_deleted":
        subgraph_selection = EdgeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "node_deleted":
        subgraph_selection = NodeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets":
        subgraph_selection = EgoNets(
            num_hops, process_subgraphs=process_subgraphs, pbar=pbar
        )
    elif policy == "ego_nets_plus":
        subgraph_selection = EgoNets(
            num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar
        )
    elif policy == "original":
        subgraph_selection = process_subgraphs
    else:
        raise ValueError("Invalid subgraph policy type")

    def transform(batch, subgraph_indices=None, enable_caching=True):
        batch_size = len(batch.y)
        transformed_batch = []
        for i in range(batch_size):
            subgraphs = subgraph_selection(batch[i], sample_fraction=sample_fraction, sample_idx=subgraph_indices, enable_caching=enable_caching)
            transformed_batch.append(subgraphs)

        return Batch.from_data_list(transformed_batch, follow_batch=["subgraph_idx"])

    return transform
