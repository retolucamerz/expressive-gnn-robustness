import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
)
from datasets import with_encoders
from datasets.metrics import with_metrics

from models.ESAN.sparse_data import policy2transform

from .sparse_conv import GNN_node


def subgraph_pool(h_node, batched_data, pool):
    subgraph_idx = batched_data.subgraph_batch

    return pool(h_node, subgraph_idx)


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layer=5,
        in_dim=300,
        emb_dim=300,
        gnn_type="gin",
        num_random_features=0,
        residual=False,
        drop_ratio=0.5,
        JK="last",
        graph_pooling="mean",
    ):
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = (
            self.emb_dim
            if self.JK == "last"
            else self.emb_dim * self.num_layer + in_dim
        )
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(
            num_layer,
            in_dim,
            emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            num_random_features=num_random_features,
        )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        return subgraph_pool(h_node, batched_data, self.pool)


@with_metrics
@with_encoders
class GNNComplete(GNN):
    def __init__(
        self,
        num_tasks,
        # subgraph selection
        policy,
        num_hops,
        sample_fraction,
        num_layer=5,
        in_dim=300,
        emb_dim=300,
        gnn_type="gin",
        num_random_features=0,
        residual=False,
        JK="last",
        graph_pooling="mean",
        enable_caching=True,
    ):
        super(GNNComplete, self).__init__(
            num_tasks,
            num_layer,
            in_dim,
            emb_dim,
            gnn_type,
            num_random_features,
            residual,
            JK,
            graph_pooling,
        )

        self.policy = policy
        self.num_hops = num_hops
        self.sample_fraction = sample_fraction
        self.transform = policy2transform(policy, num_hops, sample_fraction, enable_caching=enable_caching)

        if gnn_type == "graphconv":
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim),
                torch.nn.ELU(),
                torch.nn.Linear(
                    in_features=self.out_dim, out_features=self.out_dim // 2
                ),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim // 2, out_features=num_tasks),
            )
        else:
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=num_tasks),
            )

    def forward(self, batched_data):
        batched_data = self.transform(batched_data)
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.final_layers(h_graph)


@with_metrics
@with_encoders
class DSnetwork(torch.nn.Module):
    def __init__(
        self,
        subgraph_gnn,
        channels,
        num_tasks,
        invariant,  # subgraph selection
        policy,
        num_hops,
        sample_fraction,
        enable_caching=True,
    ):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        self.policy = policy
        self.num_hops = num_hops
        self.sample_fraction = sample_fraction
        self.transform = policy2transform(policy, num_hops, sample_fraction)
        self.enable_caching = enable_caching

        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(
                torch.nn.Linear(
                    in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                    out_features=channels[i],
                )
            )
            if self.invariant:
                fc_sum_list.append(
                    torch.nn.Linear(in_features=channels[i], out_features=channels[i])
                )
            else:
                fc_sum_list.append(
                    torch.nn.Linear(
                        in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                        out_features=channels[i],
                    )
                )

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=channels[-1], out_features=2 * channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * channels[-1], out_features=num_tasks),
        )

    def forward(self, batched_data, subgraph_indices=None, **kwargs):
        batched_data = self.transform(batched_data, subgraph_indices=subgraph_indices, enable_caching=self.enable_caching)

        h_subgraph = self.subgraph_gnn(batched_data)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(
                zip(self.fc_list, self.fc_sum_list)
            ):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(
                src=h_subgraph,
                index=batched_data.subgraph_idx_batch,
                dim=0,
                reduce="mean",
            )
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(
                zip(self.fc_list, self.fc_sum_list)
            ):
                x1 = fc(h_subgraph)
                x2 = fc_sum(
                    torch_scatter.scatter(
                        src=h_subgraph,
                        index=batched_data.subgraph_idx_batch,
                        dim=0,
                        reduce="mean",
                    )
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx_batch])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(
                src=h_subgraph,
                index=batched_data.subgraph_idx_batch,
                dim=0,
                reduce="mean",
            )

        return self.final_layers(h_graph)


@with_metrics
@with_encoders
class DSSnetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        num_tasks,
        GNNConv,
        # subgraph selection
        policy,
        num_hops,
        sample_fraction,
        enable_caching=True,
    ):
        super(DSSnetwork, self).__init__()

        self.policy = policy
        self.num_hops = num_hops
        self.sample_fraction = sample_fraction
        self.transform = policy2transform(policy, num_hops, sample_fraction)
        self.enable_caching = enable_caching

        self.emb_dim = emb_dim

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks),
        )

    def forward(self, batched_data, subgraph_indices=None, **kwargs):
        device = batched_data.x.device
        batched_data = self.transform(batched_data, subgraph_indices=subgraph_indices, enable_caching=self.enable_caching).to(device)

        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        for i in range(len(self.gnn_list)):
            gnn, bn, gnn_sum, bn_sum = (
                self.gnn_list[i],
                self.bn_list[i],
                self.gnn_sum_list[i],
                self.bn_sum_list[i],
            )

            h1 = bn(gnn(x, edge_index, edge_attr))

            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat(
                [
                    torch.zeros(
                        1,
                        device=device,
                        dtype=num_nodes_per_subgraph.dtype,
                    ),
                    torch.cumsum(num_nodes_per_subgraph, dim=0).view(-1),
                ]
            )
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx

            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            h2 = bn_sum(
                gnn_sum(
                    x_sum,
                    batched_data.original_edge_index,
                    batched_data.original_edge_attr
                    if edge_attr is not None
                    else edge_attr,
                )
            )

            x = F.relu(h1 + h2[node_idx])

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(
            src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean"
        )

        return self.final_layers(h_graph)
