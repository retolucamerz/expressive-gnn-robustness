import torch
import torch.nn.functional as F
import torch_scatter
from datasets import with_encoders
from datasets.metrics import with_metrics
from models.ESAN.dense_data import densepolicy2transform
from models.ESAN.sparse_data import policy2transform
from models.ESAN.dense_conv import DenseGINConv, DenseGINEConv, DenseGNN_node
from models.ESAN.sparse_conv import GINConv, GINEConv
from models.util import to_dense_data


def dense_pool(x, num_nodes=None, pool_type="add"):
    if pool_type == "add":
        return x.sum(1)
    elif pool_type == "mean":
        if num_nodes is None:
            raise ValueError(f"need # of nodes per subgraph for mean pooling")
        return x.sum(1) / num_nodes.unsqueeze(-1)
    else:
        raise ValueError(f"pooling type '{pool_type}' not implemented")


class DenseGNN(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        model = DenseGNN(
            1,
            num_layer=1,
            in_dim=1,
            emb_dim=1,
        )

        model.num_layer = sparse_model.num_layer
        model.drop_ratio = sparse_model.drop_ratio
        model.JK = sparse_model.JK
        model.emb_dim = sparse_model.emb_dim
        model.out_dim = sparse_model.out_dim
        model.num_tasks = sparse_model.num_tasks
        model.graph_pooling = sparse_model.graph_pooling
        model.gnn_node = DenseGNN_node.from_sparse(model, negative_slope=negative_slope)
        return model

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
        super(DenseGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.out_dim = (
            emb_dim if self.JK == "last" else emb_dim * self.num_layer + in_dim
        )
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = DenseGNN_node(
            num_layer,
            in_dim,
            emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            num_random_features=num_random_features,
        )

    def forward(self, batched_data, x, adj, edge_attr):
        h_node = self.gnn_node(x, adj, batched_data.mask, edge_attr)
        num_nodes = batched_data.mask.sum(dim=-1)
        return dense_pool(h_node, num_nodes=num_nodes, pool_type=self.graph_pooling)


@with_metrics
@with_encoders
class DenseDSnetwork(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        atom_encoder = (
            sparse_model.atom_encoder if sparse_model.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_model.bond_encoder if sparse_model.use_bond_embedding else None
        )
        model = DenseDSnetwork(
            None,
            1,
            1,
            False,
            "node_deleted",
            1,
            1,
            evaluator=sparse_model.evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )
        model.subgraph_gnn = DenseGNN.from_sparse(
            sparse_model.subgraph_gnn, negative_slope=negative_slope
        )
        model.transform = densepolicy2transform(
            sparse_model.policy, sparse_model.num_hops, sparse_model.sample_fraction
        )
        model.invariant = sparse_model.invariant
        model.fc_list = sparse_model.fc_list
        model.fc_sum_list = sparse_model.fc_sum_list
        model.final_layers = sparse_model.final_layers
        return model

    def __init__(
        self,
        subgraph_gnn,
        channels,
        num_tasks,
        invariant,  # subgraph selection
        policy,
        num_hops,
        sample_fraction,
    ):
        super(DenseDSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        self.transform = densepolicy2transform(policy, num_hops, sample_fraction)

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

    def forward(self, batched_data, x=None, adj=None, edge_attr=None, mask=None, subgraph_indices=None):
        x, mask, adj, edge_attr = to_dense_data(
            batched_data, x=x, adj=adj, edge_attr=edge_attr, mask=mask
        )
        batched_data = self.transform(x, adj, edge_attr, mask, batched_data.id, subgraph_indices=subgraph_indices)
        x = batched_data.x
        mask = batched_data.mask
        adj = batched_data.adj
        edge_attr = batched_data.edge_attr

        h_subgraph = self.subgraph_gnn(batched_data, x, adj, edge_attr)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(
                zip(self.fc_list, self.fc_sum_list)
            ):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(
                src=h_subgraph,
                index=batched_data.subgraph_idx,
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
                        index=batched_data.subgraph_idx,
                        dim=0,
                        reduce="mean",
                    )
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(
                src=h_subgraph,
                index=batched_data.subgraph_idx,
                dim=0,
                reduce="mean",
            )

        return self.final_layers(h_graph)


def conv_to_dense(sparse_conv, negative_slope=0):
    if isinstance(sparse_conv, GINConv):
        return DenseGINConv.from_sparse(sparse_conv, negative_slope=negative_slope)
    if isinstance(sparse_conv, GINEConv):
        return DenseGINEConv.from_sparse(sparse_conv, negative_slope=negative_slope)
    else:
        raise ValueError(f"no dense version of '{type(sparse_conv)}' available")


@with_metrics
@with_encoders
class DenseDSSnetwork(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        atom_encoder = (
            sparse_model.atom_encoder if sparse_model.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_model.bond_encoder if sparse_model.use_bond_embedding else None
        )

        module = lambda x, y: torch.nn.Linear(1, 1)
        model = DenseDSSnetwork(
            1,
            1,
            1,
            1,
            module,
            "node_deleted",
            1,
            1,
            evaluator=sparse_model.evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )
        model.transform = densepolicy2transform(
            sparse_model.policy, sparse_model.num_hops, sparse_model.sample_fraction
        )

        model.gnn_list = torch.nn.ModuleList(
            [conv_to_dense(conv, negative_slope) for conv in sparse_model.gnn_list]
        )
        model.gnn_sum_list = torch.nn.ModuleList(
            [conv_to_dense(conv, negative_slope) for conv in sparse_model.gnn_sum_list]
        )

        model.bn_list = sparse_model.bn_list
        model.bn_sum_list = sparse_model.bn_sum_list
        model.final_layers = sparse_model.final_layers
        model.negative_slope = negative_slope

        return model

    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        num_tasks,
        DenseGNNConv,
        # subgraph selection
        policy,
        num_hops,
        sample_fraction,
        negative_slope=0,
    ):
        super(DenseDSSnetwork, self).__init__()

        self.transform = policy2transform(policy, num_hops, sample_fraction)

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(DenseGNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(DenseGNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.LeakyReLU(negative_slope),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks),
        )

        self.negative_slope = negative_slope

    def forward(self, batched_data, x=None, adj=None, edge_attr=None, mask=None, subgraph_indices=None):
        x, mask, adj, edge_attr = to_dense_data(
            batched_data, x=x, adj=adj, edge_attr=edge_attr, mask=mask
        )
        if len(x.shape)<3:
            x = x.unsqueeze(0)
        if len(edge_attr.shape)<4:
            edge_attr = edge_attr.unsqueeze(0)
        
        batched_data = self.transform(x, adj, edge_attr, mask, batched_data.id, subgraph_indices=subgraph_indices)
        x = batched_data.x
        mask = batched_data.mask
        adj = batched_data.adj
        edge_attr = batched_data.edge_attr

        for i in range(len(self.gnn_list)):
            gnn, bn, gnn_sum, bn_sum = (
                self.gnn_list[i],
                self.bn_list[i],
                self.gnn_sum_list[i],
                self.bn_sum_list[i],
            )

            h1 = gnn(x, adj, mask, edge_attr)
            h1[mask] = bn(h1[mask])

            x_sum = torch_scatter.scatter(
                src=x, index=batched_data.graph_idx, dim=0, reduce="mean"
            )

            h2 = gnn_sum(
                x_sum,
                batched_data.original_adj,
                batched_data.original_mask,
                batched_data.original_edge_attr,
            )
            h2[batched_data.original_mask] = bn_sum(h2[batched_data.original_mask])

            x = F.leaky_relu(
                h1 + h2[batched_data.graph_idx], negative_slope=self.negative_slope
            )

        num_nodes = mask.sum(dim=-1)
        h_subgraph = dense_pool(x, num_nodes=num_nodes, pool_type="mean")
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        h_graph = torch_scatter.scatter(
            src=h_subgraph, index=batched_data.graph_idx, dim=0, reduce="mean"
        )

        return self.final_layers(h_graph)
