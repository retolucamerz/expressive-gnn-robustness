import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from models.GIN import DenseGINConv as PyDenseGINConv
from models.GINE import DenseGINEConv as PyDenseGINEConv


class DenseGINConv(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        model = DenseGINConv(1, 1)
        model.layer = PyDenseGINConv.from_sparse(
            sparse_model.layer, negative_slope=negative_slope
        )
        return model

    def __init__(self, in_dim, emb_dim):
        super(DenseGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = PyDenseGINConv(nn=mlp, train_eps=False)

    def forward(self, x, adj, mask, edge_attr):
        return self.layer(x, adj, mask)


class DenseGINEConv(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        model = DenseGINEConv(1, 1)
        model.layer = PyDenseGINEConv.from_sparse(
            sparse_model.layer, negative_slope=negative_slope
        )
        return model

    def __init__(self, in_dim, emb_dim):
        super(DenseGINEConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = PyDenseGINEConv(nn=mlp, train_eps=False)

    def forward(self, x, adj, mask, edge_attr):
        return self.layer(x, adj, edge_attr, mask)


class DenseGNN_node(torch.nn.Module):
    @staticmethod
    def from_sparse(sparse_model, negative_slope=0):
        model = DenseGNN_node(2, 1, 1, negative_slope=negative_slope)

        model.num_layer = sparse_model.num_layer
        model.drop_ratio = sparse_model.drop_ratio
        model.JK = sparse_model.JK
        model.residual = sparse_model.residual
        model.gnn_type = sparse_model.gnn_type

        model.convs = torch.nn.ModuleList()
        for sparse_conv in sparse_model.convs:
            model.convs.append(
                DenseGINConv.from_sparse(sparse_conv, negative_slope=negative_slope)
            )
        model.batch_norms = sparse_model.batch_norms

        return model

    def __init__(
        self,
        num_layer,
        in_dim,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
        num_random_features=0,
        negative_slope=0,
    ):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers

        """

        super(DenseGNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type

        self.negative_slope = negative_slope

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(
                    DenseGINConv(emb_dim if layer != 0 else in_dim, emb_dim)
                )
            else:
                raise ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, adj, mask, edge_attr):
        ### computing input node embedding
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], adj, mask, edge_attr)

            h[mask] = self.batch_norms[layer](h[mask])

            if self.gnn_type not in ["gin", "gcn"] or layer != self.num_layer - 1:
                h = F.leaky_relu(
                    h, negative_slope=self.negative_slope
                )  # remove last relu for ogb

            if self.drop_ratio > 0.0:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=2)

        return node_representation
