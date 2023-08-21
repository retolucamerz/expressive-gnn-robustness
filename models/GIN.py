import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

# from torch_geometric.nn.dense import DenseGINConv
from models.util import convert_to_leaky_relu, to_dense_data
from datasets import with_encoders
from datasets.metrics import with_metrics


@with_metrics
@with_encoders
class GIN(nn.Module):
    def __init__(
        self,
        num_outputs,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=5,
        pool_type="mean",
    ):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                )
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_units))
        self.fcs.append(nn.Linear(num_features, num_outputs))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.ReLU(),
                        nn.Linear(hidden_units, hidden_units),
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_units))
            self.fcs.append(nn.Linear(hidden_units, num_outputs))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, data.edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out = None
        for i, x in enumerate(outs):
            if self.pool_type == "add":
                x = global_add_pool(x, data.batch)
            elif self.pool_type == "mean":
                x = global_mean_pool(x, data.batch)

            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        return out


class DenseGINConv(torch.nn.Module):
    """copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/dense_gin_conv.html#DenseGINConv,
    adjusted to apply `self.nn` to a masked view of `x` which is equivalent to the data in the sparse version
    """

    @staticmethod
    def from_sparse(sparse_gine_conv, negative_slope=0):
        conv = DenseGINConv(nn=None)
        conv.nn = convert_to_leaky_relu(sparse_gine_conv.nn, negative_slope)
        conv.eps = sparse_gine_conv.eps
        conv.initial_eps = sparse_gine_conv.initial_eps
        return conv

    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self, x: Tensor, adj: Tensor, mask: Tensor, add_loop: bool = True
    ) -> Tensor:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        nn_out = self.nn(out[mask])
        out = torch.zeros((*out.shape[:2], *nn_out.shape[1:]), device=x.device)
        out[mask] = nn_out
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


@with_metrics
@with_encoders
class DenseGIN(nn.Module):
    @staticmethod
    def from_sparse(sparse_gine, negative_slope=0):
        atom_encoder = (
            sparse_gine.atom_encoder if sparse_gine.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_gine.bond_encoder if sparse_gine.use_bond_embedding else None
        )

        dense = DenseGIN(
            num_outputs=1,
            num_features=1,
            hidden_units=1,
            dropout=0,
            num_layers=1,
            evaluator=sparse_gine.evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
            negative_slope=negative_slope,
        )
        dense.dropout = sparse_gine.dropout
        dense.num_layers = sparse_gine.num_layers
        dense.pool_type = sparse_gine.pool_type

        dense.bns = sparse_gine.bns
        dense.fcs = sparse_gine.fcs
        dense.convs = nn.ModuleList()
        for conv in sparse_gine.convs:
            dense.convs.append(DenseGINConv.from_sparse(conv, negative_slope))

        return dense

    def __init__(
        self,
        num_outputs,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=5,
        pool_type="mean",
        negative_slope=0,
    ):
        super(DenseGIN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]
        self.negative_slope = negative_slope

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            DenseGINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Linear(hidden_units, hidden_units),
                )
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_units))
        self.fcs.append(nn.Linear(num_features, num_outputs))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))

        for i in range(self.num_layers - 1):
            self.convs.append(
                DenseGINConv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.LeakyReLU(negative_slope=self.negative_slope),
                        nn.Linear(hidden_units, hidden_units),
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_units))
            self.fcs.append(nn.Linear(hidden_units, num_outputs))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, DenseGINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()


    def forward(self, data, x=None, edge_attr=None, adj=None, mask=None):
        x, mask, adj, edge_attr = to_dense_data(data, x=x, adj=adj, edge_attr=edge_attr, mask=mask)

        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, adj, mask=mask)
            x[mask] = self.bns[i](x[mask])
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            outs.append(x)

        x_shape = outs[-1].shape

        out = None
        for i, x in enumerate(outs):
            x = x.view(x_shape) # reading from iterator somehow squeezes x
            if self.pool_type == "add":
                x = x.sum(1)
            elif self.pool_type == "mean":
                num_nodes = mask.sum(dim=-1)
                x = x.sum(1) / num_nodes.unsqueeze(-1)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        return out
