import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_add_pool, global_mean_pool
from datasets import with_encoders
from datasets.metrics import with_metrics
from models.util import convert_to_leaky_relu, to_dense_data


from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)


class GINEConv(MessagePassing):
    r"""copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gin_conv.html#GINEConv and adapted to allow for mean aggragation"""

    def __init__(
        self,
        nn: torch.nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        aggr_type: str = "add",
        **kwargs,
    ):
        assert aggr_type in ["add", "mean"]
        kwargs["aggr"] = aggr_type
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, "in_features"):
                in_channels = nn.in_features
            elif hasattr(nn, "in_channels"):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'GINEConv'"
            )

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


@with_metrics
@with_encoders
class GINE(nn.Module):
    def __init__(
        self,
        num_outputs,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=5,
        aggr_type="add",  # neighborhood aggregation
        pool_type="mean",  # global pooling
    ):
        super(GINE, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        assert aggr_type in ["add", "mean"]
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            GINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                ),
                aggr_type=aggr_type,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_units))
        self.fcs.append(nn.Linear(num_features, num_outputs))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.ReLU(),
                        nn.Linear(hidden_units, hidden_units),
                    ),
                    aggr_type=aggr_type,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_units))
            self.fcs.append(nn.Linear(hidden_units, num_outputs))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINEConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, data.edge_index, data.edge_attr)
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


class DenseGINEConv(torch.nn.Module):
    r"""Based on `torch_geometric.nn.conv.DenseGINConv` (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/dense_gin_conv.html#DenseGINConv)"""

    @staticmethod
    def from_sparse(sparse_gine_conv, negative_slope=0):
        conv = DenseGINEConv(nn=None)
        conv.nn = convert_to_leaky_relu(sparse_gine_conv.nn, negative_slope)
        conv.eps = sparse_gine_conv.eps
        conv.initial_eps = sparse_gine_conv.initial_eps
        conv.aggr_type = sparse_gine_conv.aggr
        conv.negative_slope = negative_slope
        return conv

    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
        negative_slope: float = 0,
        aggr_type: str = "add",
    ):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

        self.aggr_type = aggr_type
        self.negative_slope = negative_slope

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Tensor,  # (B x N x D)
        adj: Tensor,  # (B x N x N)
        edge_attr: Tensor,  # (B x N x N x D)
        mask: Tensor,
        add_loop: bool = True,
    ):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        edge_attr = edge_attr.unsqueeze(0) if edge_attr.dim() == 3 else edge_attr
        B, N, _ = adj.size()

        # SUM ReLU(x_j + e_ji)
        z = x.unsqueeze(2).repeat_interleave(N, 2) + edge_attr
        out = F.leaky_relu(adj.unsqueeze(-1) * z, negative_slope=self.negative_slope)

        if self.aggr_type == "add":
            out = torch.sum(out, 1)
        elif self.aggr_type == "mean":
            degree = adj.sum(dim=-1)
            degree[degree == 0] = 1  # ok since isolated nodes have out[...] = 0
            out = torch.sum(out, 1) / degree.unsqueeze(-1)

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
class DenseGINE(nn.Module):
    @staticmethod
    def from_sparse(sparse_gine, negative_slope=0):
        atom_encoder = (
            sparse_gine.atom_encoder if sparse_gine.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_gine.bond_encoder if sparse_gine.use_bond_embedding else None
        )

        dense = DenseGINE(
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
            dense.convs.append(
                DenseGINEConv.from_sparse(conv, negative_slope=negative_slope)
            )

        return dense

    def __init__(
        self,
        num_outputs,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=5,
        pool_type="mean",
        aggr_type="add",
        negative_slope=0,
    ):
        super(DenseGINE, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        assert aggr_type in ["add", "mean"]
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]

        self.negative_slope = negative_slope

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(
            DenseGINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.LeakyReLU(negative_slope=self.negative_slope),
                    nn.Linear(hidden_units, hidden_units),
                ),
                aggr_type=aggr_type,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_units))
        self.fcs.append(nn.Linear(num_features, num_outputs))
        self.fcs.append(nn.Linear(hidden_units, num_outputs))

        for i in range(self.num_layers - 1):
            self.convs.append(
                DenseGINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        nn.BatchNorm1d(hidden_units),
                        nn.LeakyReLU(negative_slope=self.negative_slope),
                        nn.Linear(hidden_units, hidden_units),
                    ),
                    aggr_type=aggr_type,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_units))
            self.fcs.append(nn.Linear(hidden_units, num_outputs))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, DenseGINEConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()


    def forward(self, data, x=None, edge_attr=None, adj=None, mask=None):
        x, mask, adj, edge_attr = to_dense_data(data, x=x, adj=adj, edge_attr=edge_attr, mask=mask)

        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, adj, edge_attr, mask=mask)
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
                num_nodes = mask.sum(dim=-1).unsqueeze(-1)
                x = x.sum(1) / num_nodes
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        return out
