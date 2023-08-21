import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.utils.dropout import dropout_node
from models.GIN import DenseGINConv
from datasets import with_encoders
from datasets.metrics import with_metrics
from models.util import to_dense_data



@with_metrics
@with_encoders
class DropGIN(nn.Module):
    def __init__(
        self,
        num_outputs,
        num_runs,
        p,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=4,
        pool_type="mean",
        aggr_type="add",
    ):
        super(DropGIN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_runs = num_runs
        self.p = p
        assert aggr_type in ["add", "mean"]
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
                ),
                aggr_type=aggr_type,
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
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data, drop=None, **kwargs):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Do runs in paralel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
        if drop is None:
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.num_runs, -1, x.size(-1)))
        del  run_edge_index
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
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


@with_metrics
@with_encoders
class DenseDropGIN(nn.Module):
    @staticmethod
    def from_sparse(sparse_gin, negative_slope=0, **kwargs):
        atom_encoder = (
            sparse_gin.atom_encoder if sparse_gin.use_atom_embedding else None
        )
        bond_encoder = (
            sparse_gin.bond_encoder if sparse_gin.use_bond_embedding else None
        )

        dense = DenseDropGIN(
            num_outputs=1,
            num_runs=1,
            p=0,
            num_features=1,
            hidden_units=1,
            dropout=0,
            num_layers=1,
            evaluator=sparse_gin.evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
            negative_slope=negative_slope,
        )
        dense.p = sparse_gin.p
        dense.num_runs = sparse_gin.num_runs
        dense.dropout = sparse_gin.dropout
        dense.num_layers = sparse_gin.num_layers

        dense.bns = sparse_gin.bns
        dense.fcs = sparse_gin.fcs
        dense.convs = nn.ModuleList()
        for conv in sparse_gin.convs:
            dense.convs.append(
                DenseGINConv.from_sparse(conv, negative_slope=negative_slope, **kwargs)
            )

        return dense

    def __init__(
        self,
        num_outputs,
        num_runs,
        p,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=4,
        pool_type="mean",
        # aggr_type="add",
        negative_slope=0,
    ):
        super(DenseDropGIN, self).__init__()

        self.p = p
        self.num_runs = num_runs
        self.dropout = dropout
        self.num_layers = num_layers
        # assert aggr_type in ["add", "mean"]
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
                ),
                # aggr_type=aggr_type,
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
                    ),
                    # aggr_type=aggr_type,
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

    def forward(self, data, x=None, edge_attr=None, adj=None, mask=None, drop=None):
        x, mask, adj, edge_attr = to_dense_data(data, x=x, adj=adj, edge_attr=edge_attr, mask=mask)

        # create mask for dropped nodes
        if drop is None:
            sparse_size = mask.sum().long().item()
            drop = torch.bernoulli(
                torch.ones([self.num_runs, sparse_size], device=x.device) * self.p
            ).bool()
        drop_mask = torch.zeros(
            (self.num_runs, *mask.size()), dtype=torch.bool, device=x.device
        )
        drop_mask[:, mask] = drop

        # remove dropped node (features) and adjacent edges
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1, -1).clone()
        x[drop_mask] = torch.zeros(
            [drop.sum().long().item(), x.size(-1)], device=x.device
        )
        adj = adj.unsqueeze(0).expand(self.num_runs, -1, -1, -1).clone()
        del drop
        del drop_mask

        conv_mask = mask.repeat(self.num_runs, 1)
        bn_mask = mask.unsqueeze(0).expand(self.num_runs, -1, -1)

        outs = [x]
        for i in range(self.num_layers):
            x_v = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
            adj_v = adj.view(adj.shape[0] * adj.shape[1], *adj.shape[2:])
            x = self.convs[i](x_v, adj_v, mask=conv_mask).view(
                x.shape[0], x.shape[1], x.shape[2], -1
            )
            x[bn_mask] = self.bns[i](x[bn_mask])
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            outs.append(x)

        x_shape = outs[-1].shape

        out = None
        for i, x in enumerate(outs):
            x = x.view(x_shape) # reading from iterator somehow squeezes x
            x = x.mean(dim=0)  # average over runs
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
