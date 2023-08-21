import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from datasets import with_encoders
from datasets.metrics import with_metrics
from models.GINE import GINEConv


@with_metrics
@with_encoders
class GINEE(nn.Module):
    def __init__(
        self,
        num_outputs,
        create_bond_encoder,
        num_features=300,
        hidden_units=300,
        dropout=0.5,
        num_layers=5,
        aggr_type="add",  # neighborhood aggregation
        pool_type="mean",  # global pooling
    ):
        super(GINEE, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        assert aggr_type in ["add", "mean"]
        self.pool_type = pool_type
        assert pool_type in ["add", "mean"]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bond_encoders = nn.ModuleList()

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
        self.bond_encoders.append(create_bond_encoder())

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
            self.bond_encoders.append(create_bond_encoder())

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
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.convs[i](x, data.edge_index, edge_attr)
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
