import torch.nn as nn
from datasets import with_encoders
from datasets.metrics import with_metrics
from torch_geometric.nn import global_add_pool, global_mean_pool


@with_metrics
@with_encoders
class Baseline(nn.Module):
    # taken from https://github.com/KarolisMart/AgentNet/blob/main/graph_classification.py#L183
    def __init__(self, num_outputs, num_features=300, hidden_units=300):
        super(Baseline, self).__init__()

        self.in_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_outputs),
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data, **kwargs):
        x = self.in_mlp(data.x)
        x = global_mean_pool(x, data.batch)
        return self.out_mlp(x)
