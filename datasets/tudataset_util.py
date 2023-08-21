from datasets.util import paired_edges_order
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch.nn import Module, Embedding
import torch.nn.functional as F
from math import ceil
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import to_dense_adj, to_dense_batch


class Degree_Encoder(Module):
    def __init__(self, emb_dim, max_node_deg=25):
        super(Degree_Encoder, self).__init__()
        self.embedding = Embedding(max_node_deg+1, emb_dim)
        self.max_node_deg = max_node_deg

    def forward(self, data, adj, dense=False):
        if data is None and adj is None:
            raise ValueError("cannot do Degree_Encoder")
        if adj is None:
            adj = to_dense_adj(data.edge_index, data.batch)
        deg = adj.sum(dim=-1)
        max_deg_tensor = torch.tensor(self.max_node_deg).expand(*deg.shape).to(adj.device)
        int_deg = torch.minimum(torch.round(deg.detach()), max_deg_tensor).to(dtype=torch.int64)
        onehot_deg = F.one_hot(int_deg, num_classes=self.max_node_deg+1).to(dtype=torch.float32)
        onehot_deg[onehot_deg == 1] = (1 + deg - deg.detach()).view(-1) # attach gradient

        out = onehot_deg @ self.embedding.weight
        if not dense:
            _, mask = to_dense_batch(data.x, data.batch)
            out = out[mask].view(-1, out.shape[-1])

        return out


TU_DATASETS = ["IMDB-MULTI", "IMDB-BINARY", "MUTAG"]

def imdb_encoders(atom_emb_dim, bond_emb_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # atom_encoder = Embedding(1, atom_emb_dim)
    atom_encoder = Degree_Encoder(atom_emb_dim)
    bond_encoder = Embedding(1, bond_emb_dim)
    return atom_encoder, bond_encoder

def mutag_encoders(atom_emb_dim, bond_emb_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    atom_encoder = Embedding(7, atom_emb_dim)
    bond_encoder = Embedding(4, bond_emb_dim)
    return atom_encoder, bond_encoder


def split_tudataset_dataset(name, subset=0, root="data"):
    class PreTransform:
        id = 0

        def __call__(self, data):
            order = paired_edges_order(data.edge_index)
            data.edge_index = data.edge_index[:, order]

            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[order]
            else:
                data.edge_attr = torch.zeros(data.num_edges, dtype=torch.int32)

            if not hasattr(data, "x") or data.x is None:
                data.x = torch.zeros(data.num_nodes, dtype=torch.int32)

            if name=="MUTAG":
                assert (data.x.sum(dim=-1)==1).all()
                assert (data.edge_attr.sum(dim=-1)==1).all()
                data.x = data.x.nonzero()[:,1]
                data.edge_attr = data.edge_attr.nonzero()[:,1]

            data.id = self.id
            self.id += 1

            return data

    dataset = TUDataset(
        name=name,
        root=root,
        use_node_attr=True,
        use_edge_attr=True,
        pre_transform=PreTransform(),
    )

    # split first into train / test data
    dataset_len = len(dataset)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    train_idx_, test_idx = next(skf.split(np.zeros(dataset_len), np.zeros(dataset_len)))

    # split train data in train / valid
    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=0)
    __train_idx, __valid_idx = next(skf.split(np.zeros(len(train_idx_)), np.zeros(len(train_idx_))))
    train_idx = train_idx_[__train_idx]
    valid_idx = train_idx_[__valid_idx]

    train_split = dataset[train_idx]
    valid_split = dataset[valid_idx]
    test_split = dataset[test_idx]

    if subset:
        n = int(len(train_split) / subset)
        train_split = train_split[:n]
        n = int(len(valid_split) / subset)
        valid_split = valid_split[:n]
        n = int(len(test_split) / subset)
        test_split = test_split[:n]

    return train_split, valid_split, test_split
