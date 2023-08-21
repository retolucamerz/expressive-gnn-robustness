import torch
import numpy as np
from torch.nn import Embedding
from torch_geometric.datasets.zinc import ZINC
from datasets.util import paired_edges_order


def zinc_encoders(atom_emb_dim, bond_emb_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    atom_encoder = Embedding(28, atom_emb_dim)
    bond_encoder = Embedding(4, bond_emb_dim)
    return atom_encoder, bond_encoder


def split_zinc_dataset(subset=0, subset_12k=False, root="data"):
    class PreTransform:
        id = 0

        def __call__(self, data):
            order = paired_edges_order(data.edge_index)
            data.edge_index = data.edge_index[:, order]
            data.edge_attr = data.edge_attr[order]
            data.id = self.id
            self.id += 1
            return data

    class Transform:
        def __call__(self, data):
            data.x = data.x.squeeze()
            return data

    train_split = ZINC(
        root=root,
        split="train",
        pre_transform=PreTransform(),
        transform=Transform(),
        subset=subset_12k,
    )
    valid_split = ZINC(
        root=root,
        split="val",
        pre_transform=PreTransform(),
        transform=Transform(),
        subset=subset_12k,
    )
    test_split = ZINC(
        root=root,
        split="test",
        pre_transform=PreTransform(),
        transform=Transform(),
        subset=subset_12k,
    )

    if subset:
        n = int(len(train_split) / subset)
        train_split = train_split[:n]
        n = int(len(valid_split) / subset)
        valid_split = valid_split[:n]
        n = int(len(test_split) / subset)
        test_split = test_split[:n]

    return train_split, valid_split, test_split
