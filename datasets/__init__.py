import pickle
from datasets.tudataset_util import TU_DATASETS, Degree_Encoder, imdb_encoders, mutag_encoders, split_tudataset_dataset
import torch
import numpy as np
from datasets.ogb_util import (
    load_ogb_dataset,
    ogb_mol_encoders,
    ppa_encoders,
    split_ogb_dataset,
)
from datasets.util import OGB_DATASETS
from datasets.zinc_util import split_zinc_dataset, zinc_encoders
from datasets.metrics import (
    BinaryClassifiactionEvaluator,
    MultiClassifiactionEvaluator,
    RegressionEvaluator,
)


def with_encoders(cls):
    old_init = cls.__init__

    def __init__(self, *args, atom_encoder=None, bond_encoder=None, **kwargs):
        old_init(self, *args, **kwargs)
        if not atom_encoder is None:
            self.atom_encoder = atom_encoder
        if not bond_encoder is None:
            self.bond_encoder = bond_encoder

        self.use_atom_embedding = not atom_encoder is None
        self.use_bond_embedding = not bond_encoder is None

    cls.__init__ = __init__

    old_forward = cls.forward

    def forward(self, data, *args, encode=True, x=None, adj=None, edge_attr=None, limit_edgeattr_by_adj=False, **kwargs):
        if encode:
            data = self.encode(data)

            if x is not None:
                if isinstance(self.atom_encoder, Degree_Encoder):
                    x = self.atom_encoder(data, adj, dense=True).squeeze()
                else:
                    x = self.atom_encoder(x).squeeze()
            if edge_attr is not None:
                edge_attr = self.bond_encoder(edge_attr).squeeze()
                while len(edge_attr.shape)<4:
                    edge_attr = edge_attr.unsqueeze(0)
                N = edge_attr.shape[1]
                edge_attr[:, range(N), range(N)] = 0

            if limit_edgeattr_by_adj:
                edge_attr = adj.unsqueeze(-1) * edge_attr

        return old_forward(self, data, *args, x=x, adj=adj, edge_attr=edge_attr, **kwargs)

    def encode(self, data):
        data = data.clone()
        if self.use_atom_embedding:
            if isinstance(self.atom_encoder, Degree_Encoder):
                data.x = self.atom_encoder(data, None, dense=False)
            else:
                data.x = self.atom_encoder(data.x)

        if self.use_bond_embedding:
            data.edge_attr = self.bond_encoder(data.edge_attr)
        return data

    cls.forward = forward
    cls.encode = encode
    return cls


def get_dataset_split(dataset_name, subset=0, root='data'):
    if dataset_name in OGB_DATASETS:
        dataset = load_ogb_dataset(dataset_name, root=root)
        small = dataset_name=="ogbg-molhiv-sm"
        return split_ogb_dataset(dataset, subset=subset, small=small)
    elif dataset_name.startswith("ZINC"):
        subset_12k = dataset_name.endswith("12k")
        return split_zinc_dataset(subset=subset, subset_12k=subset_12k, root=root)
    elif dataset_name in TU_DATASETS:
        return split_tudataset_dataset(dataset_name, root=root)
    else:
        raise ValueError(f"unknown dataset '{dataset_name}'")


def get_encoders(dataset, node_emb_dim, edge_emb_dim, seed):
    if dataset in ["ogbg-molhiv", "ogbg-molhiv-sm", "ogbg-molpcba"]:
        atom_encoder, bond_encoder = ogb_mol_encoders(node_emb_dim, edge_emb_dim, seed)
    elif dataset == "ogbg-ppa":
        atom_encoder, bond_encoder = ppa_encoders(node_emb_dim, edge_emb_dim, seed)
    elif dataset.startswith("ZINC"):
        atom_encoder, bond_encoder = zinc_encoders(node_emb_dim, edge_emb_dim, seed)
    elif dataset in TU_DATASETS:
        if dataset.startswith("IMDB"):
            atom_encoder, bond_encoder = imdb_encoders(node_emb_dim, edge_emb_dim, seed)
        elif dataset=="MUTAG":
            atom_encoder, bond_encoder = mutag_encoders(node_emb_dim, edge_emb_dim, seed)
    else:
        raise ValueError(f"unknown dataset '{dataset}'")
    return atom_encoder, bond_encoder


def get_evaluator(dataset):
    if dataset in ["ogbg-molhiv", "ogbg-molhiv-sm", "ogbg-molpcba", "IMDB-BINARY", "MUTAG"]:
        evaluator = BinaryClassifiactionEvaluator(dataset)
    elif dataset in ["ogbg-ppa", "IMDB-MULTI"]:
        evaluator = MultiClassifiactionEvaluator(dataset)
    elif dataset.startswith("ZINC"):
        evaluator = RegressionEvaluator()
    else:
        raise ValueError(f"unknown dataset '{dataset}'")
    return evaluator
