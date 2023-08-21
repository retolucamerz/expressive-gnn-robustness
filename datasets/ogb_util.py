import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from rdkit import Chem
from torch.nn import Embedding, Linear
from datasets.util import paired_edges_order
from torch_geometric.utils import degree

full_atom_feature_dims = get_atom_feature_dims()
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[...,i])

        return x_embedding

full_bond_feature_dims = get_bond_feature_dims()
class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[...,i])

        return bond_embedding   

def ogb_mol_encoders(atom_emb_dim, bond_emb_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    atom_encoder = AtomEncoder(atom_emb_dim)
    bond_encoder = BondEncoder(bond_emb_dim)
    return atom_encoder, bond_encoder


def ppa_encoders(node_emb_dim, edge_emb_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_encoder = Embedding(1, node_emb_dim)
    edge_encoder = Linear(7, edge_emb_dim)
    return node_encoder, edge_encoder


def load_ogb_dataset(dataset_name: str, root='data'):
    if dataset_name not in ["ogbg-molhiv", "ogbg-molhiv-sm", "ogbg-molpcba", "ogbg-ppa"]:
        raise ValueError(f"unknown dataset {dataset_name}")
    
    if dataset_name=="ogbg-molhiv-sm":
        dataset_name = "ogbg-molhiv"

    class PreTransform:
        id = 0

        def __call__(self, data):
            order = paired_edges_order(data.edge_index)
            data.edge_index = data.edge_index[:, order]
            data.edge_attr = data.edge_attr[order]
            data.id = self.id
            self.id += 1
            if dataset_name == "ogbg-ppa":
                data.x = torch.zeros(data.num_nodes, dtype=torch.int32)
            return data

    return PygGraphPropPredDataset(
        name=dataset_name, root=root, pre_transform=PreTransform()
    )


def split_ogb_dataset(dataset, subset=0, small=False):
    split_idx = dataset.get_idx_split()
    train_split = dataset[split_idx["train"]]
    valid_split = dataset[split_idx["valid"]]
    test_split = dataset[split_idx["test"]]

    if subset:
        n = int(len(train_split) / subset)
        train_split = train_split[:n]
        n = int(len(test_split) / subset)
        test_split = test_split[:n]
        n = int(len(valid_split) / subset)
        valid_split = valid_split[:n]

    if small:
        train_split = [graph for graph in train_split if graph.num_nodes<=100]
        valid_split = [graph for graph in valid_split if graph.num_nodes<=100]
        test_split = [graph for graph in test_split if graph.num_nodes<=100]

    return train_split, valid_split, test_split


def ogb_eval(
    model,
    loader,
    dataset_name: str,
    ncorrect,
    device: str,
    transform_pred=lambda x: x,
    compute_lbl_diff=lambda pred, y: (pred - y).abs().sum().item(),
    num_classes=1,
):
    """computes OGB score and accuracy in one go"""
    #     # adapted from https://github.com/snap-stanford/ogb/blob/dd9e3d0ed642619528e90d312bc8b52476338c64/examples/graphproppred/mol/main_pyg.py

    if dataset_name in ["ogbg-molhiv", "ogbg-molhiv-sm"]:
        eval_criterion = "rocauc"
    elif dataset_name == "ogbg-molpcba":
        eval_criterion = "ap"
    elif dataset_name == "ogbg-ppa":
        eval_criterion = "acc"
    else:
        raise ValueError(f"unknown dataset {dataset_name}")
    
    if dataset_name == "ogbg-molhiv-sm":
        dataset_name = "ogbg-molhiv"

    # accuracy
    correct = 0
    total = 0
    lbl_diff = 0

    # OGB score
    evaluator = Evaluator(dataset_name)
    y_true = []
    y_pred = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)

        pred = model.predict(batch)

        # OGB score
        y_true.append(batch.y.view(-1, 1).detach().cpu())
        y_pred.append(transform_pred(pred).view(-1, 1).detach().cpu())

        # accuracy
        pred = pred.squeeze()
        y = batch.y.squeeze()
        is_labeled = y == y
        pred = pred[is_labeled]
        y = y[is_labeled]
        lbl_diff += compute_lbl_diff(pred, y)
        correct += ncorrect(pred, y)
        total += len(y)

    lbl_diff = lbl_diff / num_classes

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return (
        evaluator.eval(input_dict)[eval_criterion],
        correct / total,
        lbl_diff / total,
    )
