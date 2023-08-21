import itertools
from models.util import to_dense_data
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from attacks.util import BaseTask, update_graph
import pandas as pd
from tqdm import tqdm
from attacks.evaluate import (
    eval_attacked_graphs,
)


def find_max_target_graph(model, loader, target, device, init_x=None, init_adj=None, init_edge_attr=None, **kwargs):
    with torch.no_grad():
        best_graph = None
        best_val = float("-inf")

        for batch in loader:
            batch = batch.to(device)
            val = target(model, batch, encode=True)

            idx = torch.argmax(val)
            if val[idx] > best_val:
                best_val = val[idx]
                best_graph = batch[idx]

        # ugly hack to convert best_graph to a batch
        for best_batch in DataLoader([best_graph]):
            break

        return eval_attacked_graphs(
            model,
            best_batch,
            init_x=init_x,
            init_adj=init_adj,
            init_edge_attr=init_edge_attr,
            encode=True,
            id=best_graph.id,
            **kwargs,
        )


def drop_edge(graph, k, directed=False):
    """drops the `k`th edge of graph
    ASSUMES THAT THE TWO DIRECTED EDGES REPRESENTING A SINGLE UNDIRECTED EDGE
    ARE NEXT TO EACH OTHER IN EDGE TENSORS
    """
    graph = graph.clone()
    mask = torch.ones(graph.num_edges, dtype=torch.bool)
    if directed:
        mask[k] = False
    else:
        mask[2 * k] = False
        mask[2 * k + 1] = False
    graph.edge_index = graph.edge_index[:, mask]
    graph.edge_attr = graph.edge_attr[mask]
    return graph


def unfold_values(attr_dict, feat):
    for key, values in attr_dict.items():
        if not isinstance(values, list):
            values = values(feat)

        for value in values:
            feat_mod = feat.clone().view(-1)
            if feat_mod[key] == value: continue
            feat_mod[key] = value
            yield feat_mod.squeeze()


def single_edge_additions(data, edge_features, directed=False, allow_self_loops=False):
    edge_attr_values = [torch.tensor(x) for x in itertools.product(*edge_features.values())]

    # determine which edges are not present
    n = data.num_nodes
    if directed:
        full_connectivity = {(x, y) for x in range(n) for y in range(n)}
    else:
        # assert data.is_undirected()
        full_connectivity = {(x, y) for x in range(n) for y in range(x)}
    present_edges = set(map(tuple, data.edge_index.transpose(0, 1).tolist()))
    missing_edges = full_connectivity - present_edges
    if not allow_self_loops:
        missing_edges = filter(lambda t: t[0] != t[1], missing_edges)
    missing_edges = list(missing_edges)

    device = data.edge_index.device

    # iterate over all perturbations
    for missing_edge in missing_edges:
        edge_index = data.edge_index.clone()
        other_edge = (missing_edge[1], missing_edge[0])
        edge_index = torch.cat(
            (
                edge_index,
                torch.tensor(missing_edge).unsqueeze(-1).to(device),
                torch.tensor(other_edge).unsqueeze(-1).to(device),
            ),
            dim=-1,
        )

        for missing_edge_attr in edge_attr_values:
            edge_attr = data.edge_attr.clone()
            missing_edge_attr = missing_edge_attr.to(device)
            if len(missing_edge_attr.shape) < len(edge_attr.shape):
                missing_edge_attr = missing_edge_attr.unsqueeze(0)
            edge_attr = torch.cat((edge_attr, missing_edge_attr, missing_edge_attr))

            yield update_graph(data, edge_index=edge_index, edge_attr=edge_attr, clone=False)


def single_node_attribute_mod(data, node_features):
    for k in range(data.num_nodes):
        feat = data.x[k]
        for missing_node_attr in unfold_values(node_features, feat):
            x = data.x.clone()
            x[k] = missing_node_attr
            yield update_graph(data, x=x, clone=False)


def single_edge_attribute_mod(data, edge_features, directed=False):
    m = data.num_edges if directed else int(data.num_edges / 2)

    for k in range(m):
        feat = data.edge_attr[2 * k] if not directed else data.edge_attr[k]
        for missing_edge_attr in unfold_values(edge_features, feat):
            edge_attr = data.edge_attr.clone()
            if directed:
                edge_attr[k] = missing_edge_attr
            else:
                edge_attr[2 * k] = missing_edge_attr
                edge_attr[2 * k + 1] = missing_edge_attr
            yield update_graph(data, edge_attr=edge_attr, clone=False)


class BruteforceAttackTask(BaseTask):
    params = [None]

    def __init__(
        self,
        target,
        perturbations,
        data_selection=None,
        force_update=False,
        transform_df=None,
        transform_batch=None,
        compute_cycle_count=False,
    ):
        super(BruteforceAttackTask, self).__init__(
            data_selection=data_selection,
            force_update=force_update,
            transform_df=transform_df,
            transform_batch=transform_batch,
        )
        self.target = target
        self.perturbations = perturbations
        self.compute_cycle_count = compute_cycle_count

    def set_data(self, model, param, recorder, dataset, device, args):
        for graph in tqdm(dataset):
            graph = graph.to(device)
            model.eval()

            init_x, mask, init_adj, init_edge_attr = to_dense_data(graph, max_num_nodes=graph.num_nodes)

            for pert_name, pert_func in self.perturbations.items():
                loader = DataLoader(pert_func(graph), batch_size=2*args.batch_size)
                if len(loader)==0:
                    loader = DataLoader([graph], batch_size=args.batch_size)

                graph_results = find_max_target_graph(
                    model,
                    loader,
                    self.target,
                    device,
                    init_x=init_x,
                    init_adj=init_adj,
                    init_edge_attr=init_edge_attr,
                    directed=False,
                    transform=self.transform_batch,
                    compute_cycle_count=self.compute_cycle_count,
                )
                recorder.record(perturbation=pert_name, **graph_results)

    def exists(self, data, param):
        if data.empty:
            return False

        perturbations_exist = (
            (data["perturbation"] == pert_name).any()
            for pert_name in self.perturbations.keys()
        )

        return all(perturbations_exist)


def BF_PERTURBATIONS(
    node_attr_dict, edge_attr_dict, edge_attr_dict_fixed=None, with_node_attr=True, with_edge_attr=True
):
    if edge_attr_dict_fixed is None:
        edge_attr_dict_fixed = edge_attr_dict

    ret = {
        "drop-edge": lambda graph: [
            drop_edge(graph, k, directed=False) for k in range(int(graph.num_edges / 2))
        ],
        "add-edge": lambda graph: list(single_edge_additions(graph, edge_attr_dict)),
    }
    if with_node_attr:
        ret["modify-node-attr"] = lambda graph: list(
            single_node_attribute_mod(graph, node_attr_dict)
        )
    if with_edge_attr:
        ret["modify-edge-attr"] = lambda graph: list(
            single_edge_attribute_mod(graph, edge_attr_dict)
        )
    return ret
