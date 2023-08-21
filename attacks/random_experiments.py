import random
from attacks.bruteforce_attacks import unfold_values
import torch
from math import ceil
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from attacks.data_recorder import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from attacks.util import BaseTask, OriginalScoreAccTask, update_graph
from models.util import seed_from_param, to_dense_data
from attacks.util import select_small, select_large
from attacks.evaluate import eval_attacked_graphs
from torch_geometric.utils import to_dense_adj
import pandas as pd
import attacks.consts as consts
from torch_geometric.data import Batch

use_assert = False


def drop_rdm_undir_edges(data, budget, relative=False):
    """creates a copy of the graph `data` and removes `budget`
    randomly chosen undirected edges"""
    if use_assert:
        assert data.is_undirected()
    m = int(data.num_edges / 2)
    abs_budget = ceil(budget * m) if relative else budget
    if abs_budget > m:
        abs_budget = m
    idx = torch.randperm(m)[abs_budget:]
    idx = torch.cat((2 * idx, 2 * idx + 1))
    edge_index = data.edge_index[:, idx]
    edge_attr = data.edge_attr[idx]
    new_data = update_graph(data, edge_index=edge_index, edge_attr=edge_attr)
    if use_assert:
        assert new_data.is_undirected()
    return new_data


def add_rdm_edges(
    data,
    budget: int,
    directed=False,
    relative=True,
    allow_self_loops=False,
):
    n = data.num_nodes
    m = data.num_edges if directed else int(data.num_edges / 2)
    abs_budget = ceil(budget * m) if relative else budget
    if abs_budget > m:
        abs_budget = m

    # find missing edges
    adj = to_dense_adj(data.edge_index, max_num_nodes=n)
    if not allow_self_loops:
        adj += torch.eye(n, device=adj.device)
    missing_edges = (adj==0).nonzero()[:,1:].t()
    if not directed:
        mask = missing_edges[0] < missing_edges[1]
        missing_edges = missing_edges[:,mask]

    if missing_edges.shape[-1]<abs_budget:
        abs_budget = missing_edges.shape[-1]
    
    # add new edges to edge_index
    idx = torch.randperm(missing_edges.shape[1])[:abs_budget]
    new_edge_index = missing_edges[:,idx]
    if not directed:  # also add reverse edges
        new_edge_index = torch.cat((new_edge_index, new_edge_index.flip(0)), dim=1)
    edge_index = torch.cat((data.edge_index, new_edge_index), dim=1)
    
    # assign arbitrary edge attrs from already existing edge attrs
    idx = torch.randint(data.num_edges, (abs_budget,))
    new_edge_attr = data.edge_attr[idx]
    if not directed:
        new_edge_attr = torch.cat((new_edge_attr, new_edge_attr))
    edge_attr = torch.cat((data.edge_attr, new_edge_attr), dim=0)

    new_data = update_graph(
        data, edge_index=edge_index, edge_attr=edge_attr, clone=False
    )
    if not directed and use_assert:
        assert new_data.is_undirected()

    return new_data


def rewire_rdm_edges(data, budget: int, directed=False, relative=True):
    device = data.edge_index.device
    edge_index = data.edge_index.clone()

    m = data.num_edges if directed else int(data.num_edges / 2)
    abs_budget = ceil(budget * m) if relative else budget
    if abs_budget > m:
        abs_budget = m

    idx = torch.randperm(m)[:abs_budget]

    direction = torch.randint(2, (abs_budget,)) # determines which node gets changed in edge
    if directed:
        start_node = edge_index[direction, idx]
    else:
        start_node = edge_index[direction, 2 * idx]
    # data.num_nodes - 1 to avoid self-loops
    new_neighbours = torch.randint(data.num_nodes - 1, (abs_budget,), device=device)
    new_neighbours[new_neighbours >= start_node] += 1

    if directed:
        edge_index[1-direction, idx] = new_neighbours
    else:
        edge_index[1-direction, 2 * idx] = new_neighbours
        edge_index[direction, 2 * idx + 1] = new_neighbours

    new_data = update_graph(data, edge_index=edge_index)
    if not directed and use_assert:
        assert new_data.is_undirected()
    return new_data


def rdm_node_attr_change(data, budget, node_features, relative=False):
    n = data.num_nodes
    m = int(data.num_edges / 2)

    abs_budget = ceil(budget * m) if relative else budget
    if abs_budget > n:
        abs_budget = n

    dest_idx = torch.randperm(n)[:abs_budget]
    x = data.x.clone()
    for i, feat in enumerate(x[dest_idx]):
        options = list(unfold_values(node_features, feat))
        x[i] = random.choice(options)
    new_data = update_graph(data, x=x)
    return new_data


def rdm_edge_attr_change(data, budget, edge_features, relative=False):
    m = int(data.num_edges / 2)
    abs_budget = ceil(budget * m) if relative else budget
    if abs_budget > m:
        abs_budget = m

    dest_idx = torch.randperm(m)[:abs_budget]
    edge_attr = data.edge_attr.clone()

    for dst in dest_idx:
        feat = edge_attr[2*dst]
        options = list(unfold_values(edge_features, feat))
        new_attr = random.choice(options)
        edge_attr[2*dst] = new_attr
        edge_attr[2*dst + 1] = new_attr

    new_data = update_graph(data, edge_attr=edge_attr)
    assert new_data.is_undirected()
    return new_data


class RandomScoreAccTask(BaseTask):
    repeats: int
    batch_size: int = 16

    def __init__(
        self,
        perturbation_func,
        params,
        repeats=10,
        batch_size=None,
        eps=1e-7,
        compute_cycle_count=False,
        *args,
        **kwargs,
    ):
        super(RandomScoreAccTask, self).__init__(*args, **kwargs)

        # remove duplicate params
        self.params = np.array(params)  # copy
        self.params.sort()
        mask = np.ones(len(self.params), dtype=bool)
        mask[1:] = abs(self.params[1:] - self.params[:-1]) >= eps
        self.params = self.params[mask]
        self.eps = eps

        self.perturbation_func = perturbation_func
        self.repeats = repeats
        self.batch_size = batch_size
        self.compute_cycle_count = compute_cycle_count

    def set_data(self, model, param, recorder, dataset, device, args):
        recorder.update(budget=param)
        batch_size = self.batch_size if self.batch_size else 2*args.batch_size

        for repeat in range(self.repeats):
            print(f"running repeat {repeat}")
            recorder.update(repeat=repeat)

            seed = seed_from_param(args.seed, param, repeat)
            torch.manual_seed(seed)
            np.random.seed(seed)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for batch in tqdm(loader):
                batch = batch.to(device)

                max_num_nodes = max(graph.x.shape[0] for graph in batch.to_data_list())

                init_x, _, init_adj, init_edge_attr = to_dense_data(batch, max_num_nodes=max_num_nodes)
                perturbed_batch = Batch.from_data_list(
                    [
                        self.perturbation_func(batch[i].clone(), param)
                        for i in range(len(batch.y))
                    ]
                )

                results = eval_attacked_graphs(
                    model,
                    perturbed_batch,
                    directed=False,
                    init_x=init_x,
                    init_adj=init_adj,
                    init_edge_attr=init_edge_attr,
                    compute_cycle_count=self.compute_cycle_count,
                    encode=True,
                    id=perturbed_batch.id,
                    find_node_changes=False,
                    find_edge_changes=False,
                )
                recorder.record(
                    **results,
                )

    def exists(self, data, param):
        return (
            not data.empty
            and (data["budget"] == param).any()
            and all((data["repeat"] == repeat).any() for repeat in range(self.repeats))
        )



def drop_rdm_undir_edges_abs(data, budget):
    return drop_rdm_undir_edges(data, budget, relative=False)


def drop_rdm_undir_edges_rel(data, budget):
    return drop_rdm_undir_edges(data, budget, relative=True)


def add_rdm_undir_edges_abs(data, budget):
    return add_rdm_edges(data, budget, relative=False, directed=False)


def add_rdm_undir_edges_rel(data, budget):
    return add_rdm_edges(data, budget, relative=True, directed=False)


def rewire_rdm_undir_edges_abs(data, budget):
    return rewire_rdm_edges(data, budget, relative=False, directed=False)


def rewire_rdm_undir_edges_rel(data, budget):
    return rewire_rdm_edges(data, budget, relative=True, directed=False)


def rdm_node_attr_change_abs(data, budget, node_features):
    return rdm_node_attr_change(data, budget, node_features, relative=False)


def rdm_node_attr_change_rel(data, budget, node_features):
    return rdm_node_attr_change(data, budget, node_features, relative=True)


def rdm_edge_attr_change_abs(data, budget, edge_features):
    return rdm_edge_attr_change(data, budget, edge_features, relative=False)


def rdm_edge_attr_change_rel(data, budget, edge_features):
    return rdm_edge_attr_change(data, budget, edge_features, relative=True)


def PERTURBATION_TASKS(
        repeats,
        node_features,
        edge_features,
        name_prefix=None,
        abs_budget_list=list(range(1, 21)) + list(range(20, 41, 5)),
        rel_budget_list=np.concatenate((np.linspace(0.01, 0.1, 10), np.linspace(0.12, 0.2, 5),  np.linspace(0.3, 1, 8))),
        with_node_attr=True,
        with_edge_attr=True,
        relative=True,
        *args,
        **kwargs):
    task_name = lambda s: "_".join(filter(None, (name_prefix, s)))

    tasks = {
        task_name("drop_rdm_undir_edges_abs"): RandomScoreAccTask(
            drop_rdm_undir_edges_abs,
            abs_budget_list,
            repeats=repeats,
            *args,
            **kwargs,
        ),
        task_name("add_rdm_undir_edges_abs"): RandomScoreAccTask(
            add_rdm_undir_edges_abs,
            abs_budget_list,
            repeats=repeats,
            *args,
            **kwargs,
        ),
        task_name("rewire_rdm_undir_edges_abs"): RandomScoreAccTask(
            rewire_rdm_undir_edges_abs,
            abs_budget_list,
            repeats=repeats,
            *args,
            **kwargs,
        ),
    }

    if relative:
        tasks |= {
            task_name("drop_rdm_undir_edges_rel"): RandomScoreAccTask(
                drop_rdm_undir_edges_rel,
                rel_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
            task_name("add_rdm_undir_edges_rel"): RandomScoreAccTask(
                add_rdm_undir_edges_rel,
                rel_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
            task_name("rewire_rdm_undir_edges_rel"): RandomScoreAccTask(
                rewire_rdm_undir_edges_rel,
                rel_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
        }
    if with_node_attr:
        tasks = tasks | {
            task_name("rdm_node_attr_change_abs"): RandomScoreAccTask(
                lambda x, y: rdm_node_attr_change_abs(x, y, node_features),
                abs_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
        }
        if relative:
            tasks = tasks | {
                task_name("rdm_node_attr_change_rel"): RandomScoreAccTask(
                    lambda x, y: rdm_node_attr_change_rel(x, y, node_features),
                    rel_budget_list,
                    repeats=repeats,
                    *args,
                    **kwargs,
                ),
            }

    if with_edge_attr:
        tasks = tasks | {
            task_name("rdm_edge_attr_change_abs"): RandomScoreAccTask(
                lambda x, y: rdm_edge_attr_change_abs(x, y, edge_features),
                abs_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
        }
        if relative:
            tasks = tasks | {
                task_name("rdm_edge_attr_change_rel"): RandomScoreAccTask(
                lambda x, y: rdm_edge_attr_change_rel(x, y, edge_features),
                rel_budget_list,
                repeats=repeats,
                *args,
                **kwargs,
            ),
            }

    return tasks
