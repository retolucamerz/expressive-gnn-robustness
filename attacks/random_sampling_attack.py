from attacks import consts
from models.util import to_dense_data
import numpy as np
from torch_geometric.loader import DataLoader
from attacks.bruteforce_attacks import find_max_target_graph
from attacks.random_experiments import (
    add_rdm_undir_edges_abs,
    add_rdm_undir_edges_rel,
    drop_rdm_undir_edges_abs,
    drop_rdm_undir_edges_rel,
    rdm_edge_attr_change_abs,
    rdm_edge_attr_change_rel,
    rdm_node_attr_change_abs,
    rdm_node_attr_change_rel,
    rewire_rdm_undir_edges_abs,
    rewire_rdm_undir_edges_rel,
)

from attacks.util import BaseTask
from tqdm import tqdm


class RandomSamplingAttackTask(BaseTask):
    def __init__(
        self,
        target,
        pert_func,
        n_samples,
        params,
        data_selection=None,
        force_update=False,
        transform_df=None,
        transform_batch=None,
        batch_size=None,
        compute_cycle_count=False,
    ):
        super(RandomSamplingAttackTask, self).__init__(
            data_selection=data_selection,
            force_update=force_update,
            transform_df=transform_df,
            transform_batch=transform_batch,
        )
        self.target = target
        self.pert_func = pert_func
        self.n_samples = n_samples
        self.params = params
        self.batch_size = batch_size
        self.compute_cycle_count = compute_cycle_count

    def set_data(self, model, param, recorder, dataset, device, args):
        batch_size = self.batch_size if self.batch_size else args.batch_size

        recorder.update(budget=param)

        for graph in tqdm(dataset):
            graph = graph.to(device)
            model.eval()
            init_x, mask, init_adj, init_edge_attr = to_dense_data(graph, max_num_nodes=graph.x.shape[0])

            sampled_graphs = [
                self.pert_func(graph, param) for _ in range(self.n_samples)
            ]
            loader = DataLoader(sampled_graphs, batch_size=batch_size)
            graph_results = find_max_target_graph(
                model,
                loader,
                self.target,
                device,
                init_x=init_x,
                init_adj=init_adj,
                init_edge_attr=init_edge_attr,
                directed=False,
                compute_cycle_count=self.compute_cycle_count
            )
            recorder.record(**graph_results)

    def exists(self, data, param):
        return not data.empty and (data["budget"] == param).any()


def PERTURBATION_ATTK_TASKS(
        target,
        n_samples,
        node_features,
        edge_features,
        name_suffix=None,
        abs_budget_list=list(range(1, 11)) + list(range(15, 41, 5)),
        rel_budget_list=np.concatenate((np.linspace(0.01, 0.1, 10), np.linspace(0.1, 0.5, 5))),
        with_node_attr=True,
        with_edge_attr=True,
        relative=True,
        *args,
        **kwargs):
    task_name = lambda s: "_".join(filter(None, (s, name_suffix)))
    tasks = {
        task_name("rdm_drop_atck_abs"): RandomSamplingAttackTask(
            target,
            drop_rdm_undir_edges_abs,
            n_samples,
            abs_budget_list,
            *args,
            **kwargs,
        ),
        task_name("rdm_add_atck_abs"): RandomSamplingAttackTask(
            target,
            add_rdm_undir_edges_abs,
            n_samples,
            abs_budget_list,
            *args,
            **kwargs,
        ),
        task_name("rdm_rewire_atck_abs"): RandomSamplingAttackTask(
            target,
            rewire_rdm_undir_edges_abs,
            n_samples,
            abs_budget_list,
            *args,
            **kwargs,
        ),
    }
    if relative:
        tasks = tasks | {
            task_name("rdm_drop_atck_rel"): RandomSamplingAttackTask(
                target,
                drop_rdm_undir_edges_rel,
                n_samples,
                rel_budget_list,
                *args,
                **kwargs,
            ),
            task_name("rdm_add_atck_rel"): RandomSamplingAttackTask(
                target,
                add_rdm_undir_edges_rel,
                n_samples,
                rel_budget_list,
                *args,
                **kwargs,
            ),
            task_name("rdm_rewire_atck_rel"): RandomSamplingAttackTask(
                target,
                rewire_rdm_undir_edges_rel,
                n_samples,
                rel_budget_list,
                *args,
                **kwargs,
            ),
        }

    if with_node_attr:
        tasks = tasks | {
            task_name("rdm_node_attr_atck_abs"): RandomSamplingAttackTask(
                target,
                lambda x, y: rdm_node_attr_change_abs(x, y, node_features),
                n_samples,
                abs_budget_list,
                *args,
                **kwargs,
            ),
        }
        if relative:
            tasks = tasks | {
                task_name("rdm_node_attr_atck_rel"): RandomSamplingAttackTask(
                    target,
                    lambda x, y: rdm_node_attr_change_rel(x, y, node_features),
                    n_samples,
                    rel_budget_list,
                    *args,
                    **kwargs,
                ),
            }
    if with_edge_attr:
        tasks = tasks | {
            task_name("rdm_edge_attr_atck_abs"): RandomSamplingAttackTask(
                target,
                lambda x, y: rdm_edge_attr_change_abs(x, y, edge_features),
                n_samples,
                abs_budget_list,
                *args,
                **kwargs,
            ),
        }
        if relative:
            tasks = tasks | {
                task_name("rdm_edge_attr_atck_rel"): RandomSamplingAttackTask(
                    target,
                    lambda x, y: rdm_edge_attr_change_rel(x, y, edge_features),
                    n_samples,
                    rel_budget_list,
                    *args,
                    **kwargs,
                ),
            }
    return tasks
