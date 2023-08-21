from attacks import consts
import torch
import numpy as np
from tqdm import tqdm
from attacks.feature_attack import AttrPGD
from attacks.data_recorder import (
    compute_regression_metrics,
)
from attacks.util import BaseTask, grid_search, random_search
from models import create_surrogate_model
from models.util import seed_from_param
from torch_geometric.loader import DataLoader


class FeatureAttackTask(BaseTask):
    def __init__(
        self,
        target,
        params,
        get_encoders,
        vals_per_node_attr,
        vals_per_edge_attr,
        attack_params={"pgd_steps": 25},
        data_selection=None,
        force_update=False,
        transform_df=None,
        transform_batch=None,
        pgd_batch_size=4,
    ):
        super(FeatureAttackTask, self).__init__(
            data_selection=data_selection,
            force_update=force_update,
            transform_df=transform_df,
            transform_batch=transform_batch,
        )

        self.target = target
        self.params = params
        self.attack_params = attack_params
        self.vals_per_node_attr = vals_per_node_attr
        self.vals_per_edge_attr = vals_per_edge_attr
        self.get_encoders = get_encoders
        self.pgd_batch_size = pgd_batch_size

    def set_data(self, model, param, recorder, dataset, device, args):
        node_encoders, edge_encoders = self.get_encoders(model)

        # find good hyperparams for given budget
        seed = seed_from_param(args.seed, param)
        torch.manual_seed(seed)
        np.random.seed(seed)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for tuning_batch in loader:
            break

        surrogate_model = create_surrogate_model(
            model, negative_slope=0
        )
        model.eval()
        surrogate_model.eval()

        def eval(base_lr):
            attacked_graphs = AttrPGD(
                model,
                tuning_batch.to(device),
                self.target,
                param,  # budget
                self.vals_per_node_attr,
                self.vals_per_edge_attr,
                node_encoders,
                edge_encoders,
                surrogate_model=surrogate_model,
                directed=False,
                seed=seed_from_param(args.seed, param),
                base_lr=base_lr,
                **self.attack_params,
            )
            return (
                self.target(model, attacked_graphs, encode=True).sum().item()
                / 16
            )

        (base_lr,), score = grid_search(
            eval, (1,), deviations=(3, 1), objective="max", show_progress=True
        )
        print(
            f"HYPERPARAMETERS: base_lr {base_lr:.4f} with partial score {score:.4f}"
        )
        recorder.update(
            budget=param,
            base_lr=base_lr,
            negative_slope=0,
            **self.attack_params,
        )

        # run attacks
        small_graphs = [graph for graph in dataset if graph.num_nodes<50]
        large_graphs = [graph for graph in dataset if graph.num_nodes>=50]
        for dataset_subset, batch_size in zip((large_graphs, small_graphs), (self.pgd_batch_size, 8)):
            loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)
            seed = seed_from_param(args.seed, param)
            for batch in tqdm(loader):
                AttrPGD(
                    model,
                    batch.to(device),
                    self.target,
                    param,  # budget
                    self.vals_per_node_attr,
                    self.vals_per_edge_attr,
                    node_encoders,
                    edge_encoders,
                    surrogate_model=surrogate_model,
                    base_lr=base_lr,
                    directed=False,
                    seed=seed,
                    recorder=recorder,
                    transform_batch=self.transform_batch,
                    **self.attack_params,
                )

    def exists(self, data, param):
        return not data.empty and (data["budget"] == param).any()


def get_molhiv_encoders(model):
    return (
        model.atom_encoder.atom_embedding_list,
        model.bond_encoder.bond_embedding_list,
    )

def get_ZINC_encoders(model):
    return [model.atom_encoder], [model.bond_encoder]

def get_IMDB_encoders(model):
    return [model.atom_encoder], [model.bond_encoder]

def get_MUTAG_encoders(model):
    return [model.atom_encoder], [model.bond_encoder]


def ATTRPGD_TASKS(
    targets,
    get_encoders,
    vals_per_node_attr,
    vals_per_edge_attr,
    attack_params={},
    abs_budget_list=list(range(1, 5 + 1)),
    rel_budget_list=np.linspace(0.01, 0.1, 5),
    transform_df=compute_regression_metrics,
    relative=True,
    **kwargs,
):
    def pack(name, target, params, relative_budget=False):
        return {
            name: FeatureAttackTask(
                target,
                params,
                get_encoders,
                vals_per_node_attr,
                vals_per_edge_attr,
                attack_params={"relative_budget": relative_budget} | attack_params,
                transform_df=transform_df,
                **kwargs,
            ),
        }
    
    ret = {}
    for target_name, target in targets.items():
        suffix = f"_{target_name}" if target_name else ""
        ret = ret | pack(
            f"attrpgd_abs{suffix}",
            target,
            abs_budget_list,
            relative_budget=False,
        )
        if relative:
            ret = ret | pack(
                f"attrpgd_rel{suffix}",
                target,
                rel_budget_list,
                relative_budget=True,
            )

    return ret

