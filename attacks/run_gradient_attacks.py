import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from attacks.data_recorder import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from attacks.util import BaseTask, grid_search, random_search
from attacks.gradient_attacks import AdjPGD
from torch_geometric.loader import DataLoader
from models import create_surrogate_model
from models.util import seed_from_param


class SteppedGradientAttackTask(BaseTask):
    def __init__(
        self,
        target,
        params,
        repeats=3,
        attack_params={"pgd_steps": 50},
        data_selection=None,
        force_update=False,
        transform_df=None,
        transform_batch=None,
        pgd_batch_size=4,
    ):
        super(SteppedGradientAttackTask, self).__init__(
            data_selection=data_selection,
            force_update=force_update,
            transform_df=transform_df,
            transform_batch=transform_batch,
        )

        self.target = target
        self.params = params
        self.repeats = repeats
        self.attack_params = attack_params
        self.pgd_batch_size = pgd_batch_size

    def set_data(self, model, param, recorder, dataset, device, args):
        seed = seed_from_param(args.seed, param)
        torch.manual_seed(seed)
        np.random.seed(seed)

        surrogate_model = create_surrogate_model(
            model, negative_slope=0.01
        )
        model.eval()
        surrogate_model.eval()

        # find good hyperparams for given budget
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        for tuning_batch in loader:
            break
        def eval(base_lr):
            attacked_graphs = AdjPGD(
                model,
                tuning_batch.to(device),
                self.target,
                param,  # budget
                surrogate_model=surrogate_model,
                directed=False,
                seed=seed_from_param(args.seed, param),
                base_lr=base_lr,
                transform_batch=self.transform_batch,
                **self.attack_params,
            )
            return (
                self.target(model, attacked_graphs, encode=True).sum().item()
                / 16
            )

        (base_lr,), score = grid_search(
            eval, (0.1,), deviations=(3, 1), objective="max", show_progress=True
        )
        # base_lr, score = 100, 1/7
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
            for repeat in range(self.repeats):
                recorder.update(repeat=repeat)
                seed = seed_from_param(args.seed, param, repeat)
                for batch in tqdm(loader):
                    AdjPGD(
                        model,
                        batch.to(device),
                        self.target,
                        param,  # budget
                        surrogate_model=surrogate_model,
                        base_lr=base_lr,
                        directed=False,
                        seed=seed,
                        recorder=recorder,
                        **self.attack_params,
                    )

    def exists(self, data, param):
        return (
            not data.empty
            and (data["budget"] == param).any()
            and all((data["repeat"] == repeat).any() for repeat in range(self.repeats))
        )


def ADJPGD_TASKS(
    targets,
    repeats,
    attack_params={},
    abs_budget_list=list(range(1, 5 + 1)),
    rel_budget_list=np.linspace(0.01, 0.1, 5),
    transform_df=compute_regression_metrics,
    relative=True,
    **kwargs,
):
    def pack(name, target, params, relative_budget=False):
        return {
            name: SteppedGradientAttackTask(
                target,
                params,
                repeats=repeats,
                attack_params={"relative_budget": relative_budget} | attack_params,
                transform_df=transform_df,
                **kwargs,
            ),
        }

    ret = {}
    for target_name, target in targets.items():
        suffix = f"_{target_name}" if target_name else ""
        ret = ret | pack(
            f"adjpgd_abs{suffix}",
            target,
            abs_budget_list,
            relative_budget=False,
        )
        if relative:
            ret = ret | pack(
                f"adjpgd_rel{suffix}",
                target,
                rel_budget_list,
                relative_budget=True,
            )

    return ret
