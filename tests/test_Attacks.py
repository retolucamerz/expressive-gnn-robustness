import random
import unittest
from attacks.bruteforce_attacks import BF_PERTURBATIONS, BruteforceAttackTask
import torch
from torch_geometric.data import Batch
import numpy as np

# # to enable imports from parent package
# import sys
# import os
# from pathlib import Path
# parent_dir = Path(os. getcwd()).parent.absolute()
# sys.path.append(str(parent_dir))

from attacks import consts
from attacks.data_recorder import compute_multiclass_metrics, compute_regression_metrics
from attacks.evaluate import decr_target, incr_target, untargeted_multiclass
from attacks.random_experiments import PERTURBATION_TASKS
from attacks.random_sampling_attack import PERTURBATION_ATTK_TASKS
from attacks.run_feature_attack import get_ZINC_encoders
from attacks.run_gradient_attacks import ADJPGD_TASKS
from run_attacks import fetch_data
from models import create_surrogate_model, init_model
from models.args import Args
from tests.util import rdm_graph

# class TestAttackRun(unittest.TestCase):
#     def test_integration(self):
#         device = (
#             "cuda"
#             if torch.cuda.is_available()
#             else "mps"
#             if torch.backends.mps.is_available()
#             else "cpu"
#         )

#         data_selection = lambda _, __, x: x[:14]

#         tasks = {
#             "GINE_IMDB-MULTI_300_5_0_20230717-115501": {
#                 **PERTURBATION_TASKS(
#                     1,
#                     consts.imdb_node_attr,
#                     consts.imdb_edge_attr,
#                     abs_budget_list=list(range(1,2)),
#                     rel_budget_list=np.linspace(0.01, 0.03, 1),
#                     transform_df=compute_multiclass_metrics,
#                     data_selection=data_selection,
#                     compute_cycle_count=False,
#                     with_node_attr=False,
#                     with_edge_attr=False,
#                 ),
#                 **PERTURBATION_ATTK_TASKS(
#                     untargeted_multiclass,
#                     5,
#                     consts.imdb_node_attr,
#                     consts.imdb_edge_attr,
#                     abs_budget_list=list(range(1,2)),
#                     rel_budget_list=np.linspace(0.01, 0.03, 3),
#                     transform_df=compute_multiclass_metrics,
#                     data_selection=data_selection,
#                     compute_cycle_count=False,
#                     with_node_attr=False,
#                     with_edge_attr=False,
#                 ),
#                 "bruteforce": BruteforceAttackTask(
#                     untargeted_multiclass,
#                     BF_PERTURBATIONS(consts.imdb_node_attr, consts.imdb_edge_attr, with_node_attr=False, with_edge_attr=False),
#                     transform_df=compute_multiclass_metrics,
#                     data_selection=data_selection,
#                     compute_cycle_count=False,
#                 ),
#                 # **ADJPGD_TASKS(
#                 #     {"": untargeted_multiclass},
#                 #     1,
#                 #     abs_budget_list=list(range(1,6)),
#                 #     rel_budget_list=np.linspace(0.01, 0.05, 5),
#                 #     attack_params={
#                 #         "pgd_steps": 25,
#                 #         "n_samples": 250,
#                 #         "limit_edgeattr_by_adj": False},
#                 #     data_selection=data_selection,
#                 #     transform_df=compute_multiclass_metrics,
#                 # ),
#             },
#         }

#         model_prefix = None
#         task_contains = None
#         if model_prefix is None:
#             attack_tasks_subset = tasks
#         else:
#             attack_tasks_subset = {
#                 key: val
#                 for key, val in tasks.items()
#                 if key.startswith(model_prefix)
#             }
#         if task_contains is not None:
#             for model_name, tasks in attack_tasks_subset.items():
#                 attack_tasks_subset[model_name] = {
#                     key: val for key, val in tasks.items() if task_contains in key
#                 }

#         fetch_data(attack_tasks_subset, device, results_directory="tests/outputs", write_to_disk=False, load_from_disk=False, verbose=False)
        