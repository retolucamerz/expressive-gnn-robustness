import argparse
from attacks import consts, fetch_data
from attacks.bruteforce_attacks import BF_PERTURBATIONS, BruteforceAttackTask
from attacks.evaluate import classflip_target, decr_target, incr_target, untargeted_multiclass
from attacks.random_experiments import PERTURBATION_TASKS
from attacks.run_feature_attack import ATTRPGD_TASKS, get_IMDB_encoders, get_MUTAG_encoders, get_ZINC_encoders, get_molhiv_encoders
from attacks.run_gradient_attacks import ADJPGD_TASKS
from attacks.util import OriginalScoreAccTask
import torch
import numpy as np
from attacks.data_recorder import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)
from attacks.random_sampling_attack import PERTURBATION_ATTK_TASKS



models_per_dataset = {}

MolHIV_models = {}
models_per_dataset["ogbg-molhiv"] = MolHIV_models
MolHIV_models["GINE"] = [
    "GINE_ogbg-molhiv_300_5_0_20230726-081550",
    "GINE_ogbg-molhiv_300_5_1_20230726-085122",
    "GINE_ogbg-molhiv_300_5_2_20230726-092730",
    "GINE_ogbg-molhiv_300_5_3_20230726-095447",
    "GINE_ogbg-molhiv_300_5_4_20230726-103434",
]
MolHIV_models["DropGINE"] = [
    "DropGINE_ogbg-molhiv_150_5_0_20230812-083351",
    "DropGINE_ogbg-molhiv_150_5_1_20230812-102138",
    "DropGINE_ogbg-molhiv_150_5_2_20230812-120944",
    "DropGINE_ogbg-molhiv_150_5_3_20230812-083344",
    "DropGINE_ogbg-molhiv_150_5_4_20230812-100605",
]
MolHIV_models["PPGN"] = [
    "PPGN_ogbg-molhiv_64_5_0_20230726-081732",
    "PPGN_ogbg-molhiv_64_5_1_20230726-123015",
    "PPGN_ogbg-molhiv_64_5_2_20230726-160557",
    "PPGN_ogbg-molhiv_64_5_3_20230726-195755",
    "PPGN_ogbg-molhiv_64_5_4_20230727-001306",
]
MolHIV_models["SignNet"] = [
    "SignNet_ogbg-molhiv_95_16_0_20230726-081823",
    "SignNet_ogbg-molhiv_95_16_1_20230726-185200",
    "SignNet_ogbg-molhiv_95_16_2_20230727-040952",
    "SignNet_ogbg-molhiv_95_16_3_20230726-081827",
    # "SignNet_ogbg-molhiv_95_16_4_20230726-211825", # no convergence!!!
    "SignNet_ogbg-molhiv_95_16_5_20230727-112355",
]
MolHIV_models["ESAN"] = [
    "ESAN_ogbg-molhiv_64_5_0_20230726-081849",
    "ESAN_ogbg-molhiv_64_5_1_20230726-192547",
    "ESAN_ogbg-molhiv_64_5_2_20230727-064533",
    "ESAN_ogbg-molhiv_64_5_3_20230726-082445",
    "ESAN_ogbg-molhiv_64_5_4_20230726-183051",
]
MolHIV_models["Baseline"] = [
    "Baseline_ogbg-molhiv_300_5_0_20230726-082158",
    "Baseline_ogbg-molhiv_300_5_1_20230726-090732",
    "Baseline_ogbg-molhiv_300_5_2_20230726-095908",
    "Baseline_ogbg-molhiv_300_5_3_20230726-104740",
    "Baseline_ogbg-molhiv_300_5_4_20230726-114032",
]
MolHIV_models["GIN"] = [
    "GIN_ogbg-molhiv_300_5_0_20230726-083808",
    "GIN_ogbg-molhiv_300_5_1_20230726-092410",
    "GIN_ogbg-molhiv_300_5_2_20230726-101537",
    "GIN_ogbg-molhiv_300_5_3_20230726-111002",
    "GIN_ogbg-molhiv_300_5_4_20230726-115817",
]
MolHIV_models["MeanAggrGINE"] = [
    "MeanAggrGINE_ogbg-molhiv_300_5_0_20230726-082206",
    "MeanAggrGINE_ogbg-molhiv_300_5_1_20230726-092418",
    "MeanAggrGINE_ogbg-molhiv_300_5_2_20230726-103220",
    "MeanAggrGINE_ogbg-molhiv_300_5_3_20230726-113205",
    "MeanAggrGINE_ogbg-molhiv_300_5_4_20230726-123302",
]


ZINC_models = {}
models_per_dataset["ZINC12k"] = ZINC_models
ZINC_models["GINE"] = [
    "GINE_ZINC12k_300_5_0_20230722-215952",
    "GINE_ZINC12k_300_5_1_20230722-224337",
    "GINE_ZINC12k_300_5_2_20230722-232255",
    "GINE_ZINC12k_300_5_3_20230723-000820",
    "GINE_ZINC12k_300_5_4_20230723-004155",
]
ZINC_models["DropGINE"] = [
    "DropGINE_ZINC12k_300_5_0_20230722-215957",
    "DropGINE_ZINC12k_300_5_1_20230722-232154",
    "DropGINE_ZINC12k_300_5_2_20230723-005704",
    "DropGINE_ZINC12k_300_5_3_20230723-023415",
    "DropGINE_ZINC12k_300_5_4_20230723-035349",
]
ZINC_models["PPGN"] = [
    "PPGN_ZINC12k_300_5_0_20230722-220030",
    "PPGN_ZINC12k_300_5_1_20230723-020823",
    "PPGN_ZINC12k_300_5_2_20230723-061503",
    "PPGN_ZINC12k_300_5_3_20230722-220036",
    "PPGN_ZINC12k_300_5_4_20230723-021928",
]
ZINC_models["SignNet"] = [
    "SignNet_ZINC12k_95_16_0_20230722-220050",
    # "SignNet_ZINC12k_95_16_1_20230723-033850", # extreme results in evaluation
    "SignNet_ZINC12k_95_16_2_20230723-085327",
    "SignNet_ZINC12k_95_16_3_20230723-135707",
    "SignNet_ZINC12k_95_16_4_20230723-185852",
    "SignNet_ZINC12k_95_16_5_20230807-195121",
]
ZINC_models["ESAN"] = [
    "ESAN_ZINC12k_64_5_0_20230722-220102",
    "ESAN_ZINC12k_64_5_1_20230723-033146",
    "ESAN_ZINC12k_64_5_2_20230723-084404",
    "ESAN_ZINC12k_64_5_3_20230722-220109",
    "ESAN_ZINC12k_64_5_4_20230723-033750",
]
ZINC_models["Baseline"] = [
    "Baseline_ZINC12k_300_5_0_20230722-220130",
    "Baseline_ZINC12k_300_5_1_20230722-223757",
    "Baseline_ZINC12k_300_5_2_20230722-231745",
    "Baseline_ZINC12k_300_5_3_20230722-235107",
    "Baseline_ZINC12k_300_5_4_20230723-003218",
]
ZINC_models["GIN"] = [
    "GIN_ZINC12k_300_5_0_20230722-221932",
    "GIN_ZINC12k_300_5_1_20230722-225603",
    "GIN_ZINC12k_300_5_2_20230722-233137",
    "GIN_ZINC12k_300_5_3_20230723-000605",
    "GIN_ZINC12k_300_5_4_20230723-004149",
]
ZINC_models["MeanAggrGINE"] = [
    "MeanAggrGINE_ZINC12k_300_5_0_20230722-220147",
    "MeanAggrGINE_ZINC12k_300_5_1_20230722-223002",
    "MeanAggrGINE_ZINC12k_300_5_2_20230722-231238",
    "MeanAggrGINE_ZINC12k_300_5_3_20230722-235244",
    "MeanAggrGINE_ZINC12k_300_5_4_20230723-003640",
]


IMDB_BIN_models = {}
models_per_dataset["IMDB_BIN"] = IMDB_BIN_models
IMDB_BIN_models["GINE"] = [
    "GINE_IMDB-BINARY_300_5_0_20230812-183524",
    "GINE_IMDB-BINARY_300_5_1_20230812-195820",
    "GINE_IMDB-BINARY_300_5_2_20230812-210747",
    "GINE_IMDB-BINARY_300_5_3_20230812-222824",
    "GINE_IMDB-BINARY_300_5_4_20230812-232659",
]
IMDB_BIN_models["DropGINE"] = [
    "DropGINE_IMDB-BINARY_300_5_0_20230812-183726",
    "DropGINE_IMDB-BINARY_300_5_1_20230812-200006",
    "DropGINE_IMDB-BINARY_300_5_2_20230812-210912",
    "DropGINE_IMDB-BINARY_300_5_3_20230812-222941",
    "DropGINE_IMDB-BINARY_300_5_4_20230812-232829",
]
IMDB_BIN_models["PPGN"] = [
    "PPGN_IMDB-BINARY_300_5_0_20230812-184526",
    "PPGN_IMDB-BINARY_300_5_1_20230812-200825",
    "PPGN_IMDB-BINARY_300_5_2_20230812-211250",
    "PPGN_IMDB-BINARY_300_5_3_20230812-223613",
    "PPGN_IMDB-BINARY_300_5_4_20230812-233518",
]
IMDB_BIN_models["SignNet"] = [
    "SignNet_IMDB-BINARY_95_16_0_20230812-193258",
    "SignNet_IMDB-BINARY_95_16_1_20230812-203642",
    "SignNet_IMDB-BINARY_95_16_2_20230812-213531",
    "SignNet_IMDB-BINARY_95_16_3_20230812-225953",
    "SignNet_IMDB-BINARY_95_16_4_20230813-000319",
]
IMDB_BIN_models["ESAN"] = [
    "ESAN_IMDB-BINARY_64_5_0_20230812-194620",
    "ESAN_IMDB-BINARY_64_5_1_20230812-204651",
    "ESAN_IMDB-BINARY_64_5_2_20230812-221158",
    "ESAN_IMDB-BINARY_64_5_3_20230812-231304",
    "ESAN_IMDB-BINARY_64_5_4_20230813-003141",
]
IMDB_BIN_models["Baseline"] = [
    "Baseline_IMDB-BINARY_300_5_0_20230812-121322",
    "Baseline_IMDB-BINARY_300_5_1_20230812-121509",
    "Baseline_IMDB-BINARY_300_5_2_20230812-121657",
    "Baseline_IMDB-BINARY_300_5_3_20230812-121842",
    "Baseline_IMDB-BINARY_300_5_4_20230812-122132",
]
IMDB_BIN_models["MeanAggrGINE"] = [
    "MeanAggrGINE_IMDB-BINARY_300_5_0_20230812-220253",
    "MeanAggrGINE_IMDB-BINARY_300_5_1_20230812-231923",
    "MeanAggrGINE_IMDB-BINARY_300_5_2_20230813-004106",
    "MeanAggrGINE_IMDB-BINARY_300_5_3_20230813-014958",
    "MeanAggrGINE_IMDB-BINARY_300_5_4_20230813-031353",
]

IMDB_MULTI_models = {}
models_per_dataset["IMDB_MULTI"] = IMDB_MULTI_models
IMDB_MULTI_models["GINE"] = [
    "GINE_IMDB-MULTI_300_5_0_20230812-220542",
    "GINE_IMDB-MULTI_300_5_1_20230812-232114",
    "GINE_IMDB-MULTI_300_5_2_20230813-004318",
    "GINE_IMDB-MULTI_300_5_3_20230813-015150",
    "GINE_IMDB-MULTI_300_5_4_20230813-031614",
]
IMDB_MULTI_models["DropGINE"] = [
    "DropGINE_IMDB-MULTI_300_5_0_20230812-220904",
    "DropGINE_IMDB-MULTI_300_5_1_20230812-232819",
    "DropGINE_IMDB-MULTI_300_5_2_20230813-004824",
    "DropGINE_IMDB-MULTI_300_5_3_20230813-015744",
    "DropGINE_IMDB-MULTI_300_5_4_20230813-032113",
]
IMDB_MULTI_models["PPGN"] = [
    "PPGN_IMDB-MULTI_300_5_0_20230812-221551",
    "PPGN_IMDB-MULTI_300_5_1_20230812-233432",
    "PPGN_IMDB-MULTI_300_5_2_20230813-005406",
    "PPGN_IMDB-MULTI_300_5_3_20230813-020637",
    "PPGN_IMDB-MULTI_300_5_4_20230813-032509",
]
IMDB_MULTI_models["SignNet"] = [
    "SignNet_IMDB-MULTI_95_16_0_20230812-223530",
    "SignNet_IMDB-MULTI_95_16_1_20230812-235440",
    "SignNet_IMDB-MULTI_95_16_2_20230813-011927",
    "SignNet_IMDB-MULTI_95_16_3_20230813-024145",
    "SignNet_IMDB-MULTI_95_16_4_20230813-035209",
]
IMDB_MULTI_models["ESAN"] = [
    "ESAN_IMDB-MULTI_64_5_0_20230812-224244",
    "ESAN_IMDB-MULTI_64_5_1_20230813-001537",
    "ESAN_IMDB-MULTI_64_5_2_20230813-012618",
    "ESAN_IMDB-MULTI_64_5_3_20230813-025052",
    "ESAN_IMDB-MULTI_64_5_4_20230813-040439",
]
IMDB_MULTI_models["Baseline"] = [
    "Baseline_IMDB-MULTI_300_5_0_20230812-121416",
    "Baseline_IMDB-MULTI_300_5_1_20230812-121607",
    "Baseline_IMDB-MULTI_300_5_2_20230812-121752",
    "Baseline_IMDB-MULTI_300_5_3_20230812-121932",
    "Baseline_IMDB-MULTI_300_5_4_20230812-122234",
]
IMDB_MULTI_models["MeanAggrGINE"] = [
    "MeanAggrGINE_IMDB-MULTI_300_5_0_20230812-231713",
    "MeanAggrGINE_IMDB-MULTI_300_5_1_20230813-003901",
    "MeanAggrGINE_IMDB-MULTI_300_5_2_20230813-014724",
    "MeanAggrGINE_IMDB-MULTI_300_5_3_20230813-031131",
    "MeanAggrGINE_IMDB-MULTI_300_5_4_20230813-042549",
]

MUTAG_models = {}
models_per_dataset["MUTAG"] = MUTAG_models
MUTAG_models["GINE"] = [
    "GINE_MUTAG_300_5_0_20230729-143636",
    "GINE_MUTAG_300_5_1_20230729-145048",
    "GINE_MUTAG_300_5_2_20230729-150323",
    "GINE_MUTAG_300_5_3_20230729-151427",
    "GINE_MUTAG_300_5_4_20230729-152453",
    ## adversarial
    "GINE_MUTAG_300_5_0_20230804-094645",
    "GINE_MUTAG_300_5_1_20230804-113531",
    "GINE_MUTAG_300_5_2_20230804-124836",
    "GINE_MUTAG_300_5_3_20230804-145430",
    "GINE_MUTAG_300_5_4_20230804-163447",
]
MUTAG_models["DropGINE"] = [
    "DropGINE_MUTAG_300_5_0_20230729-143747",
    "DropGINE_MUTAG_300_5_1_20230729-145141",
    "DropGINE_MUTAG_300_5_2_20230729-150412",
    "DropGINE_MUTAG_300_5_3_20230729-151542",
    "DropGINE_MUTAG_300_5_4_20230729-152552",
    ## adversarial
    "DropGINE_MUTAG_300_5_0_20230804-095856",
    "DropGINE_MUTAG_300_5_1_20230804-114811",
    "DropGINE_MUTAG_300_5_2_20230804-125807",
    "DropGINE_MUTAG_300_5_3_20230804-145904",
    "DropGINE_MUTAG_300_5_4_20230804-164729",
]
MUTAG_models["PPGN"] = [
    "PPGN_MUTAG_300_5_0_20230729-143930",
    "PPGN_MUTAG_300_5_1_20230729-145322",
    "PPGN_MUTAG_300_5_2_20230729-150521",
    "PPGN_MUTAG_300_5_3_20230729-151716",
    "PPGN_MUTAG_300_5_4_20230729-152730",
    ## adversarial
    "PPGN_MUTAG_300_5_0_20230804-103501",
    "PPGN_MUTAG_300_5_1_20230804-120842",
    "PPGN_MUTAG_300_5_2_20230804-133403",
    "PPGN_MUTAG_300_5_3_20230804-153502",
    "PPGN_MUTAG_300_5_4_20230804-172311",
]
MUTAG_models["SignNet"] = [
    "SignNet_MUTAG_95_16_0_20230729-144300",
    "SignNet_MUTAG_95_16_1_20230729-145522",
    "SignNet_MUTAG_95_16_2_20230729-150649",
    "SignNet_MUTAG_95_16_3_20230729-151947",
    "SignNet_MUTAG_95_16_4_20230729-152923",
    ## adversarial
    "SignNet_MUTAG_95_16_0_20230804-110442",
    "SignNet_MUTAG_95_16_1_20230804-121614",
    "SignNet_MUTAG_95_16_2_20230804-134125",
    "SignNet_MUTAG_95_16_3_20230804-160412",
    "SignNet_MUTAG_95_16_4_20230804-173013",
]
MUTAG_models["ESAN"] = [
    "ESAN_MUTAG_64_5_0_20230729-144457",
    "ESAN_MUTAG_64_5_1_20230729-145716",
    "ESAN_MUTAG_64_5_2_20230729-150837",
    "ESAN_MUTAG_64_5_3_20230729-152135",
    "ESAN_MUTAG_64_5_4_20230729-153111",
    ## adversarial
    "ESAN_MUTAG_64_5_0_20230804-094651",
    "ESAN_MUTAG_64_5_1_20230804-110703",
    "ESAN_MUTAG_64_5_2_20230804-122931",
    "ESAN_MUTAG_64_5_3_20230804-135018",
    "ESAN_MUTAG_64_5_4_20230804-151411",
]
MUTAG_models["MeanAggrGINE"] = [
    "MeanAggrGINE_MUTAG_300_5_0_20230729-145013",
    "MeanAggrGINE_MUTAG_300_5_1_20230729-150240",
    "MeanAggrGINE_MUTAG_300_5_2_20230729-151350",
    "MeanAggrGINE_MUTAG_300_5_3_20230729-152411",
    "MeanAggrGINE_MUTAG_300_5_4_20230729-153625",
    ## adversarial
    "MeanAggrGINE_MUTAG_300_5_0_20230804-111829",
    "MeanAggrGINE_MUTAG_300_5_1_20230804-123016",
    "MeanAggrGINE_MUTAG_300_5_2_20230804-143620",
    "MeanAggrGINE_MUTAG_300_5_3_20230804-161640",
    "MeanAggrGINE_MUTAG_300_5_4_20230804-174357",
]
MUTAG_models["GIN"] = [
    "GIN_MUTAG_300_5_0_20230804-191623",
    "GIN_MUTAG_300_5_1_20230804-192606",
    "GIN_MUTAG_300_5_2_20230804-193954",
    "GIN_MUTAG_300_5_3_20230804-195357",
    "GIN_MUTAG_300_5_4_20230804-201054",
    ## adversarial
    "GIN_MUTAG_300_5_0_20230804-191701",
    "GIN_MUTAG_300_5_1_20230804-192711",
    "GIN_MUTAG_300_5_2_20230804-194031",
    "GIN_MUTAG_300_5_3_20230804-195432",
    "GIN_MUTAG_300_5_4_20230804-201138",
]
MUTAG_models["Baseline"] = [
    "Baseline_MUTAG_300_5_0_20230807-082719",
    "Baseline_MUTAG_300_5_1_20230807-082744",
    "Baseline_MUTAG_300_5_2_20230807-082808",
    "Baseline_MUTAG_300_5_3_20230807-082834",
    "Baseline_MUTAG_300_5_4_20230807-082855",
]

data_selection = lambda _, __, x: x

def TASKS(node_attr, edge_attr, transform_df, get_encoders, concrete_node_attr=None, concrete_edge_attr=None, targets=classflip_target, num_rep_adjpgd=3, compute_cycle_count=False, relative=True):
    pgd_steps = 25
    pgd_samples = 250

    if not isinstance(targets, dict):
        targets = {"": targets}

    if concrete_node_attr is None:
        concrete_node_attr = node_attr
    if concrete_edge_attr is None:
        concrete_edge_attr = edge_attr

    def inner(limit_edgeattr_by_adj=False, with_node_attr=True, with_edge_attr=True, only_original=False, attack_params={}, pgd_batch_size=4):
        tasks = {
            "original": OriginalScoreAccTask(data_selection=data_selection, compute_cycle_count=compute_cycle_count, transform_df=transform_df),
        }

        if only_original:
            return tasks

        tasks |= {
            **PERTURBATION_TASKS(
                5,
                node_attr,
                edge_attr,
                abs_budget_list=list(range(1, 11)) + list(range(10, 41, 5)),
                rel_budget_list=np.concatenate((np.linspace(0.01, 0.1, 10), np.linspace(0.12, 0.2, 5), np.linspace(0.3, 0.5, 3))),
                transform_df=transform_df,
                data_selection=data_selection,
                with_node_attr=with_node_attr,
                with_edge_attr=with_edge_attr,
                compute_cycle_count=compute_cycle_count,
            ),
            **ADJPGD_TASKS(
                targets,
                num_rep_adjpgd,
                abs_budget_list=list(range(1,6)),
                rel_budget_list=np.linspace(0.01, 0.05, 5),
                attack_params={
                    "pgd_steps": pgd_steps,
                    "n_samples": pgd_samples,
                    "limit_edgeattr_by_adj": limit_edgeattr_by_adj,
                    "compute_cycle_count": compute_cycle_count,
                    **attack_params
                    },
                pgd_batch_size=pgd_batch_size,
                data_selection=data_selection,
                transform_df=transform_df,
                relative=relative,
            ),
        }

        for target_name, target_func in targets.items():
            suffix = f"_{target_name}" if target_name else ""
            tasks |= {
                **PERTURBATION_ATTK_TASKS(
                    target_func,
                    50,
                    node_attr,
                    edge_attr,
                    name_suffix=target_name,
                    abs_budget_list=list(range(1,11)),
                    rel_budget_list=np.linspace(0.01, 0.1, 10),
                    transform_df=transform_df,
                    data_selection=data_selection,
                    with_node_attr=with_node_attr,
                    with_edge_attr=with_edge_attr,
                ),
                f"bruteforce{suffix}": BruteforceAttackTask(
                    target_func,
                    BF_PERTURBATIONS(
                        node_attr,
                        edge_attr,
                        with_node_attr=with_node_attr,
                        with_edge_attr=with_edge_attr),
                    transform_df=transform_df,
                    data_selection=data_selection,
                    compute_cycle_count=compute_cycle_count,
                ),
            }
        if with_node_attr or with_edge_attr:
            tasks |= ATTRPGD_TASKS(
                targets,
                get_encoders,
                concrete_node_attr,
                concrete_edge_attr,
                abs_budget_list=list(range(1,6)),
                rel_budget_list=np.linspace(0.01, 0.05, 5),
                attack_params={
                    "pgd_steps": pgd_steps,
                    "n_samples": pgd_samples,
                    "limit_edgeattr_by_adj": limit_edgeattr_by_adj,
                    "update_x": with_node_attr,
                    "update_edge_attr": with_edge_attr,
                    "compute_cycle_count": compute_cycle_count,
                    **attack_params},
                pgd_batch_size=pgd_batch_size,
                data_selection=data_selection,
                transform_df=transform_df,
                relative=relative,
            )
        return tasks
    
    return inner


create_tasks_by_dataset = {}
create_tasks_by_dataset["ZINC12k"] = TASKS(
    consts.zinc12k_node_attr,
    consts.zinc12k_edge_attr,
    compute_regression_metrics,
    get_ZINC_encoders,
    targets={"incr": incr_target, "decr": decr_target},
    compute_cycle_count=True
)
create_tasks_by_dataset["ogbg-molhiv"] = TASKS(
    consts.ogbmol_atom_attr,
    consts.ogbmol_bond_attr,
    compute_classification_metrics,
    get_molhiv_encoders,
    concrete_node_attr=consts.occuring_MolHIV_atom_vals,
    concrete_edge_attr=consts.occuring_MolHIV_bond_vals,
    targets=classflip_target,
)
create_tasks_by_dataset["IMDB_BIN"] = TASKS(
    consts.imdb_node_attr,
    consts.imdb_edge_attr,
    compute_classification_metrics,
    get_IMDB_encoders,
    targets=classflip_target,
    num_rep_adjpgd=1
)
create_tasks_by_dataset["IMDB_MULTI"] = TASKS(
    consts.imdb_node_attr,
    consts.imdb_edge_attr,
    compute_multiclass_metrics,
    get_IMDB_encoders,
    targets=untargeted_multiclass,
    num_rep_adjpgd=1
)
create_tasks_by_dataset["MUTAG"] = TASKS(
    consts.mutag_node_attr,
    consts.mutag_edge_attr,
    compute_classification_metrics,
    get_MUTAG_encoders,
    targets=classflip_target,
    num_rep_adjpgd=3
)


all_tasks = {}
for dataset, model_types in models_per_dataset.items():
    for model_type, models in model_types.items():
        only_original = model_type=="Baseline"

        limit_edgeattr_by_adj = model_type in ["PPGN", "PPGNnoEdgeFeat"]
        with_node_attr = not dataset.startswith("IMDB")
        with_edge_attr = not dataset.startswith("IMDB") and model_type not in ["GIN", "DropGIN", "PPGNnoEdgeFeat"]

        attack_params = {}

        if dataset=="ogbg-molhiv":
            if model_type=="DropGINE":
                pgd_batch_size = 1
            elif model_type=="ESAN":
                pgd_batch_size = 1
            else:
                pgd_batch_size = 4
        else:
            pgd_batch_size = 16

        model_tasks = create_tasks_by_dataset[dataset](
            limit_edgeattr_by_adj=limit_edgeattr_by_adj,
            with_node_attr=with_node_attr,
            with_edge_attr=with_edge_attr,
            only_original=only_original,
            attack_params=attack_params,
            pgd_batch_size=pgd_batch_size)
        for model in models:
            all_tasks[model] = model_tasks


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--task_contains", type=str, default=None)
    parser.add_argument(
        "--write",
        type=str2bool,
        default=True,
    )
    args = parser.parse_args()

    if args.model_prefix is None:
        attack_tasks_subset = all_tasks
    else:
        attack_tasks_subset = {
            key: val
            for key, val in all_tasks.items()
            if key.startswith(args.model_prefix)
        }

    if args.task_contains is not None:
        for model_name, tasks in attack_tasks_subset.items():
            attack_tasks_subset[model_name] = {
                key: val for key, val in tasks.items() if args.task_contains in key
            }

    print(f"evaluating {len(attack_tasks_subset)} models")
    print(list(attack_tasks_subset.keys()))

    if not args.write:
        print("not writing back results to disk")

    fetch_data(attack_tasks_subset, device, write_to_disk=args.write, models_directory="final_models")
