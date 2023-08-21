import random
from models.GINEE import GINEE
from models.util import set_all_seeds
import torch
import time
import numpy as np
import os.path as osp
from models.BaselineMLP import Baseline
from models.DropGINE import DropGINE
from models.ESAN import ESANArgs, get_esan_model
from models.ESAN.dense_models import DenseDSSnetwork
from models.ESAN.sparse_models import DSSnetwork
from models.GIN import GIN, DenseGIN
from models.DropGIN import DenseDropGIN, DropGIN
from models.GINE import GINE, DenseGINE
from models.DropGINE import DropGINE, init_runs, DenseDropGINE
from models.GINE import GINE
from models.SignNet import DenseSignGINE, SignGINE
from models.ppgn import Powerful, PowerfulNoEdgeFeat
from models.args import load_args
from datasets import get_dataset_split, get_encoders, get_evaluator


def create_model_name(args):
    return f'{args.model}_{args.dataset}_{args.hidden_units}_{args.num_layers}_{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'


def init_model(args, train_data):
    num_node_features = args.hidden_units
    num_edge_features = args.hidden_units
    if args.dataset in ["ogbg-molhiv", "ogbg-molhiv-sm", "IMDB-BINARY", "MUTAG"] or args.dataset.startswith("ZINC"):
        num_outputs = 1
    elif args.dataset == "ogbg-ppa":
        num_outputs = 37
    elif args.dataset == "IMDB-MULTI":
        num_outputs = 3
    else:
        raise ValueError(f"dataset '{args.dataset}' not known")

    atom_encoder, bond_encoder = get_encoders(
        args.dataset, args.hidden_units, args.hidden_units, args.seed
    )
    evaluator = get_evaluator(args.dataset)

    pool_type = "mean"
    aggr_type = "mean" if args.model.startswith("MeanAggr") else "add"
    model_type = (
        args.model[8:]
        if args.model.startswith("MeanAggr")
        else args.model[4:]
        if args.model.startswith("Mean")
        else args.model
    )

    if aggr_type != "add" and model_type not in ["GINE", "DropGINE"]:
        raise ValueError(
            f"neighborhood aggregation type '{aggr_type}' not implemented for model '{model_type}'"
        )

    kwargs = {
        "num_features": num_node_features,
        "hidden_units": args.hidden_units,
        "dropout": args.dropout,
        "num_layers": args.num_layers,
        "atom_encoder": atom_encoder,
        "bond_encoder": bond_encoder,
    }

    set_all_seeds(args.seed)

    if model_type == "GINE":
        model = GINE(
            num_outputs,
            evaluator=evaluator,
            pool_type=pool_type,
            aggr_type=aggr_type,
            **kwargs,
        )
        model.reset_parameters()
        return model

    elif model_type == "DropGINE":
        p, num_runs = init_runs(train_data)
        model = DropGINE(
            num_outputs,
            num_runs,
            p,
            evaluator=evaluator,
            pool_type=pool_type,
            aggr_type=aggr_type,
            **kwargs,
        )
        model.reset_parameters()
        return model

    elif model_type == "PPGN":
        model = Powerful(
            num_classes=num_outputs,
            num_input_features=num_node_features + num_edge_features,
            num_layers=args.num_layers,
            hidden=args.hidden_units,
            hidden_final=args.hidden_units,
            dropout_prob=args.dropout,
            simplified=False,
            evaluator=evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )
        model.reset_parameters()
        return model
    elif model_type == "SignNet":
        model = SignGINE(
            num_outputs,
            pos_enc_dim=args.pos_enc_dim,
            hidden_dim=args.hidden_units,
            n_layers=args.num_layers,
            sign_inv_layers=args.sign_inv_layers,
            evaluator=evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
            dropout=args.dropout,
        )
        # model.reset_parameters()
        return model
    elif model_type == "ESAN":
        esan_args = ESANArgs.from_model_args(args)
        model = get_esan_model(
            esan_args,
            out_dim=num_outputs,
        )
        return model
    elif model_type == "DenseESAN":
        esan_args = ESANArgs.from_model_args(args)
        model = get_esan_model(
            esan_args,
            out_dim=num_outputs,
        )
        model = create_surrogate_model(model, negative_slope=0)
        return model
    elif model_type == "GIN":
        model = GIN(
            num_outputs,
            evaluator=evaluator,
            pool_type=pool_type,
            **kwargs,
        )
        model.reset_parameters()
        return model
    elif model_type == "DropGIN":
        p, num_runs = init_runs(train_data)
        model = DropGIN(
            num_outputs,
            num_runs,
            p,
            evaluator=evaluator,
            pool_type=pool_type,
            **kwargs,
        )
        model.reset_parameters()
        return model
    elif model_type == "PPGNnoEdgeFeat":
        model = PowerfulNoEdgeFeat(
            num_classes=num_outputs,
            num_input_features=num_node_features,
            num_layers=args.num_layers,
            hidden=args.hidden_units,
            hidden_final=args.hidden_units,
            dropout_prob=args.dropout,
            simplified=False,
            evaluator=evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )
        model.reset_parameters()
        return model
    elif model_type == "Baseline":
        model = Baseline(
            num_outputs,
            num_features=args.hidden_units,
            hidden_units=args.hidden_units,
            evaluator=evaluator,
            atom_encoder=atom_encoder,
            bond_encoder=bond_encoder,
        )
        model.reset_parameters()
        return model
    elif model_type == "GINEE":
        kwargs["bond_encoder"] = lambda x: x
        create_bond_encoder = lambda: get_encoders(
            args.dataset, args.hidden_units, args.hidden_units, args.seed
        )[1]
        model = GINEE(
            num_outputs,
            create_bond_encoder,
            evaluator=evaluator,
            pool_type=pool_type,
            **kwargs,
        )
        model.reset_parameters()
        return model

    else:
        raise ValueError(f"model '{model_type}' not implemented")


def create_surrogate_model(model, **kwargs):
    if isinstance(model, GINE):
        return DenseGINE.from_sparse(model, **kwargs)
    elif isinstance(model, DropGINE):
        return DenseDropGINE.from_sparse(model, **kwargs)
    elif isinstance(model, Powerful):
        return model
    elif isinstance(model, GIN):
        return DenseGIN.from_sparse(model, **kwargs)
    elif isinstance(model, DropGIN):
        return DenseDropGIN.from_sparse(model, **kwargs)
    elif isinstance(model, SignGINE):
        return DenseSignGINE.from_sparse(model, **kwargs)
    elif isinstance(model, DSSnetwork):
        return DenseDSSnetwork.from_sparse(model, **kwargs)
    else:
        raise ValueError("cannot create surrogate model")


def load_model(model_name, directory="saved_models", ext="best", attack=False):
    model_type = model_name.split("_")[0]
    dataset = model_name.split("_")[1]
    model_directory = osp.join(directory, dataset, model_name)

    # load pickeled arguments, init dataset & model
    args = load_args(model_name, directory=directory)
    train_split, valid_split, test_split = get_dataset_split(
        args.dataset, subset=0
    )
    model = init_model(args, train_split)

    # load model parameters
    model_path = osp.join(model_directory, f"{model_name}_{ext}.model")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    if attack and model_type=="ESAN":
        model.enable_caching = False

    return model, args, (train_split, valid_split, test_split)
