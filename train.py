from attacks import consts
from attacks.evaluate import classflip_target, untargeted
from attacks.gradient_attacks import AdjPGD
from attacks.run_feature_attack import get_MUTAG_encoders, get_ZINC_encoders, get_molhiv_encoders
from models.ESAN.sparse_models import DSSnetwork, DSnetwork
from models.util import count_parameters, set_all_seeds
import torch
import numpy as np
import os.path as osp
import time
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from attacks.feature_attack import AttrPGD

from models import create_surrogate_model, init_model, create_model_name
from models.args import Args, store_args, parse_args
from datasets import get_dataset_split
import logging
from dataclasses import asdict, replace
import wandb
import torch.nn.functional as F


def train(model, loader, optimizer, device, args):
    """trains `model` for a single epoch on data from `loader`"""
    loss_all = 0
    n = 0

    model.train()
    for data in loader:
        data = data.to(device)

        pred = model.predict(data)
        optimizer.zero_grad()
        loss = model.loss(pred, data.y)
        loss.backward()
        optimizer.step()

        loss_all += data.num_graphs * loss.item()
        n += len(data.y)

    return loss_all / n


def train_adversarial(model, loader, optimizer, device, args, attack_type="attr"):
    loss_all = 0
    n = 0

    if args.dataset=="ZINC12k":
        target = untargeted
        node_encoders, edge_encoders = get_ZINC_encoders(model)
        vals_per_node_attr = consts.zinc12k_node_attr
        vals_per_edge_attr = consts.zinc12k_edge_attr
    elif args.dataset=="ogbg-molhiv":
        target = classflip_target
        node_encoders, edge_encoders = get_molhiv_encoders(model)
        vals_per_node_attr = consts.ogbmol_atom_attr
        vals_per_edge_attr = consts.ogbmol_bond_attr
    elif args.dataset=="MUTAG":
        target = classflip_target
        node_encoders, edge_encoders = get_MUTAG_encoders(model)
        vals_per_node_attr = consts.mutag_node_attr
        vals_per_edge_attr = consts.mutag_edge_attr
    limit_edgeattr_by_adj = args.model=="PPGN"
    update_edge_attr = args.model not in ["GIN", "DropGIN", "PPGNnoEdgeFeat"]

    if isinstance(model, DSSnetwork) or isinstance(model, DSnetwork):
        model.enable_caching = False
    
    model.train()
    for data in loader:
        data = data.to(device)

        surrogate_model = create_surrogate_model(model, negative_slope=0)
        if attack_type=="attr":
            attacked_data = AttrPGD(
                model,
                data,
                target,
                1, # budget
                vals_per_node_attr,
                vals_per_edge_attr,
                node_encoders,  # list of torch.Embedding's
                edge_encoders,  # list of torch.Embedding's
                surrogate_model=surrogate_model,
                pgd_steps=3,
                n_samples=10,
                base_lr=0.1,
                directed=False,
                seed=0,
                limit_edgeattr_by_adj=limit_edgeattr_by_adj,
                update_edge_attr=update_edge_attr,
            )
        elif attack_type=="adj":
            attacked_data = AdjPGD(
                model,
                data,
                target,
                1, # budget
                surrogate_model=surrogate_model,
                pgd_steps=3,
                n_samples=10,
                directed=False,
                limit_edgeattr_by_adj=limit_edgeattr_by_adj,
                base_lr=0.1,
                seed=0,
                relative_budget=False,
            )
        else:
            raise ValueError(f"unknown attack type '{attack_type}'")
        
        optimizer.zero_grad()
        pred = model.predict(attacked_data)
        loss = model.loss(pred, data.y)
        loss_all += data.num_graphs * loss.item()
        n += len(data.y)

        loss.backward()
        optimizer.step()
        del pred

    return loss_all / n


def create_lr_scheduler(optimizer, objective, args):
    if "ESAN" in args.model:
        if "ZINC" in args.dataset:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=20
            )
        elif "ogb" in args.dataset:
            return None
        else:
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif "SignNet" in args.model:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=objective,
            factor=0.5,
            patience=25,
            verbose=True,
        )
    elif (
        "GIN" in args.model or "PPGN" in args.model
    ):  # catures (MeanAggr)(Drop)GIN(E)...
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=objective,
            factor=0.5,
            patience=25,
            verbose=True,
        )
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        # return None
    elif args.model == "Baseline":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=objective,
            factor=0.5,
            patience=25,
            verbose=True,
        )

    raise ValueError(f"no lr scheduler defined for '{args.model}'")


def update_scheduler(sched, step):
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        sched.step(step)
    elif isinstance(sched, torch.optim.lr_scheduler.StepLR):
        sched.step()


def get_lr(sched, args):
    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau) or isinstance(
        sched, torch.optim.lr_scheduler.StepLR
    ):
        return sched.optimizer.param_groups[0]["lr"]
    else:
        return args.lr

def train_model(args, use_wandb_run=False, use_wandb_sweep=False):
    # torch.use_deterministic_algorithms(True)
    set_all_seeds(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_split, valid_split, test_split = get_dataset_split(
        args.dataset, subset=args.subset
    )
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_split, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_split, batch_size=args.batch_size, shuffle=False)

    model = init_model(args, train_split).to(device)
    model_name = create_model_name(args)
    print(f"Model: {model_name}")
    print(f"Number of Parameters: {count_parameters(model)}")
    print(f"Device: {device}")
    print(f"Dataset Split: {len(train_split)} / {len(valid_split)} / {len(test_split)}")
    print(args)

    if use_wandb_run:
        wandb.init(
            project="expressive-gnn-robustness",
            config=asdict(args),  # | {"model_name": model_name}
            name=model_name,
        )

    if not args.save_model is None:
        import os

        dataset_dir = osp.join("saved_models", args.dataset)
        if not osp.exists(dataset_dir):
            os.makedirs(dataset_dir)
        model_directory = osp.join(dataset_dir, model_name)
        if not osp.exists(model_directory):
            os.makedirs(model_directory)
        store_args(model_name, args)

        logging.basicConfig(
            filename=osp.join(model_directory, "train.log"),
            encoding="utf-8",
            format="%(message)s",
            level=logging.INFO,
        )
        logging.info(f"Model: {model_name}")
        logging.info(f"Device: {device}")
        logging.info(f"Number of Parameters: {count_parameters(model)}")
        logging.info(
            f"Dataset Split: {len(train_split)} / {len(valid_split)} / {len(test_split)}"
        )
        logging.info(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_lr_scheduler(optimizer, model.metric_objective, args)

    if args.tensorboard:
        dataset_runs_dir = osp.join("runs", args.dataset)
        if not osp.exists(dataset_runs_dir):
            os.makedirs(dataset_runs_dir)
        writer = SummaryWriter(log_dir=osp.join(dataset_runs_dir, model_name))

    best_metrics = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch > args.patience:
            msg = f"stopping: best score has not improved since {args.patience} epochs"
            print(msg)
            if not args.save_model is None:
                logging.info(msg)
            break

        if args.verbose or epoch == args.epochs:
            start = time.time()

        if args.adversarial_training:
            train_loss = train_adversarial(model, train_loader, optimizer, device, args)
        else:
            train_loss = train(model, train_loader, optimizer, device, args)

        train_metrics = model.eval_metrics(train_loader, device)
        valid_metrics = model.eval_metrics(valid_loader, device)
        update_scheduler(scheduler, valid_metrics[model.main_metric])
        test_metrics = model.eval_metrics(test_loader, device)

        lr = get_lr(scheduler, args)
        if args.tensorboard:
            writer.add_scalar("Learning Rate", lr, epoch)
            writer.add_scalar("Train Loss", train_loss, epoch)
            for metric, value in train_metrics.items():
                writer.add_scalar(f"Train {metric}", value, epoch)
            for metric, value in valid_metrics.items():
                writer.add_scalar(f"Valid {metric}", value, epoch)
            for metric, value in test_metrics.items():
                writer.add_scalar(f"Test {metric}", value, epoch)

        if use_wandb_run or use_wandb_sweep:
            wandb_dict = (
                {f"train_{m}": v for m, v in train_metrics.items()}
                | {f"valid_{m}": v for m, v in valid_metrics.items()}
                | {f"test_{m}": v for m, v in test_metrics.items()}
                | {"lr": lr, "train_loss": train_loss}
            )
            wandb.log(wandb_dict)

        if not args.save_model is None and epoch % args.save_model == 0:
            torch.save(
                model.state_dict(),
                osp.join(model_directory, f"{model_name}_e{epoch:03d}.model"),
            )

        if args.verbose or epoch == args.epochs:
            epoch_time = time.time() - start
            msg = ", ".join(
                [
                    f"Epoch: {epoch:03d}",
                    f"Train Loss: {train_loss:7f}",
                    *[
                        f"Valid {metric}: {value:7f}"
                        for metric, value in valid_metrics.items()
                    ],
                    *[
                        f"Test {metric}: {value:7f}"
                        for metric, value in test_metrics.items()
                    ],
                    f"Time: {epoch_time:7f}",
                ]
            )
            print(msg)
            if not args.save_model is None:
                logging.info(msg)

        if best_metrics is None or model.compare_metrics(best_metrics, valid_metrics):
            best_metrics = valid_metrics
            best_epoch = epoch

            if not args.save_model is None:
                torch.save(
                    model.state_dict(),
                    osp.join(model_directory, f"{model_name}_best.model"),
                )

    msg = f"best metrics {best_metrics} achieved on epoch {best_epoch:03d}"
    print(msg)
    if not args.save_model is None:
        logging.info(msg)

    if use_wandb_run:
        wandb.finish()

    return model_name

if __name__ == "__main__":
    args = parse_args()
    train_model(args, use_wandb_run=False)
