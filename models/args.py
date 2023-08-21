import argparse
from dataclasses import dataclass
import pickle
import os.path as osp


@dataclass
class Args:
    model: str = "DropGIN"
    dataset: str = "ogbg-molhiv"
    num_layers: int = 5
    hidden_units: int = 64
    dropout: float = 0.5
    lr: float = 0.01  # learning rate
    batch_size: int = 32
    epochs: int = 400  # nr. of epochs to train for at most
    save_model: int = 100  # None or # of epochs between saving model
    verbose: bool = True
    tensorboard: bool = True
    seed: int = 0
    subset: int = 0  # 1 / (fraction of data to use)
    patience: int = (
        100  # nr. of epochs to continue training when valid score doesn't improve
    )
    adversarial_training: bool = False
    ## model-specific settings
    # SignNet
    pos_enc_dim = 0
    sign_inv_layers = 8
    # ESAN
    policy = "ego_nets_plus"
    num_hops = 3
    sample_fraction = 0.2

    def __init__(
        self,
        model="DropGIN",
        dataset="ogbg-molhiv",
        num_layers=5,
        hidden_units=64,
        dropout=0.5,
        lr=0.01,
        batch_size=32,
        epochs=400,
        save_model=100,
        verbose=True,
        tensorboard=True,
        seed=0,
        subset=0,
        patience=100,
        adversarial_training=False,
        pos_enc_dim=0,
        sign_inv_layers=8,
        policy="ego_nets_plus",
        num_hops=3,
        sample_fraction=0.2,
    ):
        self.model = model
        self.dataset = dataset
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_model = save_model
        self.verbose = verbose
        self.tensorboard = tensorboard
        self.seed = seed
        self.subset = subset
        self.patience = patience
        self.adversarial_training = adversarial_training
        self.pos_enc_dim = pos_enc_dim
        self.sign_inv_layers = sign_inv_layers
        self.policy = policy
        self.num_hops = num_hops
        self.sample_fraction = sample_fraction

    def __str__(self) -> str:
        attrs = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        args_str = ""
        for attr in attrs:
            args_str += f"{attr}={getattr(self, attr)}, "
        return f"Args({args_str[:-2]})"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for the given dataclass"
    )
    parser.add_argument(
        "--model", type=str, default="DropGIN", help="Options are ['DropGIN', 'PPGN']"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        help="Options are ['ogbg-molhiv', 'ogbg-molpcba']",
    )
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save_model", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--adversarial_training", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pos_enc_dim", type=int, default=0)
    parser.add_argument("--sign_inv_layers", type=int, default=8)
    parser.add_argument("--policy", type=str, default="ego_nets_plus")
    parser.add_argument("--num_hops", type=int, default=3)
    parser.add_argument("--sample_fraction", type=float, default=0.2)

    args = parser.parse_args()
    arguments = vars(args)
    return Args(**arguments)


def load_args(model_name, directory="saved_models"):
    dataset_name = model_name.split("_")[1]
    model_directory = osp.join(directory, dataset_name, model_name)
    with open(osp.join(model_directory, f"{model_name}.args"), "rb") as f:
        args = pickle.load(f)
    return args


def store_args(model_name, args, directory="saved_models"):
    dataset_name = model_name.split("_")[1]
    model_directory = osp.join(directory, dataset_name, model_name)
    with open(osp.join(model_directory, f"{model_name}.args"), "wb+") as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
