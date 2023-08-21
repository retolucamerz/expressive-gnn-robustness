"""
THE CODE IN THIS DIRECTORY IS A MODIFIED VERSION OF
https://github.com/beabevi/ESAN

"""


from datasets import get_encoders, get_evaluator
from models.ESAN.sparse_conv import (
    GCNConv,
    GINConv,
    GINEConv,
)
from models.ESAN.sparse_models import (
    GNN,
    DSSnetwork,
    DSnetwork,
)
from dataclasses import dataclass


@dataclass
class ESANArgs:
    model: str = "dss"
    dataset: str = "ZINC"
    hidden_units: int = 10
    num_layer: int = 5
    policy: str = "ego_nets_plus"
    num_hops: int = 3
    sample_fraction: float = 0.5
    gnn_type: str = "gine"
    jk: str = "last"
    channels: int = 64
    seed: int = 0

    @staticmethod
    def from_model_args(model_args, **kwargs):
        dataset = model_args.dataset
        if dataset == "ZINC12k":
            dataset = "ZINC"

        esan_args = ESANArgs(
            gnn_type="gine",
            dataset=dataset,
            hidden_units=model_args.hidden_units,
            num_layer=model_args.num_layers,
            policy=model_args.policy,
            num_hops=model_args.num_hops,
            sample_fraction=model_args.sample_fraction,
            seed=model_args.seed,
            **kwargs,
        )
        return esan_args

def get_esan_model(args, out_dim):
    evaluator = get_evaluator(args.dataset)

    node_emb_dim = args.hidden_units if args.policy != "ego_nets_plus" else args.hidden_units - 2
    atom_encoder, bond_encoder = get_encoders(
        args.dataset, node_emb_dim, args.hidden_units, args.seed
    )

    kwargs = {
        "evaluator": evaluator,
        "atom_encoder": atom_encoder,
        "bond_encoder": bond_encoder,
        "policy": args.policy,
        "num_hops": args.num_hops,
        "sample_fraction": args.sample_fraction,
    }

    if args.model == "deepsets":
        subgraph_gnn = GNN(
            gnn_type=args.gnn_type,
            num_tasks=out_dim,
            num_layer=args.num_layer,
            in_dim=args.hidden_units,
            emb_dim=args.hidden_units,
            drop_ratio=args.drop_ratio,
            JK=args.jk,
            graph_pooling="sum" if args.gnn_type != "gin" else "mean",
        )
        model = DSnetwork(
            subgraph_gnn=subgraph_gnn,
            channels=args.channels,
            num_tasks=out_dim,
            invariant=args.dataset.contains("ZINC"),
            **kwargs,
        )

    elif args.model == "dss":
        if args.gnn_type == "gin":
            GNNConv = GINConv
        if args.gnn_type == "gine":
            GNNConv = GINEConv
        elif args.gnn_type == "gcn":
            GNNConv = GCNConv
        else:
            raise ValueError("Undefined GNN type called {}".format(args.gnn_type))

        model = DSSnetwork(
            num_layers=args.num_layer,
            in_dim=args.hidden_units,
            emb_dim=args.hidden_units,
            num_tasks=out_dim,
            GNNConv=GNNConv,
            **kwargs,
        )

    # elif args.model == "gnn":
    #     num_random_features = int(args.random_ratio * args.emb_dim)
    #     model = GNNComplete(
    #         gnn_type=args.gnn_type,
    #         num_tasks=out_dim,
    #         num_layer=args.num_layer,
    #         in_dim=in_dim,
    #         emb_dim=args.hidden_units,
    #         policy=args.policy,
    #         num_hops=args.num_hops,
    #         sample_fraction=args.sample_fraction,
    #         JK=args.jk,
    #         graph_pooling="sum" if args.gnn_type != "gin" else "mean",
    #         num_random_features=num_random_features,
    #         **kwargs,
    #     )

    else:
        raise ValueError("Undefined model type called {}".format(args.model))

    return model
