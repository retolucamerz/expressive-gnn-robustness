import argparse
from models.args import Args, parse_args
from dataclasses import asdict, replace
from train import train_model
import wandb

base_params = {
    'hidden_units': {'values': [300]},
    'num_layers': {'values': [5]},
    'lr': {'values': [0.01, 0.001]},
    'batch_size': {'values': [32, 64]},
    'dropout': {'values': [0.5, 0]},
    'epochs': {'values': [300]},
}

dropgine_molhiv_params = {
    **base_params,
    'hidden_units': {'values': [150]},
}

ppgn_zinc_params = {
    **base_params,
    'lr': {'values': [0.001, 0.0001]},
}

ppgn_imdbbin_params = {
    **ppgn_zinc_params,
    'batch_size': {'values': [16]},
}

ppgn_molhiv_params = {
    **base_params,
    'hidden_units': {'values': [64]},
    'batch_size': {'values': [16]},
    'lr': {'values': [0.001, 0.0001]},
}

signnet_params = {
    **base_params,
    'num_layers': {'values': [16]},
    'sign_inv_layers': {'values': [8]},
    'hidden_units': {'values': [95]},
    'batch_size': {'values': [128]},
    'dropout': {'values': [0]},
    'epochs': {'values': [1000]},
    'patience': {'values': [200]}
}

signnet_zinc_params = {
    **signnet_params,
    'pos_enc_dim': {'values': [37]},
}

signnet_molhiv_params = {
    **signnet_params,
    'pos_enc_dim': {'values': [30, 50]},
    'batch_size': {'values': [64]},
}

signnet_tu_params = {
    **signnet_params,
    'pos_enc_dim': {'values': [15, 30]},
}

esan_params = {
    **base_params,
    'hidden_units': {'values': [64]},
    'batch_size': {'values': [64, 128]},
    'policy': {'values': ["ego_nets_plus"]},
    'num_hops': {'values': [3]},
    'sample_fraction': {'values': [0.2]},
}


def get_sweep_config(model, dataset, adversarial_training=False):
    if model=="DropGINE" and "molhiv" in dataset:
        params = dropgine_molhiv_params
    elif "GIN" in model or model=="Baseline":
        params = base_params
    elif model=="PPGN":
        if dataset == "IMDB-BINARY":
            params = ppgn_imdbbin_params
        elif "molhiv" in dataset:
            params = ppgn_molhiv_params
        else: # ZINC12k, IMDB-MULTI, MUTAG
            params = ppgn_zinc_params
    elif model=="SignNet":
        params = signnet_zinc_params if "ZINC" in dataset else signnet_tu_params if dataset.startswith("IMDB") or dataset=="MUTAG" else signnet_molhiv_params
    elif model=="ESAN":
        params = esan_params
    else:
        raise ValueError(f"unknown model type '{model}'")
    
    if adversarial_training:
        params |= {
            "adversarial_training": {"values": [adversarial_training]}
        }

    goal = "maximize" if dataset=="ogbg-molhiv" else "minimize"
    metric_name = "valid_mae" if dataset=="ZINC12k" else "valid_score"
    return {
        'method': 'grid',
        'name': f'{model}_{dataset}',
        'metric': {'goal': goal, 'name': metric_name},
        'parameters': params
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--adversarial_training", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    sweep_id = parser.parse_args().sweep_id


    if sweep_id is None:
        # start sweep
        sweep_config = get_sweep_config(args.model, args.dataset, adversarial_training=args.adversarial_training)
        sweep_config['parameters']['model'] = {'values': [args.model]}
        sweep_config['parameters']['dataset'] = {'values': [args.dataset]}
        
        print(f"------- {args.model} {args.dataset} -------")
        print(args)
        sweep_id = wandb.sweep(
            sweep=sweep_config, 
            project='expressive-gnn-robustness'
        )
        print(sweep_config)
        print()
        input()
    
    else:
        # add agent to sweep
        from functools import partial

        def main_func(args):
            wandb.init()
            args = replace(args, **wandb.config)
            args.seed = 0
            train_model(args, use_wandb_sweep=True)

        args = Args()

        wandb.agent(sweep_id, project='expressive-gnn-robustness', function=partial(main_func, args))
