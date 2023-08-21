import argparse
import os
import os.path as osp
import traceback
import pandas as pd
import time
from attacks.data_recorder import DataRecorder
from models import load_model
from models.args import load_args


def fetch_data(
    attack_tasks,
    device,
    models_directory="saved_models",
    results_directory="results",
    only_load=False,
    write_to_disk=True,
    load_from_disk=True,
    verbose=True,
):
    ret = {}

    for model_name, tasks in attack_tasks.items():
        ret[model_name] = {}
        model_ret = ret[model_name]

        # only loaded when needed
        model = None
        args = load_args(model_name, directory=models_directory)
        dataset_splits = None
        model_path = osp.join(results_directory, args.dataset, model_name)

        for task_name, task in tasks.items():
            if verbose:
                print(f"===============  {model_name} - {task_name} ===============")
                if task.force_update:
                    print(f"forced update - recomputing all values")
            
            task_path = osp.join(model_path, f"{task_name}.data")
            if osp.exists(task_path) and load_from_disk:
                data = pd.read_pickle(task_path)
                if verbose:
                    print(f"loaded data")
            else:
                if verbose:
                    if load_from_disk:
                        print(f"no data found at '{task_path}'")
                    else:
                        print("not using data from disk")
                data = pd.DataFrame()
                if not osp.exists(model_path):
                    os.mkdir(model_path)

            model_ret[task_name] = data

            missing = []
            for param in task.params:
                if task.force_update or not task.exists(data, param):
                    missing.append(param)
            if not missing:
                continue
            if verbose: print(f"missing params: '{missing}'")
            if only_load:
                continue

            # lazy-load model and dataset
            if model is None:
                model, _, dataset_splits = load_model(model_name, attack=True, directory=models_directory)
                model = model.to(device)

            dataset = task.data_selection(*dataset_splits)
            if verbose: print(f"loaded dataset of size {len(dataset)}")

            # compute values for missing params
            for param in missing:
                if verbose:
                    print(
                        f"====== running param {param} - {time.strftime('%Y%m%d-%H%M%S')}"
                    )
                recorder = DataRecorder(
                    model=model_name,
                    task=task_name,
                    model_type=args.model,
                    dataset=args.dataset,
                    param=param,
                )

                try:
                    task.set_data(model, param, recorder, dataset, device, args)
                except Exception as e:
                    print("AN EXCEPTION OCCURRED")
                    print(e)
                    traceback.print_exc()
                    continue

                df = recorder.dataframe
                if task.transform_df is not None:
                    task.transform_df(df)
                dropped_cols = {
                    "pred",
                    "y",
                    "id",
                    "base_lr",
                    "negative_slope",
                    "repeat",
                    "alpha",
                    "pgd_steps",
                    "n_samples",
                } & set(df.columns)
                if verbose:
                    print(
                        df.drop(
                            dropped_cols,
                            axis=1,
                        ).describe(
                            include=["float32", "float64", "int64"],
                            percentiles=[0.05, 0.5, 0.95],
                        )
                    )

                if write_to_disk:
                    if osp.exists(task_path):
                        data = pd.read_pickle(task_path)
                    else:
                        data = pd.DataFrame()
                    data = pd.concat((data, df))
                    data.to_pickle(task_path)

                model_ret[task_name] = data

    return ret
