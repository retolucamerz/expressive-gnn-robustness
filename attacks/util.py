from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from attacks.evaluate import eval_attacked_graphs
import math
from random import random
import itertools
from tqdm import tqdm

def grid_search(
    evaluate_,
    start_values,
    deviations=3,
    start_factor=10,
    factor_decay=0.1,
    stages=2,
    objective="max",
    show_progress=False,
):
    if objective == "max":
        evaluate = evaluate_
    elif objective == "min":
        evaluate = lambda *args: -evaluate_(*args)
    else:
        raise ValueError(f"invalid objective '{objective}'")

    if isinstance(deviations, int):
        deviations = [deviations for _ in range(stages)]

    best_values = start_values
    best_score = evaluate(*start_values)
    factor = start_factor

    for dev in deviations:  # iterate over stages
        values_to_try = [
            [value * math.pow(factor, i + 1) for i in range(dev)]
            + [value * math.pow(factor, -(i + 1)) for i in range(dev)]
            + [value * math.pow(factor, i + 0.5) for i in range(dev)]
            + [value * math.pow(factor, -(i + 0.5)) for i in range(dev)]
            for value in start_values
        ]

        iter = itertools.product(*values_to_try)
        if show_progress:
            iter = tqdm(list(iter))
        for values in iter:
            score = evaluate(*values)
            if score > best_score:
                best_values = values
                best_score = score

        factor = factor_decay * (factor - 1) + 1
        start_values = best_values

    return best_values, best_score


def random_search(
    evaluate_,
    start_values,
    deviations=3,
    start_factor=10,
    factor_decay=0.1,
    stages=2,
    objective="max",
    show_progress=False,
):
    if objective == "max":
        evaluate = evaluate_
    elif objective == "min":
        evaluate = lambda *args: -evaluate_(*args)
    else:
        raise ValueError(f"invalid objective '{objective}'")

    if isinstance(deviations, int):
        deviations = [deviations for _ in range(stages)]

    best_values = None
    best_score = float("-inf")
    factor = start_factor

    for dev in deviations:  # iterate over stages
        values_to_try = [
            [val * math.pow(factor, dev * (2 * random() - 1)) for _ in range(2 * dev)]
            for val in start_values
        ]

        iter = itertools.product(*values_to_try)
        if show_progress:
            iter = tqdm(list(iter))
        for values in iter:
            score = evaluate(*values)
            if score > best_score:
                best_values = values
                best_score = score

        factor = factor_decay * (factor - 1) + 1
        start_values = best_values

    return best_values, best_score


def update_graph(data, clone=True, **kwargs):
    if clone:
        data = data.clone()
    current = {key: data[key] for key in data.keys}
    current.update(kwargs)
    return Data(**current)


class BaseTask(ABC):
    def __init__(self, transform_df=None, data_selection=None, force_update=False, transform_batch=None):
        if data_selection is None:
            self.data_selection = lambda _, __, test_split: test_split
        else:
            self.data_selection = data_selection

        self.transform_df = transform_df
        self.force_update = force_update
        self.transform_batch = transform_batch

    @abstractmethod
    def set_data(self, model, param, results, dataset, device, args):
        pass

    @abstractmethod
    def exists(self, results, param):
        pass


class OriginalScoreAccTask(BaseTask):
    params = [None]

    def __init__(self, batch_size=None, compute_cycle_count=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.compute_cycle_count = compute_cycle_count

    def set_data(self, model, param, recorder, dataset, device, args):
        recorder.update(budget=0)

        batch_size = self.batch_size if self.batch_size is not None else args.batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            graph_results = eval_attacked_graphs(
                model,
                batch.to(device),
                id=batch.id,
                encode=True,
                compute_cycle_count=self.compute_cycle_count,
            )
            recorder.record(**graph_results)

    def exists(self, data, param):
        return not data.empty and not data["id"].isna().any()


select_small = lambda _, __, test: [d for d in test if d.num_edges <= 75]
select_large = lambda _, __, test: [d for d in test if 75 < d.num_edges]

BASELINE_TASKS = lambda transform_df: {
    "original": OriginalScoreAccTask(transform_df=transform_df),
    "small_original": OriginalScoreAccTask(
        transform_df=transform_df, data_selection=select_small
    ),
    "large_original": OriginalScoreAccTask(
        transform_df=transform_df, data_selection=select_large
    ),
}
