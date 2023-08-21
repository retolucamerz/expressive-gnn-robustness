from collections import OrderedDict
import seaborn as sns
import numpy as np
import pandas as pd



model_types = ['GINE', 'DropGINE', 'PPGN', 'SignNet', 'ESAN', 'GIN', 'meanGINE']
model_types_adv = ['GINE-adv', 'DropGINE-adv', 'PPGN-adv', 'SignNet-adv', 'ESAN-adv', 'GIN-adv', 'meanGINE-adv']
palette = {model: color for model, color in zip(model_types, sns.color_palette())} | {model: color for model, color in zip(model_types_adv, sns.color_palette())}

line_type = {
    'GINE': '',
    'GIN': '',
    'meanGINE': '',
    'DropGINE': '',
    'PPGN': '',
    'SignNet': '',
    'ESAN': '',
    'GINE-adv': (4, 1.5),
    'GIN-adv': (4, 1.5),
    'meanGINE-adv': (4, 1.5),
    'DropGINE-adv': (4, 1.5),
    'PPGN-adv': (4, 1.5),
    'SignNet-adv': (4, 1.5),
    'ESAN-adv': (4, 1.5),
}

def aggregate_over_splits(
    data, # dataframe
    group_by, # a pandas series indicating which rows in `data` belong to which group
    shared_cols,
    target_col,
    objective = "max",
):
    splits = OrderedDict({key: data[group_by==key].copy() for key in set(group_by)})
    for key, split in splits.items():
        split.reset_index(drop=True, inplace=True)
        split["original_key"] = key

    for col in shared_cols:
        split_list = list(splits.values())
        split_keys = list(splits.keys())

        for k1, k2, s1, s2 in zip(split_keys, split_keys[1:], split_list, split_list[1:]):
            if len(s1) != len(s2):
                print(k1, k2)
                print(len(s1), len(s2))
            assert len(s1) == len(s2)
            if not (s1[col]==s2[col]).all():
                raise ValueError(f"col '{col}' doesn't match on split '{k1}'/'{k2}'")

    some_split = next(iter(splits.values()))
    ret = some_split.copy()

    for i, (key, split) in enumerate(splits.items()):
        if objective=="max":
            mask = ret[target_col] < split[target_col]
        elif objective=="min":
            mask = ret[target_col] > split[target_col]

        for col in data.columns:
            if col not in shared_cols:
                ret[col] = ret[col].mask(mask, other=split[col])

    return ret

def remove_unused_categories(df):
    df = df.copy()
    for col in df.columns:
        if not df[col].dtype.name=="category":
            continue
        df[col] = df[col].cat.remove_unused_categories()
    return df


def group(df, by=[]):
    df_num = df._get_numeric_data()
    df_num[by] = df[by]
    return df_num.groupby(by)


def remove_top_percentage(df, precentage, cols, groupby):
    df = df.copy()
    DFs = []
    for value in set(groupby):
        mask = groupby == value

        if mask.sum()<=3:
            raise ValueError(f"value '{value}' has only {mask.sum()} associated elements")

        subdf = df[mask].copy().reset_index()
        some_col = cols[0]
        cutoff = subdf[some_col].quantile(q=1-precentage)
        keep = subdf[some_col] < cutoff
        subdf = subdf[keep]
        DFs.append(subdf)

    return pd.concat(DFs)


def groupby_to_table(grouped, round_to=3):
    mean = grouped.mean().round(round_to)
    std = grouped.std().round(round_to)
    for col in mean.columns:
        mean[col] = "$" + mean[col].astype(str) + " \pm " + std[col].astype(str) + "$"
    return mean.reset_index()

def rename_col(df, old_name, new_name):
    df[new_name] = df[old_name]
    return df.drop(columns=old_name)

def concatenate(dfs):
    """Concatenate while preserving categorical columns.

    NB: We change the categories in-place for the input dataframes"""
    from pandas.api.types import union_categoricals
    import pandas as pd
    # Iterate on categorical columns common to all dfs
    for col in set.intersection(
        *[
            set(df.select_dtypes(include='category').columns)
            for df in dfs
        ]
    ):
        # Generate the union category across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    df = pd.concat(dfs)
    return df.reset_index(drop=True)

def append_model_type(df):
    model_types = [model_name.split("_")[0] for model_name in df["model"].astype("str")]
    df["model_type"] = [model_type.replace("MeanAggrGINE", "meanGINE") for model_type in model_types]


def compute_baseline_score(df_per_model, baseline_value, target_metric, below=True):
    df = df_per_model[["model", "budget", target_metric]].copy()
    if below:
        df[target_metric] = np.maximum(0, df[target_metric] - baseline_value)
    else:
        df[target_metric] = np.maximum(0, baseline_value - df[target_metric])

    budgets = sorted(list(set(df_per_model["budget"])))
    total_weight = 0
    prev_budget = 2*budgets[0] - budgets[1]
    for budget in budgets:
        total_weight += budget - prev_budget

        mask = df["budget"]==budget
        mae = df[target_metric].copy()
        mae[mask] = (budget - prev_budget)*mae[mask]
        df[target_metric] = mae

        prev_budget = budget

    df[target_metric] = df[target_metric]/total_weight

    df = pd.DataFrame(df.groupby(by=["model"]).sum()[target_metric]).reset_index()
    append_model_type(df)
    return df