import pandas as pd
import numpy as np


def is_sequence(arg):
    # anything that has __len__ but exclude strings
    return hasattr(arg, "__len__") and not hasattr(arg, "strip")


class DataRecorder:
    def __init__(self, **kwargs):
        self.reset()

        self.global_data = {}
        self.update(**kwargs)

    def reset(self):
        self.data = {}  # the structure to be filled when recording values
        self.extra_data = {"timestamp": []}  # set by the recorder

        self._df = None  # lazy
        self._df_updated = True

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if is_sequence(val):
                raise ValueError(
                    f"key '{key}' corresponds to a sequence, which is not allowed"
                )

        if not self._df_updated:
            self._build_df()  # flush out all old data previous globals
        self.global_data.update(kwargs)

    def record(self, **kwargs):
        if self.data:
            # check if kwargs and data are compatible
            data_keys = set(self.data.keys())
            kwargs_keys = set(kwargs.keys())
            if data_keys != kwargs_keys:
                raise ValueError(
                    f"passed data with keys {kwargs_keys} is not compatible with previousely passed keys {data_keys}"
                )

        list_data = [val for val in kwargs.values() if is_sequence(val)]
        if list_data:
            n = len(list_data[0])
            if any(len(ls) != n for ls in list_data):
                ls = [key for key, val in kwargs.items() if is_sequence(val)]
                for key in ls:
                    print(f"{key} - {len(kwargs[key])}")
                raise ValueError(f"all lists need to have the same length")
        else:
            n = 1

        for key, val in kwargs.items():
            if key not in self.data:
                self.data[key] = []

            if not is_sequence(val):
                val = np.repeat(val, n)

            self.data[key].append(val)

        timestamp = np.repeat(np.datetime64("now"), n)
        self.extra_data["timestamp"].append(timestamp)

        self._df_updated = False

    def _build_df(self):
        """moves all data stored in `self.data` and `self.extra_data` given the current `self.global_data` to `self._df` and
        clears `self.data` and `self.extra_data`"""
        data = {}
        for key, val in self.data.items():
            data[key] = np.concatenate(val)

        for key, val in self.extra_data.items():
            data[key] = np.concatenate(val)

        n = len(data[list(data.keys())[0]])
        for key, val in self.global_data.items():
            data[key] = np.repeat(val, n)

        new_df = pd.DataFrame(data=data)
        # convert to factors
        for key, type in new_df.dtypes.items():
            if type == "object":
                new_df[key] = new_df[key].astype("category")

        self._df = pd.concat((self._df, new_df))
        self._df_updated = True

        for col in self.data:
            self.data[col] = []
        for col in self.extra_data:
            self.extra_data[col] = []

    @property
    def dataframe(self):
        if not self._df_updated:
            self._build_df()

        return self._df.copy()

    def summarize(self):
        if not self._df_updated:
            self._build_df()

        return self._df.describe(
            include=["float64", "int64"], percentiles=[0.05, 0.5, 0.95]
        )


def compute_classification_metrics(df, y_col="y", pred_col="pred"):
    df["lbl_diff"] = (df[pred_col] - df[y_col]).abs()
    df["missclassified"] = ((df[pred_col] > 0.5) != df[y_col]) + 0


def compute_regression_metrics(df, y_col="y", pred_col="pred", mae_col="mae"):
    df[mae_col] = (df[pred_col] - df[y_col]).abs()
    df["above_y"] = np.maximum(df[pred_col] - df[y_col], 0)
    df["below_y"] = np.maximum(df[y_col] - df[pred_col], 0)

def compute_multiclass_metrics(df, y_col="y", pred_col_prefix="pred"):
    pred_cols = {}
    for col in df.columns:
        if not col.startswith(pred_col_prefix):
            continue
        i = int(col.split("_")[-1])
        pred_cols[i] = col

    pred = np.stack([df[pred_cols[i]] for i in range(len(pred_cols))], axis=-1)
    y = df[y_col]
    pred_class = np.argmax(pred, axis=-1)
    df["pred_class"] = pred_class
    df["missclassified"] = (pred_class!=y) + 0

    onehot_y = np.zeros_like(pred)
    onehot_y[np.arange(len(y)), y] = 1
    df["lbl_diff"] = np.absolute(pred-onehot_y).sum(axis=-1) / pred.shape[-1]


if __name__ == "__main__":
    from numpy.random import rand

    n = 3
    recorder = DataRecorder(experiment="drop_random_edges", budget=2)
    recorder.record(y=(rand(n) > 0.5) + 0, pred=1 * rand(n))
    recorder.record(y=(rand(n) > 0.5) + 0, pred=1 * rand(n))

    recorder.update(budget=3)
    recorder.record(y=(rand(n) > 0.5) + 0, pred=1 * rand(n))

    df = recorder.dataframe
    compute_classification_metrics(df)

    print(df)
    print(df.info())
    print(df.describe(include=["float64", "int64"], percentiles=[0.05, 0.5, 0.95]))
