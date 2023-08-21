import torch
import torch.nn.functional as F

from datasets.ogb_util import ogb_eval, ogb_eval
from datasets.util import OGB_DATASETS


def ncorrect_binclass(pred, y):
    return (pred.round() == y).sum().item()


def ncorrect_multiclass(pred, y):
    pred = torch.argmax(pred, dim=1)
    return (pred.view(-1) == y.view(-1)).sum().item()


def compute_acc_lbl_diff(model, loader, compute_lbl_diff, n_correct, device):
    correct = 0
    total = 0
    lbl_diff = 0

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        pred = model.predict(batch)

        pred = pred.squeeze()
        y = batch.y.squeeze()
        lbl_diff += compute_lbl_diff(pred, y)
        correct += n_correct(pred, y)
        total += len(y)

    acc = correct / total
    lbl_diff = lbl_diff / total
    return {"score": acc, "lbl_diff": lbl_diff}


class BinaryClassifiactionEvaluator:
    main_metric = "score"
    metric_objective = "max"
    metrics = ["score", "acc"]

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def predict(self, data, *args, **kwargs):
        y = self(data, *args, **kwargs)
        return y if self.training else torch.sigmoid(y)

    def loss(self, pred, y):
        return F.binary_cross_entropy_with_logits(pred.view(-1), y.float().view(-1))

    def eval_metrics(self, loader, device):
        self.eval()
        if self.dataset_name in OGB_DATASETS:
            score, acc, lbl_diff = ogb_eval(
                self,
                loader,
                self.dataset_name,
                ncorrect_binclass,
                device,
            )
            return {"score": score, "acc": acc, "lbl_diff": lbl_diff}
        else:
            return compute_acc_lbl_diff(self, loader, lambda pred, y: (pred - y).abs().sum().item(), ncorrect_binclass, device)

    def compare_metrics(self, m1, m2):
        """returns True iff metrics m2 are better than metrics m1"""
        return m1["score"] <= m2["score"]


class MultiClassifiactionEvaluator:
    main_metric = "score"
    metric_objective = "max"
    metrics = ["score", "acc"]

    @staticmethod
    def compute_lbl_diff(pred, y):
        num_classes = pred.shape[-1]
        y = F.one_hot(y, num_classes=num_classes)
        # pred = F.softmax(pred, dim=-1)
        return (pred - y).abs().sum().item() / num_classes

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def predict(self, data, *args, **kwargs):
        pred = self(data, *args, **kwargs)
        pred = F.softmax(pred, dim=-1)
        return pred

    def loss(self, pred, y):
        return F.cross_entropy(pred, y.squeeze())

    def eval_metrics(self, loader, device):
        self.eval()
        if self.dataset_name in OGB_DATASETS:
            score, acc, lbl_diff = ogb_eval(
                self,
                loader,
                self.dataset_name,
                ncorrect_multiclass,
                device,
                transform_pred=lambda p: torch.argmax(p, dim=1),
                compute_lbl_diff=MultiClassifiactionEvaluator.compute_lbl_diff,
            )
            return {"score": score, "acc": acc, "lbl_diff": lbl_diff}
        else:
            return compute_acc_lbl_diff(self, loader, MultiClassifiactionEvaluator.compute_lbl_diff, ncorrect_multiclass, device)

    def compare_metrics(self, m1, m2):
        """returns True iff metrics m2 are better than metrics m1"""
        return m1["score"] <= m2["score"]


class RegressionEvaluator:
    main_metric = "mae"
    metric_objective = "min"
    metrics = ["mae"]

    def predict(self, data, *args, **kwargs):
        return self(data, *args, **kwargs)

    def loss(self, pred, y):
        return F.l1_loss(pred.squeeze(), y)

    def mae(self, loader, device):
        with torch.no_grad():
            error = 0
            total = 0
            for batch in loader:
                batch = batch.to(device)
                pred = self.predict(batch).view(-1)
                y = batch.y.view(-1)
                error += F.l1_loss(pred, y, reduction="sum").item()
                total += len(y)

        return error / total

    def eval_metrics(self, loader, device):
        self.eval()
        return {"mae": self.mae(loader, device)}

    def compare_metrics(self, m1, m2):
        """returns True iff metrics m2 are better than metrics m1"""
        return m1["mae"] >= m2["mae"]


def with_metrics(cls):
    """this class decorator merges the decorated class with the `evaluator` object passed
    to the new constructor"""
    old_init = cls.__init__

    def __init__(self, *args, evaluator=None, **kwargs):
        old_init(self, *args, **kwargs)

        self.evaluator = evaluator

        if not evaluator is None:
            # this is a hack; don't try this at home
            self.__dict__.update(evaluator.__dict__)
            for name, value in evaluator.__class__.__dict__.items():
                if not name.startswith("__"):
                    if callable(value):
                        setattr(self, name, value.__get__(self, cls))
                    else:
                        # potential problem: this couples both classes via shared ref
                        setattr(self, name, value)

    cls.__init__ = __init__
    return cls
