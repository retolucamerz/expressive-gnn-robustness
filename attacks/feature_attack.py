import torch
import numpy as np
from attacks.bruteforce_attacks import eval_attacked_graphs
from attacks.data_recorder import DataRecorder
import torch.nn.functional as F
from attacks.gradient_attacks import expand_const
from models import create_surrogate_model
from models.util import (
    to_dense_data,
    to_sparse_batch,
)
import math
import torch_scatter
from torch.distributions.categorical import Categorical

use_asserts = False


def create_val_map(vals):
    map = -torch.ones(max(-1, *vals) + 1, dtype=torch.long)
    map[vals] = torch.arange(len(vals))
    return map


def encode_as_distr(tensor, vals_per_attr):
    flat_attr = tensor.view(-1, tensor.shape[-1])

    # re-map values to index in distribution w.r.t. other values of same attribute
    val_idx = torch.zeros_like(flat_attr, device=tensor.device)
    for attr, vals in vals_per_attr.items():
        map = (
            create_val_map(vals)
            .to(tensor.device)
            .unsqueeze(1)
            .repeat(1, flat_attr.shape[0])
        )
        val_idx[..., attr] = torch.gather(
            map, dim=0, index=flat_attr[..., attr].unsqueeze(0)
        ).squeeze()

    # figure out where values of given attribute start
    attr_start = []
    k = 0
    for ls in vals_per_attr.values():
        attr_start.append(k)
        k += len(ls)
    attr_start = (
        torch.tensor(attr_start, device=tensor.device)
        .unsqueeze(0)
        .repeat(val_idx.shape[0], 1)
    )
    index = val_idx + attr_start  # combine to get index into distribution

    # set entries at index to 1
    d = sum(
        len(v) for v in vals_per_attr.values()
    )  # length of the attribute distributions
    tensor_dist = torch.zeros([*tensor.shape[:-1], d], device=tensor.device)
    tensor_dist.view(-1, tensor_dist.shape[-1]).scatter_(index=index, dim=1, value=1)
    return tensor_dist


def embed_distr(distr_tensor, emb_list, vals_per_attr):
    x = torch.tensor(
        [(k, v) for k, vs in vals_per_attr.items() for v in vs]
    ).T  # a (2, d) tensor, where the i-th entry is the attribute and value of index i in a distribution
    attr, val = x[0], x[1]

    distr_tensor = distr_tensor.to(emb_list[0].weight)
    res = []
    for i, emb in enumerate(emb_list):
        select_attr_from_tensor = (attr == i).nonzero().view(-1)
        select_attr_from_emb = val[attr == i].view(-1)
        res.append(
            distr_tensor[..., select_attr_from_tensor]
            @ emb.weight[select_attr_from_emb]
        )

    return torch.stack(res, dim=-1).sum(dim=-1)


def reduce_over_attr(distr, vals_per_attr, reduce="sum"):
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()],
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)
    return torch_scatter.scatter(distr, mask, reduce=reduce)


def normalize_distr(distr, init_distr, vals_per_attr, inplace=True):
    """normalizes `distr` by dividing all entries for each attribute by their sum,
    special case: if all values of some attr have prb. 0, replace by init_"""
    if not inplace:
        distr = distr.clone()
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()],
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)

    # re-scale such that most negative value becomes 0
    min = torch_scatter.scatter(distr, mask, reduce="min")
    min = torch.minimum(min, torch.zeros_like(min))  # don't scale if min>0
    for attr in vals_per_attr.keys():
        distr[..., mask == attr] -= min[..., attr].unsqueeze(-1)

    # divide by sum
    sum = torch_scatter.scatter(distr, mask)
    # handle attributes where all values == 0 by setting to 1/len(vals)
    removed_mask = sum == 0
    sum[removed_mask] = 1

    for attr in vals_per_attr.keys():
        distr[..., mask == attr] /= sum[..., attr].unsqueeze(-1)

        temp = distr[..., mask == attr]
        temp[removed_mask[..., attr]] = init_distr[..., mask == attr][
            removed_mask[..., attr]
        ]
        distr[..., mask == attr] = temp

    return distr

def reset_distr(distr, vals_per_attr, inplace=True):
    if not inplace:
        distr = distr.clone()
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()],
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)

    for attr, vals in vals_per_attr.items():
        distr[..., mask == attr] = 1 / len(vals)

    return distr


def fix_init_value(distr, init_distr, vals_per_attr, inplace=True):
    """normalizes `distr` by replacing the probability of the initial values by 1 - P[other values of same attr]"""
    if not inplace:
        distr = distr.clone()
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()]
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)

    other_sum = torch_scatter.scatter((1 - init_distr) * distr, mask, reduce="sum")
    active_distrs = (init_distr == 1).sum(dim=-1) == len(vals_per_attr)
    other_sum = other_sum[active_distrs]

    if use_asserts: assert ((0 <= other_sum) & (other_sum <= 1.00001)).all()
    init_value_mask = (init_distr == 1) & active_distrs.unsqueeze(
        -1
    )  # to select initial values of active distrs
    distr[init_value_mask] = 1 - other_sum.view(-1)

    return distr


def decrease_distr_budget(distr, init_distr, mu, vals_per_attr, scale_by=1):
    """decreases the budget used by each attr of distr: (B, ..., D) by mu: (B) per batch"""
    # scale up mu
    while len(mu.shape) < len(distr.shape):
        mu = mu.unsqueeze(-1)
    mu = mu.repeat(1, *distr.shape[1:])

    # compute the budget used per distr
    other_sum_ = reduce_over_attr((1 - init_distr) * distr, vals_per_attr, reduce="sum")
    other_sum_[other_sum_ == 0] = 1
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()]
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)
    other_sum = torch.zeros_like(distr)
    for attr, vals in vals_per_attr.items():
        masked_shape = (*distr.shape[:-1], len(vals))
        other_sum[..., mask == attr] = (
            other_sum_[..., attr].unsqueeze(-1).expand(masked_shape)
        )

    ret = (distr * (1 - scale_by * mu / other_sum)).clamp(0, 1)
    fix_init_value(ret, init_distr, vals_per_attr)
    return ret


def decrease_adj_budget(adj, init_adj, mu):
    """decreases the budget used by adj: (B, N, N) by mu: (B) per batch"""
    full_mu = expand_const(mu, adj.shape)
    # need to add value to decrease difference if init_adj==1
    adj_ = adj + (2 * init_adj - 1) * full_mu
    adj_ = adj_.clamp(0, 1)
    return adj_


def _expected_budget(distr, initial_distr, weight=1):
    diff = weight * initial_distr * (1 - distr)
    sum_dims = [i + 1 for i in range(len(diff.shape) - 1)]
    return diff.sum(dim=sum_dims)


def expected_budget(
    x_distr,
    init_x_distr,
    adj,
    init_adj,
    edge_attr_distr,
    init_edge_attr_distr,
    directed=False,
):
    x_budget = _expected_budget(x_distr, init_x_distr)
    adj_budget = (init_adj - adj).abs().sum(dim=(-2, -1))
    edge_attr_budget = _expected_budget(
        edge_attr_distr, init_edge_attr_distr, weight=(init_adj * adj).unsqueeze(-1)
    )
    if not directed:
        adj_budget = adj_budget / 2
        edge_attr_budget = edge_attr_budget / 2
    return x_budget + adj_budget + edge_attr_budget


def sample_from_distr(distr, vals_per_attr):
    mask = torch.concat(
        [attr * torch.ones(len(vals)) for attr, vals in vals_per_attr.items()]
    )
    mask = mask.to(device=distr.device, dtype=torch.int64)

    ret = []
    for attr, vals in vals_per_attr.items():
        sampled = Categorical(distr[..., mask == attr]).sample()
        # re-map from index into vals to original value
        shape = sampled.shape
        map = torch.tensor(vals, device=distr.device)
        sampled = torch.gather(map, dim=0, index=sampled.view(-1)).view(*shape)
        ret.append(sampled)

    return torch.stack(ret, dim=-1)


def is_normalized(distr, vals_per_attr):
    sum = reduce_over_attr(distr, vals_per_attr, reduce="sum")
    return torch.isclose(sum, torch.ones_like(sum))


def compute_adj_grad(f, x_distr, adj, edge_attr_distr, node_mask, directed=False, nan_mask_list=None, target_list=None):
    adj.requires_grad_()
    target = f(x_distr, adj, edge_attr_distr)
    if target_list is not None:
        target_list.append(target)
    grad = torch.autograd.grad(target.sum(), adj)[0]


    # compute noisy gradient when clean grad fails (NaN values)
    nan_mask = grad.isnan().sum(dim=(-2,-1))>0
    if nan_mask_list is not None:
        nan_mask_list.append(nan_mask)
    for noise in [0.001, 0.01, 0.1]:
        if nan_mask.sum()>0:
            sub_x = x_distr[nan_mask]
            sub_adj = adj[nan_mask]
            sub_edge_attr = edge_attr_distr[nan_mask]
            sub_node_mask = node_mask[nan_mask]
            grad[nan_mask] = torch.autograd.grad(f(sub_x, sub_adj, sub_edge_attr, mask=sub_node_mask, batch_mask=nan_mask, noisy=noise).sum(), sub_adj)[0]

            nan_mask = grad.isnan().sum(dim=(-2,-1))>0

    adj.requires_grad_(False)
    if use_asserts: assert not grad.isnan().any()

    if not directed:
        grad = grad + grad.transpose(1, 2)
    # discard diagonal elements
    B, N, _ = adj.shape
    full = list(range(N))
    grad[:, full, full] = 0
    # discard gradient on non-existing nodes
    grad[~node_mask] = 0
    grad.transpose(-2, -1)[~node_mask] = 0

    return grad

def pgd_step(
    f,
    x_distr,
    adj,
    edge_attr_distr,
    node_mask,
    init_x_distr,
    init_adj,
    init_edge_attr_distr,
    vals_per_node_attr,
    vals_per_edge_attr,
    abs_budget,
    directed=False,
    lr=0.001,
    grad_clip=None,
    match_gradients=False,
    record_dict=None,
    update_x=True,
    update_edge_attr="full", # "full" | "negative" | "none"
    target_list=None,
):
    if isinstance(record_dict, dict):
        if not "num_nan_grad" in record_dict:
            record_dict["num_nan_grad"] = []
        nan_mask_list = record_dict["num_nan_grad"]
    else:
        nan_mask_list = None
    
    adj_grad = compute_adj_grad(f, x_distr, adj, edge_attr_distr, node_mask, directed=directed, nan_mask_list=nan_mask_list, target_list=target_list)

    B, N, _ = adj.shape
    full = list(range(N))

    if update_x:
        x_distr.requires_grad_()
        x_grad = torch.autograd.grad(f(x_distr, adj, edge_attr_distr).sum(), x_distr)[0]
        x_distr.requires_grad_(False)

        x_grad[~node_mask] = 0 # discard gradient on non-existing nodes
    else:
        x_grad = torch.zeros_like(x_distr)

    if update_edge_attr in ["full", "negative"]:
        edge_attr_distr.requires_grad_()
        edge_attr_grad = torch.autograd.grad(
            f(x_distr, adj, edge_attr_distr).sum(), edge_attr_distr
        )[0]
        edge_attr_distr.requires_grad_(False)

        if not directed:
            edge_attr_grad = edge_attr_grad + edge_attr_grad.transpose(1, 2)

        edge_attr_grad[:, full, full] = 0 # no self-loops
        # no grad on non-existant nodes
        edge_attr_grad[~node_mask] = 0
        edge_attr_grad.transpose(1, 2)[~node_mask] = 0

        if update_edge_attr == "negative":
            edge_attr_grad[init_adj==1] = 0

    else:
        edge_attr_grad = torch.zeros_like(edge_attr_distr)
    

    if match_gradients:
        sum = (
            x_grad.abs().sum(dim=(1, 2))
            + adj_grad.abs().sum(dim=(1, 2))
            + edge_attr_grad.abs().sum(dim=(1, 2, 3))
        )
        x_grad = 3 * x_grad / sum.unsqueeze(-1).unsqueeze(-1)
        adj_grad = 3 * adj_grad / sum.unsqueeze(-1).unsqueeze(-1)
        edge_attr_grad = (
            3 * edge_attr_grad / sum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

    # gradient clipping
    if not grad_clip is None:
        grad_len_sq = x_grad.square().sum(dim=(1, 2))
        idx = grad_len_sq > grad_clip * grad_clip
        x_grad[idx] *= (grad_clip / grad_len_sq[idx].sqrt()).view(-1, 1, 1)

        grad_len_sq = adj_grad.square().sum(dim=(1, 2))
        idx = grad_len_sq > grad_clip * grad_clip
        adj_grad[idx] *= (grad_clip / grad_len_sq[idx].sqrt()).view(-1, 1, 1)

        grad_len_sq = edge_attr_grad.square().sum(dim=(1, 2, 3))
        idx = grad_len_sq > grad_clip * grad_clip
        edge_attr_grad[idx] *= (grad_clip / grad_len_sq[idx].sqrt()).view(-1, 1, 1, 1)

    if isinstance(record_dict, dict):
        recorded_entries = [
            "x_grad_size",
            "adj_grad_size",
            "edge_attr_grad_size",
            "node_attr_grad",
        ]
        for key in recorded_entries:
            if not key in record_dict:
                record_dict[key] = []

        n_nodes = node_mask.sum(dim=-1)
        num_node_features = len(vals_per_node_attr)
        num_edge_features = len(vals_per_edge_attr)
        record_dict["x_grad_size"].append(x_grad.abs().sum(dim=(-2, -1)) / n_nodes / num_node_features)
        record_dict["adj_grad_size"].append(adj_grad.abs().sum(dim=(-2, -1)) / n_nodes / n_nodes)
        record_dict["edge_attr_grad_size"].append(
            edge_attr_grad.abs().sum(dim=(-3, -2, -1)) / n_nodes / n_nodes / num_edge_features
        )

        node_attr_grad = reduce_over_attr(
            x_grad.abs(), vals_per_node_attr, reduce="max"
        ).sum(dim=1)
        record_dict["node_attr_grad"].append(node_attr_grad)


    # update
    x_distr += lr * x_grad
    adj += lr * adj_grad
    edge_attr_distr += lr * edge_attr_grad

    # PROJECTION
    x_distr = torch.clamp(x_distr, min=0, max=1)
    normalize_distr(x_distr, init_x_distr, vals_per_node_attr)
    adj = torch.clamp(adj, min=0, max=1)
    edge_attr_distr = torch.clamp(edge_attr_distr, min=0, max=1)
    normalize_distr(edge_attr_distr, init_edge_attr_distr, vals_per_edge_attr)

    if use_asserts:
        assert (x_distr[~node_mask]==0).all()
        assert is_normalized(x_distr[node_mask], vals_per_node_attr).all()
        valid_edge_mask = torch.einsum('bi,bj->bij', (node_mask, node_mask))
        valid_edge_mask[:,range(N),range(N)] = 0
        assert (adj[~valid_edge_mask]==0).all()
        assert (edge_attr_distr[~valid_edge_mask]==0).all()
        assert is_normalized(edge_attr_distr[valid_edge_mask], vals_per_edge_attr).all()

    difference = expected_budget(
        x_distr,
        init_x_distr,
        adj,
        init_adj,
        edge_attr_distr,
        init_edge_attr_distr,
        directed=directed,
    )
    in_budget = difference <= abs_budget

    n_out_of_budget = B - in_budget.sum()
    if n_out_of_budget > 0:
        budget = abs_budget[~in_budget]

        l = torch.zeros(n_out_of_budget).to(adj.device)
        h = torch.ones(n_out_of_budget).to(adj.device)
        for _ in range(18):  # reduces error to at most 2^(-18)
            mu = (l + h) / 2

            x_distr_ = decrease_distr_budget(
                x_distr[~in_budget], init_x_distr[~in_budget], mu, vals_per_node_attr
            )
            # only change distr of initial edges, proportional to adj_
            scale_by = adj[~in_budget] * init_adj[~in_budget]
            edge_attr_distr_ = decrease_distr_budget(
                edge_attr_distr[~in_budget],
                init_edge_attr_distr[~in_budget],
                mu,
                vals_per_edge_attr,
                scale_by=scale_by.unsqueeze(-1),
            )
            adj_ = decrease_adj_budget(adj[~in_budget], init_adj[~in_budget], mu)
            used_budget = expected_budget(
                x_distr_,
                init_x_distr[~in_budget],
                adj_,
                init_adj[~in_budget],
                edge_attr_distr_,
                init_edge_attr_distr[~in_budget],
                directed=directed,
            )

            l[used_budget >= budget] = mu[used_budget >= budget]
            h[used_budget < budget] = mu[used_budget < budget]

        mu = (l + h) / 2
        x_distr[~in_budget] = decrease_distr_budget(
            x_distr[~in_budget], init_x_distr[~in_budget], mu, vals_per_node_attr
        )
        adj[~in_budget] = decrease_adj_budget(adj[~in_budget], init_adj[~in_budget], mu)
        scale_by = adj_ * init_adj[~in_budget]
        edge_attr_distr[~in_budget] = decrease_distr_budget(
            edge_attr_distr[~in_budget],
            init_edge_attr_distr[~in_budget],
            mu,
            vals_per_edge_attr,
            scale_by=scale_by.unsqueeze(-1),
        )

    if use_asserts:
        assert (x_distr[~node_mask]==0).all()
        assert is_normalized(x_distr[node_mask], vals_per_node_attr).all()
        valid_edge_mask = torch.einsum('bi,bj->bij', (node_mask, node_mask))
        valid_edge_mask[:,range(N),range(N)] = 0
        assert (adj[~valid_edge_mask]==0).all()
        assert (edge_attr_distr[~valid_edge_mask]==0).all()
        assert is_normalized(edge_attr_distr[valid_edge_mask], vals_per_edge_attr).all()

    return x_distr, adj, edge_attr_distr


def create_x_distr(x, vals_per_node_attr, node_mask, device):
    num_node_vals = sum(len(vals) for vals in vals_per_node_attr.values())
    x_distr = torch.zeros((*x.shape[:-1], num_node_vals), device=device)
    x_distr[node_mask] = encode_as_distr(x[node_mask], vals_per_node_attr)
    return x_distr

def create_edge_attr_distr(edge_attr, vals_per_edge_attr, init_adj, node_mask, device):
    num_edge_vals = sum(len(vals) for vals in vals_per_edge_attr.values())
    edge_attr_distr = torch.zeros(
        (*edge_attr.shape[:-1], num_edge_vals), device=device
    )
    edge_attr_distr[init_adj == 1] = encode_as_distr(
        edge_attr[init_adj == 1], vals_per_edge_attr
    )
    edge_attr_distr[init_adj == 0] = reset_distr(
        edge_attr_distr[init_adj == 0], vals_per_edge_attr, inplace=False
    )  # uniform distr over missing edges

    N = init_adj.shape[1]
    valid_edge_mask = torch.einsum('bi,bj->bij', (node_mask, node_mask)) # batched outer product
    valid_edge_mask[:,range(N),range(N)] = 0
    edge_attr_distr[~valid_edge_mask] = 0
    
    return edge_attr_distr

def AttrPGD(
    model,
    batch_,
    target,
    budget,
    vals_per_node_attr,
    vals_per_edge_attr,
    node_encoders,  # list of torch.Embedding's
    edge_encoders,  # list of torch.Embedding's
    pgd_steps=50,
    surrogate_model=None,
    directed=False,
    base_lr=0.01,
    grad_clip=None,
    seed=0,
    n_samples=100,
    relative_budget=False,
    limit_edgeattr_by_adj=False,
    update_x=True,
    update_edge_attr="full", # "full" | "negative" | "none"
    transform_batch=None,
    recorder: DataRecorder = None,
    target_list=None,
    clean_target_list=None,
    compute_cycle_count=False,
):
    device = batch_.edge_index.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    if surrogate_model is None:
        surrogate_model = create_surrogate_model(model, negative_slope=0)

    with torch.no_grad():
        init_x, node_mask, init_adj, init_edge_attr = to_dense_data(batch_.clone())
        B, N, _ = init_adj.shape

        # create x distributions
        x_shape = init_x.shape
        while len(init_x.shape) < 3:
            init_x = init_x.unsqueeze(-1)
        init_x_distr = create_x_distr(init_x, vals_per_node_attr, node_mask, device)

        # create edge_attr distributions
        edge_attr_shape = init_edge_attr.shape
        while len(init_edge_attr.shape) < 4:
            init_edge_attr = init_edge_attr.unsqueeze(-1)
        init_edge_attr_distr = create_edge_attr_distr(init_edge_attr, vals_per_edge_attr, init_adj, node_mask, device)

        # setup UNDIRECTED budget
        batch_size = len(batch_.y)
        if relative_budget:
            num_edges = np.array([batch_[i].num_edges for i in range(batch_size)])
            if not directed:
                num_edges = num_edges / 2
            abs_budget = torch.tensor([math.ceil(budget * m) for m in num_edges])
        else:
            abs_budget = torch.tensor([budget for _ in range(batch_size)])
        abs_budget = abs_budget.to(device)

    record_per_step = {}

    x_distr = init_x_distr.clone()
    adj = init_adj.clone()
    edge_attr_distr = init_edge_attr_distr.clone()

    def f(x_distr, adj, edge_attr_distr, mask=node_mask, model=surrogate_model, batch=batch_, batch_mask=None, **kwargs):
        x = embed_distr(x_distr, node_encoders, vals_per_node_attr)
        edge_attr = embed_distr(edge_attr_distr, edge_encoders, vals_per_edge_attr)
        if batch_mask is not None:
            batch = batch.clone()
            batch.y = batch.y[batch_mask]
        if use_asserts:
            B, N, _ = adj.shape
            assert (x[~mask]==0).all()
            assert (x_distr[~mask]==0).all()
            assert is_normalized(x_distr[mask], vals_per_node_attr).all()
            valid_edge_mask = torch.einsum('bi,bj->bij', (mask, mask))
            valid_edge_mask[:,range(N),range(N)] = 0
            assert (adj[~valid_edge_mask]==0).all()
            assert ((0<=adj) & (adj<=1)).all()
            assert (edge_attr[~valid_edge_mask]==0).all()
            assert (edge_attr_distr[~valid_edge_mask]==0).all()
            assert is_normalized(edge_attr_distr[valid_edge_mask], vals_per_edge_attr).all()
        return target(
            model, batch, x=x, adj=adj, edge_attr=edge_attr, mask=mask, encode=False, limit_edgeattr_by_adj=limit_edgeattr_by_adj, **kwargs
        )
    
    clean_surrogate_model = create_surrogate_model(model, negative_slope=0)

    for step in range(1, pgd_steps + 1):
        if clean_target_list is not None:
            with torch.no_grad():
                clean_target_list.append(f(x_distr, adj, edge_attr_distr, model=clean_surrogate_model))

        lr = base_lr * budget / math.sqrt(step)
        x_distr, adj, edge_attr_distr = pgd_step(
            f,
            x_distr,
            adj,
            edge_attr_distr,
            node_mask,
            init_x_distr,
            init_adj,
            init_edge_attr_distr,
            vals_per_node_attr,
            vals_per_edge_attr,
            abs_budget,
            directed=directed,
            lr=lr,
            grad_clip=grad_clip,
            match_gradients=False,
            record_dict=record_per_step,
            update_x=update_x,
            update_edge_attr=update_edge_attr,
            target_list=target_list,
        )

    if target_list is not None:
        with torch.no_grad():
            target_list.append(f(x_distr, adj, edge_attr_distr))
    if clean_target_list is not None:
        with torch.no_grad():
            clean_target_list.append(f(x_distr, adj, edge_attr_distr, model=clean_surrogate_model))

    if recorder:
        final_target = f(x_distr, adj, edge_attr_distr)
        final_clean_target = f(x_distr, adj, edge_attr_distr, model=clean_surrogate_model)

    if recorder:
        exp_budget = expected_budget(
            x_distr, init_x_distr, adj, init_adj, edge_attr_distr, init_edge_attr_distr
        )
        exp_budget = exp_budget.detach().cpu().numpy()

    if use_asserts:
        B, N, _ = adj.shape
        assert (x_distr[~node_mask]==0).all()
        assert is_normalized(x_distr[node_mask], vals_per_node_attr).all()
        valid_edge_mask = torch.einsum('bi,bj->bij', (node_mask, node_mask))
        valid_edge_mask[:,range(N),range(N)] = 0
        assert (adj[~valid_edge_mask]==0).all()
        assert ((0<=adj) & (adj<=1)).all()
        assert (edge_attr_distr[~valid_edge_mask]==0).all()
        assert is_normalized(edge_attr_distr[valid_edge_mask], vals_per_edge_attr).all()
        assert not adj.isnan().any()
        assert not x_distr.isnan().any()
        assert not edge_attr_distr.isnan().any()

    # sample graphs from distributions
    B, N, _ = init_adj.shape
    best_val = torch.full((B,), float("-inf")).to(device)
    best_x = init_x.clone().view(x_shape)
    best_adj = init_adj.clone().to(dtype=torch.int64)
    best_edge_attr = init_edge_attr.clone().view(edge_attr_shape)
    successful_samples = torch.zeros(B).to(device)
    with torch.no_grad():
        for _ in range(n_samples):
            if update_x:
                x_sample = torch.zeros_like(init_x, device=device)
                x_sample[node_mask] = sample_from_distr(
                    x_distr[node_mask], vals_per_node_attr
                )
            else:
                x_sample = init_x
            adj_sample = torch.bernoulli(adj).to(dtype=torch.int64)
            if not directed:
                adj_sample = torch.tril(adj_sample, diagonal=-1)
                adj_sample = adj_sample + adj_sample.transpose(-2, -1)
            if update_edge_attr:
                edge_attr_sample = torch.zeros_like(init_edge_attr, device=device)
                valid_edge_mask = torch.einsum('bi,bj->bij', (node_mask, node_mask))
                valid_edge_mask[:,range(N),range(N)] = 0
                edge_attr_sample[valid_edge_mask] = sample_from_distr(edge_attr_distr[valid_edge_mask], vals_per_edge_attr)
                if not directed:
                    edge_attr_sample = torch.tril(
                        edge_attr_sample.transpose(-3, -1), diagonal=-1
                    ).transpose(-3, -1)
                    edge_attr_sample = edge_attr_sample + edge_attr_sample.transpose(-3, -2)
                edge_attr_sample = adj_sample.unsqueeze(-1) * edge_attr_sample
            else:
                edge_attr_sample = init_edge_attr

            if use_asserts and not directed:
                assert torch.all(adj_sample == adj_sample.transpose(-2, -1))
                assert torch.all(edge_attr_sample == edge_attr_sample.transpose(1, 2))

            x_diff = (x_sample != init_x).sum(dim=(-2, -1))
            adj_diff = (adj_sample != init_adj).sum(dim=(-2, -1))
            in_both = (adj_sample * init_adj).unsqueeze(-1)
            edge_attr_diff = (in_both*edge_attr_sample != in_both*init_edge_attr).sum(dim=(-3, -2, -1))
            if not directed:
                if use_asserts: assert ((adj_diff % 2 == 0) & (edge_attr_diff % 2 == 0)).all()
                adj_diff = adj_diff / 2
                edge_attr_diff = edge_attr_diff / 2
            difference = x_diff + adj_diff + edge_attr_diff

            in_budget = difference <= abs_budget
            successful_samples += in_budget

            x_sample = x_sample.view(x_shape)
            edge_attr_sample = edge_attr_sample.view(edge_attr_shape)

            sparse_data = to_sparse_batch(
                x_sample,
                adj_sample,
                edge_attr_sample,
                batch_.y,
                node_mask=node_mask,
            ).to(device)

            val = target(
                model,
                sparse_data,
                encode=True,
            )
            better = val > best_val

            update = torch.logical_and(in_budget, better)
            best_val[update] = val[update]
            best_x[update] = x_sample[update]
            best_adj[update] = adj_sample[update]
            best_edge_attr[update] = edge_attr_sample[update]

    if recorder:
        successful_samples = (successful_samples / n_samples).detach().cpu().numpy()

    # evaluate best sampled graphs
    sparse_batch = to_sparse_batch(
        best_x,
        best_adj,
        best_edge_attr,
        batch_.y,
        node_mask=node_mask,
    ).to(device)

    if recorder:
        results = eval_attacked_graphs(
            model,
            sparse_batch,
            directed=directed,
            init_x=init_x,
            init_adj=init_adj,
            init_edge_attr=init_edge_attr,
            encode=True,
            id=batch_.id,
            compute_cycle_count=compute_cycle_count,
            transform=transform_batch,
        )

        # sum over all pgd steps
        for key, values in record_per_step.items():
            record_per_step[key] = torch.stack(record_per_step[key], dim=-1).sum(dim=-1)

        for attr in vals_per_node_attr.keys():
            key = f"node_attr_grad_a{attr}"
            record_per_step[key] = record_per_step["node_attr_grad"][:, attr]
        record_per_step["node_attr_grad"] = record_per_step["node_attr_grad"].sum(
            dim=-1
        )

        for key, values in record_per_step.items():
            record_per_step[key] = (
                (values / pgd_steps).to(dtype=torch.float32).detach().cpu().numpy()
            )

        final_target = final_target.detach().cpu().numpy()
        final_clean_target = final_clean_target.detach().cpu().numpy()

        recorder.record(
            expected_budget=exp_budget,
            successful_samples=successful_samples,
            final_target=final_target,
            final_clean_target=final_clean_target,
            **results,
            **record_per_step,
        )

    return sparse_batch

