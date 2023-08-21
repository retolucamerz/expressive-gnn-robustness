from models import create_surrogate_model
from models.ESAN.sparse_models import DSSnetwork
import torch
import numpy as np
from attacks.bruteforce_attacks import eval_attacked_graphs
from attacks.data_recorder import DataRecorder
from models.util import (
    to_dense_data,
    to_sparse_batch,
)
import math


def expand_const(x, new_shape):
    old_dims = len(x.shape)
    view_shape = list(x.shape) + [1 for _ in range(len(new_shape) - old_dims)]
    return x.view(view_shape).expand(new_shape)


def compute_adj_grad(f, x, adj, edge_attr, node_mask, directed=False, nan_mask_list=None, target_list=None):
    adj.requires_grad_()
    target = f(adj, edge_attr)
    grad = torch.autograd.grad(target.sum(), adj)[0]

    if target_list is not None:
        target_list.append(target)

    # compute noisy gradient when clean grad fails (NaN values)
    nan_mask = grad.isnan().sum(dim=(-2,-1))>0
    if nan_mask_list is not None:
        nan_mask_list.append(nan_mask)

    for noise in [0.001, 0.01, 0.1]:
        if nan_mask.sum()>0:
            sub_x = x[nan_mask]
            sub_adj = adj[nan_mask]
            sub_edge_attr = edge_attr[nan_mask]
            sub_node_mask = node_mask[nan_mask]
            grad[nan_mask] = torch.autograd.grad(f(sub_adj, sub_edge_attr, x=sub_x, mask=sub_node_mask, batch_mask=nan_mask, noisy=noise).sum(), sub_adj)[0]

            nan_mask = grad.isnan().sum(dim=(-2,-1))>0

    # give up on graphs where 0.1 noise doesn't help
    if nan_mask.sum()>0:
        grad[nan_mask] = torch.zeros_like(grad[nan_mask]).to(grad.device)

    adj.requires_grad_(False)

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
    budget,
    x,
    adj,
    edge_attr,
    node_mask,
    init_adj,
    directed=False,
    lr=0.001,
    grad_clip=None,
    nan_mask_list=None,
    target_list=None,
    ):
    B, N, _ = adj.shape
    adj = adj.clone()

    grad = compute_adj_grad(f, x, adj, edge_attr, node_mask, directed=directed, nan_mask_list=nan_mask_list, target_list=target_list)

    # gradient clipping
    if not grad_clip is None:
        grad_len_sq = grad.square().sum(dim=(1, 2))
        idx = grad_len_sq > grad_clip * grad_clip
        grad[idx] *= (grad_clip / grad_len_sq[idx].sqrt()).view(-1, 1, 1)

    assert not torch.isnan(grad).any()

    # UPDATE
    adj += lr * grad

    # PROJECTION
    adj = torch.clamp(adj, min=0, max=1)

    difference = (adj - init_adj).abs()
    in_budget = difference.sum(dim=(-2, -1)) <= budget

    def reduce_adj(adj, init_adj, mu):
        full_mu = expand_const(mu, adj.shape)
        # need to add value to decrease difference if init_adj==1
        adj_ = adj + (2 * init_adj - 1) * full_mu
        adj_ = adj_.clamp(0, 1)
        return adj_

    # out-of-budget graphs: reduce difference from init adj to fit in budget
    #   needed change is determined via bisection
    n_out_of_budget = B - in_budget.sum()
    if n_out_of_budget > 0:
        budget = budget[~in_budget]

        l = torch.zeros(n_out_of_budget).to(adj.device)
        h = torch.ones(n_out_of_budget).to(adj.device)
        for _ in range(18):  # reduces error to at most 2^(-18)
            mu = (l + h) / 2

            adj_ = reduce_adj(adj[~in_budget], init_adj[~in_budget], mu)
            used_budget = (adj_ - init_adj[~in_budget]).abs().sum(dim=(-2, -1))

            l[used_budget >= budget] = mu[used_budget >= budget]
            h[used_budget < budget] = mu[used_budget < budget]

        mu = ((l + h) / 2).clamp(0, 1)
        full_mu = expand_const(mu, [n_out_of_budget, N, N])
        adj[~in_budget] = (
            adj[~in_budget] + (2 * init_adj[~in_budget] - 1) * full_mu
        ).clamp(0, 1)

    return adj


def fill_edge_attributes(batch, adj, edge_attr, directed=False):
    """randomly samples misisng edge features from the already present edge features"""
    B, N, _ = adj.shape
    full_edge_attr = edge_attr.clone()

    for i in range(B):
        n_missing = (adj[i] == 0).sum()
        attrs = batch[i].edge_attr
        num_edges = attrs.shape[0]
        idx = torch.randint(num_edges, (n_missing,))
        attr_per_graph = full_edge_attr[i]
        mask = (adj[i] == 0).unsqueeze(-1).repeat(1, 1, edge_attr.shape[-1])
        attr_per_graph[mask] = attrs[idx].view(-1)
        full_edge_attr[i] = attr_per_graph

    full_edge_attr[:, range(N), range(N)] = 0
    if not directed:
        full_edge_attr = torch.tril(
            full_edge_attr.transpose(1, -1), diagonal=-1
        ).transpose(1, -1)
        full_edge_attr = full_edge_attr + full_edge_attr.transpose(1, 2)
    return full_edge_attr


def AdjPGD(
    model,
    batch_,
    target,
    budget,
    pgd_steps=50,
    surrogate_model=None,
    directed=False,
    limit_edgeattr_by_adj=False,
    base_lr=0.01,
    grad_clip=None,
    seed=0,
    n_samples=100,
    relative_budget=False,
    transform_batch=None,
    recorder: DataRecorder = None,
    target_list=None,
    clean_target_list=None,
    compute_cycle_count=False,
):
    if surrogate_model is None:
        surrogate_model = model
    model.eval()
    surrogate_model.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        init_x, node_mask, adj, init_edge_attr = to_dense_data(batch_)
        if len(init_x.shape)<3:
            init_x = init_x.unsqueeze(-1)
        if len(init_edge_attr.shape)<4:
            init_edge_attr = init_edge_attr.unsqueeze(-1)

        init_edge_attr = init_edge_attr.detach()
        init_adj = adj.clone().detach()
        device = adj.device

        batch = batch_.clone()
        full_edge_attr = fill_edge_attributes(batch, adj, init_edge_attr, directed=directed).detach()

        # setup DIRECTED budget
        if not directed:
            budget = 2 * budget  # undirected edges are counted twice
        batch_size = len(batch_.y)
        if relative_budget:
            num_edges = [batch_[i].num_edges for i in range(batch_size)]
            abs_budget = torch.tensor([math.ceil(budget * m) for m in num_edges])
            if not directed:
                abs_budget[abs_budget % 2 == 1] += 1
        else:
            abs_budget = torch.tensor([budget for _ in range(batch_size)])
        abs_budget = abs_budget.to(device)

    def f(adj, edge_attr, x=init_x, mask=node_mask, model=surrogate_model, batch=batch, batch_mask=None, **kwargs):
        if batch_mask is not None:
            batch = batch.clone()
            batch.y = batch.y[batch_mask]
        return target(model, batch, adj=adj, edge_attr=edge_attr, mask=mask, x=x, limit_edgeattr_by_adj=limit_edgeattr_by_adj, **kwargs)
    
    
    nan_mask_list = []
    clean_surrogate_model = create_surrogate_model(model, negative_slope=0)

    # optimize relaxed adjacency matrix
    for step in range(1, pgd_steps + 1):
        if clean_target_list is not None:
            with torch.no_grad():
                clean_target_list.append(f(adj, full_edge_attr, model=clean_surrogate_model))

        lr = base_lr * budget / math.sqrt(step)
        adj = pgd_step(
            f,
            abs_budget,
            init_x,
            adj,
            full_edge_attr,
            node_mask,
            init_adj,
            directed=directed,
            lr=lr,
            grad_clip=grad_clip,
            nan_mask_list=nan_mask_list,
            target_list=target_list,
        )

    if target_list is not None:
        with torch.no_grad():
            target_list.append(f(adj, full_edge_attr))
    if clean_target_list is not None:
        with torch.no_grad():
            clean_target_list.append(f(adj, full_edge_attr, model=clean_surrogate_model))

    if recorder:
        final_target = f(adj, full_edge_attr)
        final_clean_target = f(adj, full_edge_attr, model=clean_surrogate_model)

    if recorder:
        exp_budget = (adj - init_adj).abs().sum(dim=(-2, -1)).detach().cpu().numpy()
        if not directed:
            exp_budget = exp_budget / 2

    test_on_sparse = not isinstance(model, DSSnetwork)

    # sample graphs from relaxed adjacency matrix
    B, N, _ = init_adj.shape
    best_val = torch.full((B,), float("-inf")).to(device)
    best_adj = init_adj.clone()
    successful_samples = torch.zeros(B).to(device)
    with torch.no_grad():
        for _ in range(n_samples):
            discrete_adj = torch.bernoulli(adj)
            if not directed:
                discrete_adj = torch.tril(discrete_adj, diagonal=-1)
                discrete_adj = discrete_adj + discrete_adj.transpose(-2, -1)

            difference = (discrete_adj - init_adj).abs().sum(dim=(-2, -1))
            in_budget = difference <= abs_budget
            successful_samples += in_budget

            if test_on_sparse:
                sparse_data = to_sparse_batch(
                    init_x, discrete_adj, full_edge_attr, batch_.y, node_mask=node_mask
                ).to(device)

                val = target(
                    model,
                    sparse_data,
                )
            else:
                val = target(
                    surrogate_model,
                    batch_,
                    x=init_x,
                    adj=discrete_adj,
                    edge_attr=full_edge_attr
                )
            better = val > best_val

            update = torch.logical_and(in_budget, better)
            best_val[update] = val[update]
            best_adj[update] = discrete_adj[update]

    if recorder:
        successful_samples = (successful_samples / n_samples).detach().cpu().numpy()

    # evaluate best sampled graphs
    sparse_batch = to_sparse_batch(
        init_x,
        best_adj,
        full_edge_attr,
        batch_.y,
        node_mask=node_mask,
    ).to(device)

    if recorder:
        results = eval_attacked_graphs(
            model,
            sparse_batch,
            directed=directed,
            init_adj=init_adj,
            id=batch_.id,
            compute_cycle_count=compute_cycle_count,
            transform=transform_batch,
        )
        num_nan_grad = torch.stack(nan_mask_list, dim=-1).sum(dim=-1) / n_samples
        num_nan_grad = num_nan_grad.detach().cpu().numpy()
        abs_budget = abs_budget.detach().cpu().numpy()
        final_target = final_target.detach().cpu().numpy()
        final_clean_target = final_clean_target.detach().cpu().numpy()
        recorder.record(
            expected_budget=exp_budget,
            successful_samples=successful_samples,
            num_nan_grad=num_nan_grad,
            final_target=final_target,
            final_clean_target=final_clean_target,
            **results,
        )

    return sparse_batch


# def AdjPGD_improved(model,
#     batch_,
#     target,
#     budget,
#     pgd_steps=50,
#     surrogate_model=None,
#     directed=False,
#     limit_edgeattr_by_adj=False,
#     base_lr=0.01,
#     grad_clip=None,
#     seed=0,
#     n_samples=100,
#     relative_budget=False,
#     transform_batch=None,
#     recorder: DataRecorder = None,
#     target_list=None,
#     clean_target_list=None,):

#     return AttrPGD(
#         model,
#         batch_,
#         target,
#         budget,
#         vals_per_node_attr,
#         vals_per_edge_attr,
#         node_encoders,  # list of torch.Embedding's
#         edge_encoders,  # list of torch.Embedding's
#         pgd_steps=pgd_steps,
#         surrogate_model=surrogate_model,
#         directed=directed,
#         base_lr=base_lr,
#         grad_clip=grad_clip,
#         seed=seed,
#         n_samples=n_samples,
#         relative_budget=relative_budget,
#         limit_edgeattr_by_adj=limit_edgeattr_by_adj,
#         update_x=False,
#         update_edge_attr="negative", # "full" | "negative" | "none"
#         transform_batch=transform_batch,
#         recorder=recorder,
#     )