# test_FeatureAttack
import unittest
import torch
import torch_scatter
import random
from attacks.feature_attack import (
    _expected_budget,
    create_edge_attr_distr,
    create_x_distr,
    decrease_distr_budget,
    embed_distr,
    encode_as_distr,
    expected_budget,
    fix_init_value,
    is_normalized,
    normalize_distr,
    reduce_over_attr,
    reset_distr,
    sample_from_distr,
)
from datasets import get_dataset_split
from models.SignNet import get_laplacian_dense
from models.util import to_dense_data
from tests.util import rdm_graph
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj


class Encoder(torch.nn.Module):
    def __init__(self, num_feat, embedding_dim, num_embeddings=10):
        super(Encoder, self).__init__()
        self.embeddings = torch.nn.ModuleList()
        for _ in range(num_feat):
            self.embeddings.append(torch.nn.Embedding(num_embeddings, embedding_dim))

    def forward(self, x):
        ret = 0
        for d in range(x.shape[-1]):
            ret += self.embeddings[d](x[..., d])
        return ret


class TestFeatureAttack(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0xDEADBEEF)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.num_node_feat = 3
        self.num_edge_feat = 5
        self.batch_size = 16
        self.batch = Batch.from_data_list(
            [
                rdm_graph(
                    21 + 2 * i,
                    0.3 - 0.2 * (i / self.batch_size),
                    self.num_node_feat,
                    self.num_edge_feat,
                    feature_type="discrete",
                )
                for i in range(self.batch_size)
            ]
        ).to(self.device)

        random.seed(11)
        self.vals_per_node_attr = {
            k: list(range(10)) for k in range(self.num_node_feat)
        }
        for val in self.vals_per_node_attr.values():
            random.shuffle(val)
        self.vals_per_edge_attr = {
            k: list(range(10)) for k in range(self.num_edge_feat)
        }
        for val in self.vals_per_edge_attr.values():
            random.shuffle(val)

        self.init_x, self.mask, self.adj, self.init_edge_attr = to_dense_data(
            self.batch
        )

        self.node_embedding = Encoder(self.num_node_feat, 7).to(self.device)
        self.edge_embedding = Encoder(self.num_edge_feat, 7).to(self.device)

        self.x_distr = create_x_distr(self.init_x, self.vals_per_node_attr, self.mask, self.device)
        assert is_normalized(self.x_distr[self.mask], self.vals_per_node_attr).all()

        self.edge_attr_distr = create_edge_attr_distr(self.init_edge_attr, self.vals_per_edge_attr, self.adj, self.mask, self.device)

        N = self.adj.shape[1]
        valid_edge_mask = torch.einsum('bi,bj->bij', (self.mask, self.mask)) # batched outer product
        valid_edge_mask[:,range(N),range(N)] = 0
        assert is_normalized(self.edge_attr_distr[valid_edge_mask], self.vals_per_edge_attr).all()

    def test_encode_embed(self):
        # check sum p == 1
        x_sum_per_attr = reduce_over_attr(self.x_distr, self.vals_per_node_attr)
        orig = self.mask.float().unsqueeze(-1) * torch.ones_like(self.init_x, dtype=torch.float32) 
        self.assertTrue(
            torch.isclose(
                x_sum_per_attr, orig
            ).all()
        )

        self.edge_attr_distr[self.adj == 0] = reset_distr(
            self.edge_attr_distr[self.adj == 0], self.vals_per_edge_attr, inplace=False
        )
        edge_sum_per_attr = reduce_over_attr(
            self.edge_attr_distr, self.vals_per_edge_attr
        )
        self.assertTrue(
            torch.isclose(
                edge_sum_per_attr,
                torch.ones_like(self.init_edge_attr, dtype=torch.float32),
            ).all()
        )

        # encode empty distr
        distr = torch.zeros((13,))

        # check embed(x) == embed_distr(x_distr, ...)
        x_emb = self.node_embedding(self.batch.x)
        x_emb_ = embed_distr(
            self.x_distr, self.node_embedding.embeddings, self.vals_per_node_attr
        )
        self.assertTrue(torch.isclose(x_emb, x_emb_[self.mask], atol=1e-6).all())
        self.assertTrue((x_emb_[~self.mask]==0).all())

        edge_attr_emb = self.edge_embedding(self.batch.edge_attr)
        edge_attr_emb = to_dense_adj(
            self.batch.edge_index, self.batch.batch, edge_attr_emb
        )
        edge_attr_emb_ = embed_distr(
            self.edge_attr_distr,
            self.edge_embedding.embeddings,
            self.vals_per_edge_attr,
        )
        self.assertTrue(
            torch.isclose(
                edge_attr_emb, edge_attr_emb_ * self.adj.unsqueeze(-1), atol=1e-6
            ).all()
        )

        # check reset distr
        tmp = self.edge_attr_distr.clone()
        tmp[self.adj == 0] = reset_distr(
            tmp[self.adj == 0], self.vals_per_edge_attr, inplace=False
        )
        self.assertTrue(
            (
                tmp * self.adj.unsqueeze(-1)
                == self.edge_attr_distr * self.adj.unsqueeze(-1)
            ).all()
        )
        self.assertTrue(
            torch.isclose(
                tmp[self.adj == 0], 0.1 * torch.ones_like(tmp[self.adj == 0])
            ).all()
        )

    def test_normalize_distr(self):
        # no change
        normed = normalize_distr(
            self.x_distr, self.x_distr, self.vals_per_node_attr, inplace=False
        )
        self.assertTrue(torch.isclose(normed, self.x_distr).all())

        # zero
        normed = normalize_distr(
            torch.zeros_like(self.x_distr),
            self.x_distr,
            self.vals_per_node_attr,
            inplace=False,
        )
        self.assertTrue(torch.isclose(normed, self.x_distr).all())

        # random change
        for _ in range(3):
            x_distr_ = self.x_distr.clone()
            x_distr_ += (
                0.5
                * (1 - self.x_distr)
                * torch.rand(self.x_distr.shape, device=self.device)
                - 0.1
            )
            normed = normalize_distr(
                x_distr_, self.x_distr, self.vals_per_node_attr, inplace=False
            )

            self.assertTrue(((0 <= normed) & (normed <= 1)).all())
            sum_per_attr = reduce_over_attr(normed, self.vals_per_node_attr)
            self.assertTrue(
                torch.isclose(
                    sum_per_attr,
                    torch.ones_like(sum_per_attr, dtype=torch.float32),
                ).all()
            )

    def test_decrease_distr_budget(self):
        # simple example
        vals_per_attr = {0: [2, 0, 1]}
        init_distr = torch.zeros((3,))
        init_distr[1] = 1
        distr = 1 / 3 * torch.ones((3,))

        init_budget = _expected_budget(distr, init_distr).sum()

        def decr_by(v):
            decreased = decrease_distr_budget(
                distr, init_distr, torch.tensor(v), vals_per_attr
            )
            budget_diff = init_budget - _expected_budget(decreased, init_distr).sum()
            return budget_diff

        for v in [0.1, 0.25, 0.5, 0.6, 2 / 3, 0.8, 1]:
            res = torch.tensor(min(v, init_budget.item()))
            self.assertTrue(torch.isclose(decr_by(v), res))

        for _ in range(3):
            x_distr_ = self.x_distr.clone()
            x_distr_ += (
                0.8
                * (1 - self.x_distr)
                * torch.rand(self.x_distr.shape, device=self.device)
            )
            x_distr_[~self.mask] = 0
            normed = normalize_distr(
                x_distr_, self.x_distr, self.vals_per_node_attr, inplace=False
            )
            assert (normed[~self.mask]==0).all()

            mu = torch.tensor([0.2 for _ in range(self.batch_size)], device=self.device)
            decreased_distr = decrease_distr_budget(
                normed, self.x_distr, mu, self.vals_per_node_attr
            )

            # test smaller entries
            self.assertTrue((decreased_distr[self.mask] * (1 - self.x_distr[self.mask]) < normed[self.mask]).all())
            self.assertTrue((decreased_distr[~self.mask] == normed[~self.mask]).all())

            # test if normed
            sum_per_attr = reduce_over_attr(decreased_distr[self.mask], self.vals_per_node_attr)
            self.assertTrue(
                torch.isclose(
                    sum_per_attr,
                    torch.ones_like(sum_per_attr, dtype=torch.float32),
                ).all()
            )

            # test budget
            init_budget = (self.x_distr * (1 - normed)).sum(-1)
            decr_budget = (self.x_distr * (1 - decreased_distr)).sum(-1)
            diff = init_budget - decr_budget
            self.assertTrue(((0 < diff[self.mask]) & (diff[self.mask] < self.num_node_feat)).all())
            self.assertTrue((diff[~self.mask]==0).all())

            # test decrease completely
            mu = torch.tensor([1 for _ in range(self.batch_size)], device=self.device)
            decreased_distr = decrease_distr_budget(
                normed, self.x_distr, mu, self.vals_per_node_attr
            )
            self.assertTrue(
                torch.isclose(
                    decreased_distr,
                    self.x_distr,
                ).all()
            )

    def test_fix_init_val(self):
        rdm_distr = (
            torch.rand(self.x_distr.shape, device=self.device) / 10
        )  # ensures 0 <= sum over distr <= 1
        rdm_distr[~self.mask] = 0
        fixed = fix_init_value(
            rdm_distr, self.x_distr, self.vals_per_node_attr, inplace=False
        )
        assert (fixed[~self.mask]==0).all()

        # fixed should be distribution
        sum_per_attr = reduce_over_attr(fixed, self.vals_per_node_attr)[self.mask]
        self.assertTrue(
            torch.isclose(
                sum_per_attr,
                torch.ones_like(sum_per_attr, dtype=torch.float32),
            ).all()
        )
        self.assertTrue(((0 <= fixed) & (fixed <= 1)).all())

        # should not change normalized distr
        x_distr_ = self.x_distr.clone()
        x_distr_ += (
            0.8
            * (1 - self.x_distr)
            * torch.rand(self.x_distr.shape, device=self.device)
        )
        normed = normalize_distr(
            x_distr_, self.x_distr, self.vals_per_node_attr, inplace=False
        )
        fixed = fix_init_value(
            normed, self.x_distr, self.vals_per_node_attr, inplace=False
        )
        self.assertTrue(torch.isclose(fixed, normed).all())

        # fixing 'zero distribution' should yield initial distr
        fixed = fix_init_value(
            torch.zeros_like(self.x_distr),
            self.x_distr,
            self.vals_per_node_attr,
            inplace=False,
        )
        self.assertTrue(torch.isclose(fixed, self.x_distr).all())

    def test_expected_budget(self):
        exp = expected_budget(
            self.x_distr,
            self.x_distr,
            self.adj,
            self.adj,
            self.edge_attr_distr,
            self.edge_attr_distr,
        )
        self.assertTrue((exp == 0).all())

        active_edge_distrs = (self.edge_attr_distr == 1).sum(dim=-1) == len(
            self.vals_per_edge_attr
        )
        edge_attr_distr = self.edge_attr_distr.clone()
        edge_attr_distr[~active_edge_distrs] += 0.8 * torch.rand(
            edge_attr_distr[~active_edge_distrs].shape, device=self.device
        )
        normalize_distr(edge_attr_distr, self.edge_attr_distr, self.vals_per_edge_attr)
        exp = expected_budget(
            self.x_distr,
            self.x_distr,
            self.adj,
            self.adj,
            edge_attr_distr,
            self.edge_attr_distr,
        )
        self.assertTrue((exp == 0).all())

    def test_sample(self):
        sampled_edge_attr = torch.zeros_like(self.init_edge_attr)
        N = self.adj.shape[1]
        valid_edge_mask = torch.einsum('bi,bj->bij', (self.mask, self.mask))
        valid_edge_mask[:,range(N),range(N)] = 0

        sampled_edge_attr[valid_edge_mask] = sample_from_distr(self.edge_attr_distr[valid_edge_mask], self.vals_per_edge_attr)
        sampled_edge_attr = sampled_edge_attr * self.adj.unsqueeze(-1)

        self.assertTrue((sampled_edge_attr[valid_edge_mask]==self.init_edge_attr[valid_edge_mask]).all())
