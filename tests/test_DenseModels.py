import unittest
from datasets import get_dataset_split
import torch
from torch_geometric.data import Batch
from models import create_surrogate_model, init_model
from models.ESAN.dense_conv import DenseGINConv as ESANDenseGINConv
from models.ESAN.sparse_conv import GINConv as ESANGINConv
from models.args import Args
from tests.util import rdm_graph
from models.util import to_dense_data
import math
import random



class TestDenseModel(unittest.TestCase):
    has_edge_attr = True

    def setUp(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        train_split, valid_split, test_split = get_dataset_split("ogbg-molhiv")

        torch.manual_seed(1234)
        train_data = train_split[:10]

        torch.manual_seed(1234)
        self.batch = Batch.from_data_list(
            test_split[:self.args.batch_size]
        ).to(self.device)

        self.model = init_model(self.args, train_data)
        self.model = self.model.to(self.device)

        for param in self.model.parameters():
            # modify parameters to get more diverse output on random data
            if hasattr(param, "data"):
                param.data = 2 * param.data

        self.surrogate_model = create_surrogate_model(self.model, negative_slope=0)
        self.model.eval()
        self.surrogate_model.eval()

    def check_func(
        self,
        rtol=1e-05,
        atol=1e-08,
        **kwargs,
    ):
        torch.manual_seed(1234)
        pred_sparse = self.model.predict(self.batch, **kwargs)
        pred_dense = self.surrogate_model.predict(self.batch, **kwargs)
        self.assertTrue(
            torch.isclose(pred_sparse, pred_dense, rtol=rtol, atol=atol).all()
        )

    def check_grad(self, detect_anomaly=True):
        torch.manual_seed(12)
        batch = self.model.encode(self.batch)
        x, _, adj, edge_attr = to_dense_data(batch)

        def f(x, adj, edge_attr):
            return self.surrogate_model(
                batch, x=x, adj=adj, edge_attr=edge_attr, encode=False
            ).sum()

        torch.autograd.set_detect_anomaly(detect_anomaly)
        x = x.requires_grad_()
        x_grad = torch.autograd.grad(f(x, adj, edge_attr), x)[0]
        self.assertTrue(x_grad.abs().sum() > 0)

        adj = adj.requires_grad_()
        adj_grad = torch.autograd.grad(f(x, adj, edge_attr), adj)[0]
        if not detect_anomaly:
            self.assertTrue((~adj_grad.isnan()).sum() > 0)
            adj_grad[adj_grad.isnan()] = 0
        self.assertTrue(adj_grad.abs().sum() > 0)

        if self.has_edge_attr:
            edge_attr = edge_attr.requires_grad_()
            edge_attr_grad = torch.autograd.grad(f(x, adj, edge_attr), edge_attr)[0]
            self.assertTrue(edge_attr_grad.abs().sum() > 0)


class TestDenseGINE(TestDenseModel):
    args = Args(
        model="GINE",
        dataset="ogbg-molhiv",
        batch_size=16,
        seed=3821,
        num_layers=2,
        hidden_units=4,
    )
    num_node_feat = 3
    num_edge_feat = 5

    def test_func(self):
        self.check_func()

    def test_grad(self):
        self.check_grad()


class TestDenseGIN(TestDenseModel):
    args = Args(
        model="GIN",
        dataset="ogbg-molhiv",
        batch_size=16,
        seed=3821,
        num_layers=2,
        hidden_units=4,
    )
    num_node_feat = 3
    num_edge_feat = 5
    has_edge_attr = False

    def test_func(self):
        self.check_func()

    def test_grad(self):
        self.check_grad()


class TestDenseDropGINE(TestDenseModel):
    args = Args(
        model="DropGINE",
        dataset="ogbg-molhiv",
        batch_size=16,
        seed=3821,
        num_layers=2,
        hidden_units=4,
    )
    num_node_feat = 3
    num_edge_feat = 5

    # cannot test function since not deterministic (and dense impl uses different randomness)
    def test_func(self):
        torch.manual_seed(1234)
        drop = torch.bernoulli(
            torch.ones([self.model.num_runs, self.batch.num_nodes], device=self.device) * self.model.p
        ).bool()
        self.check_func(drop=drop)

    def test_grad(self):
        self.check_grad()


class TestDenseDropGIN(TestDenseModel):
    args = Args(
        model="DropGIN",
        dataset="ogbg-molhiv",
        batch_size=16,
        seed=3821,
        num_layers=2,
        hidden_units=4,
    )
    num_node_feat = 3
    num_edge_feat = 5
    has_edge_attr = False

    # cannot test function since not deterministic (and dense impl uses different randomness)
    def test_func(self):
        torch.manual_seed(1234)
        drop = torch.bernoulli(
            torch.ones([self.model.num_runs, self.batch.num_nodes], device=self.device) * self.model.p
        ).bool()
        self.check_func(drop=drop)

    def test_grad(self):
        self.check_grad()


class TestDenseSignNet(TestDenseModel):
    args = Args(
        model="SignNet",
        dataset="ogbg-molhiv",
        batch_size=16,
        seed=3821,
        num_layers=1,
        hidden_units=5,
        pos_enc_dim=3,
    )
    num_node_feat = 3
    num_edge_feat = 5

    def test_func(self):
        self.check_func(rtol=1e-04)

    def test_grad(self):
        self.check_grad(detect_anomaly=False)



class TestDenseESAN(TestDenseModel):
    args = Args(
        model="ESAN",
        dataset="ogbg-molhiv",
        batch_size=8,
        seed=3821,
        num_layers=1,
        hidden_units=5,
        policy="ego_nets_plus",
        num_hops=2,
        sample_fraction=0.5,
    )
    num_node_feat = 3
    num_edge_feat = 5

    def setUp(self):
        super().setUp()
        self.surrogate_model = create_surrogate_model(self.model, negative_slope=0)
        self.model.eval()
        self.surrogate_model.eval()

    def test_func(self):
        if self.args.sample_fraction<1:
            n_subgraphs = min(graph.num_nodes for graph in self.batch.to_data_list())
            count = math.ceil(self.args.sample_fraction * n_subgraphs)
            sample_idx = random.sample(list(range(n_subgraphs)), count)
            self.check_func(subgraph_indices=sample_idx)
        else:
            self.check_func()

    def test_grad(self):
        self.check_grad()

