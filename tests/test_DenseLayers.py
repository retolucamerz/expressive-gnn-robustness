import unittest
import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GINConv
from models.GIN import DenseGINConv
from models.GINE import GINEConv, DenseGINEConv
from models.util import to_dense_data
from tests.util import rdm_graph

f = lambda x: x * x + 2 * x


class TestDenseGINEConv(unittest.TestCase):
    def test_func(self):
        torch.manual_seed(1234)

        for i in range(10):
            data = rdm_graph(20 + i, 0.3, 5 + i, 5 + i)

            conv = GINEConv(f, aggr_type="mean")
            denseconv = DenseGINEConv(f, aggr_type="mean")

            g = conv.forward(data.x, data.edge_index, data.edge_attr)

            x, mask = to_dense_batch(data.x, None)  # data.batch
            adj = to_dense_adj(data.edge_index, None).requires_grad_()  # data.batch
            edge_attr = to_dense_adj(data.edge_index, None, data.edge_attr)
            g_ = denseconv.forward(x, adj, edge_attr, mask).squeeze()

            self.assertTrue(torch.all(torch.isclose(g, g_)))

    def test_grad(self):
        torch.manual_seed(42)
        denseconv = DenseGINEConv(f)
        denseconv.reset_parameters()
        data = rdm_graph(20, 0.2, 3, 3, directed=False)
        x, mask, adj, edge_attr = to_dense_data(data)

        x = x.requires_grad_()
        adj = adj.requires_grad_()
        edge_attr = edge_attr.requires_grad_()

        self.assertTrue((adj == adj.transpose(1, 2)).all())
        self.assertTrue((edge_attr == edge_attr.transpose(1, 2)).all())

        x_grad = torch.autograd.grad(
            denseconv.forward(x, adj, edge_attr, mask).sum(), x
        )[0]
        adj_grad = torch.autograd.grad(
            denseconv.forward(x, adj, edge_attr, mask).sum(), adj
        )[0]
        edge_attr_grad = torch.autograd.grad(
            denseconv.forward(x, adj, edge_attr, mask).sum(), edge_attr
        )[0]

        self.assertTrue((x_grad[~mask] == 0).all())
        self.assertTrue((adj_grad[~mask] == 0).all())
        self.assertTrue((edge_attr_grad[~mask] == 0).all())


class TestDenseGINConv(unittest.TestCase):
    def test_func(self):
        torch.manual_seed(42)

        for i in range(10):
            data = rdm_graph(20 + i, 0.3, 3 + i, 3 + i)

            conv = GINConv(f)
            denseconv = DenseGINConv(f)

            g = conv.forward(data.x, data.edge_index)

            x, mask = to_dense_batch(data.x, None)  # data.batch
            adj = to_dense_adj(data.edge_index, None).requires_grad_()  # data.batch
            g_ = denseconv.forward(x, adj, mask).squeeze()

            self.assertTrue(torch.all(torch.isclose(g, g_)))

    def test_grad(self):
        torch.manual_seed(42)
        denseconv = DenseGINConv(f)
        data = rdm_graph(20, 0.2, 2, 3, directed=False)
        x, mask, adj, edge_attr = to_dense_data(data)

        x = x.requires_grad_()
        adj = adj.requires_grad_()

        x_grad = torch.autograd.grad(denseconv.forward(x, adj, mask).sum(), x)[0]
        adj_grad = torch.autograd.grad(denseconv.forward(x, adj, mask).sum(), adj)[0]

        self.assertTrue((x_grad[~mask] == 0).all())
        self.assertTrue((adj_grad[~mask] == 0).all())


if __name__ == "__main__":
    unittest.main()
