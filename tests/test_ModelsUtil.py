import unittest
import torch
from datasets import get_dataset_split
from datasets.util import paired_edges_order
from models.util import to_dense_data, to_sparse_batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.data import Batch

from tests.util import rdm_graph


class TestModelsUtil(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

        self.num_node_feat = 1
        self.num_edge_feat = 1
        self.batch_size = 4
        self.batch = Batch.from_data_list(
            [
                rdm_graph(
                    21 + 2 * i,
                    0.3 - 0.1 * (i / self.batch_size),
                    self.num_node_feat,
                    self.num_edge_feat,
                    feature_type="real",
                )
                for i in range(self.batch_size)
            ]
        )
        self.batch.x = self.batch.x.squeeze()
        self.batch.edge_attr = self.batch.edge_attr.squeeze()

    def test_to_sparse_data(self):
        x, mask, adj, edge_attr = to_dense_data(self.batch)
        batch_from_dense = to_sparse_batch(x, adj, edge_attr, self.batch.y)

        # check if nodes are alligned by checking if degrees align
        torch.all(
            degree(self.batch.edge_index[0]) == degree(batch_from_dense.edge_index[0])
        )
        torch.all(
            degree(self.batch.edge_index[0]) == degree(batch_from_dense.edge_index[0])
        )

        # check if node features match
        self.assertTrue(torch.all(batch_from_dense.x == self.batch.x))

        # check if the same edges are present
        self.assertTrue(
            torch.all(
                torch.sort(batch_from_dense.edge_index)[0]
                == torch.sort(self.batch.edge_index)[0]
            )
        )

        # check if same edge attributes are present
        self.assertTrue(
            torch.all(
                torch.sort(batch_from_dense.edge_attr)[0]
                == torch.sort(self.batch.edge_attr)[0]
            )
        )

        # check if edge_attributes line up
        order = paired_edges_order(self.batch.edge_index)
        order_dense = paired_edges_order(batch_from_dense.edge_index)
        torch.all(
            self.batch.edge_attr[order] == batch_from_dense.edge_attr[order_dense]
        )

        # test if node batch matches
        torch.all(self.batch.batch == batch_from_dense.batch)


if __name__ == "__main__":
    unittest.main()
