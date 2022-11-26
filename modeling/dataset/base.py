from utils import MetaParent, DEVICE

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


class BaseDataset(metaclass=MetaParent):

    def get_samplers(self):
        raise NotImplementedError


class GraphDataset(BaseDataset, config_name='graph'):

    def __init__(self, dataset):
        self._dataset = dataset
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        train_sampler, validation_sampler, test_sampler = self._dataset.get_samplers()

        train_interactions, train_user_interactions, train_item_interactions = [], [], []
        val_interactions, val_user_interactions, val_item_interactions = [], [], []
        test_interactions, test_user_interactions, test_item_interactions = [], [], []
        train_data_size, val_data_size, test_data_size = 0, 0, 0

        for sample in train_sampler.dataset:
            user_id = sample['user_id']
            item_ids = sample['sample.ids'] + sample['answer.ids']

            for item_id in item_ids:
                train_interactions.append((user_id, item_id))
                train_user_interactions.append(user_id)
                train_item_interactions.append(item_id)

            train_data_size += len(item_ids)

        for sample in validation_sampler.dataset:
            user_id = sample['user_id']
            item_ids = sample['sample.ids'] + sample['answer.ids']

            for item_id in item_ids:
                val_interactions.append((user_id, item_id))
                val_user_interactions.append(user_id)
                val_item_interactions.append(item_id)

            val_data_size += len(item_ids)

        for sample in test_sampler.dataset:
            user_id = sample['user_id']
            item_ids = sample['sample.ids'] + sample['answer.ids']

            for item_id in item_ids:
                test_interactions.append((user_id, item_id))
                test_user_interactions.append(user_id)
                test_item_interactions.append(item_id)

            test_data_size += len(item_ids)

        self._num_users += 1
        self._num_items += 1

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        self._val_interactions = np.array(val_interactions)
        self._val_user_interactions = np.array(val_user_interactions)
        self._val_item_interactions = np.array(val_item_interactions)

        self._test_interactions = np.array(test_interactions)
        self._test_user_interactions = np.array(test_user_interactions)
        self._test_item_interactions = np.array(test_item_interactions)

        # (users, items), bipartite graph
        self._user2item_connections = csr_matrix(
            (np.ones(len(train_user_interactions)), (train_user_interactions, train_item_interactions)),
            shape=(self._num_users, self._num_items)
        )

        self.users_degree = np.array(self._user2item_connections.sum(axis=1)).squeeze()
        self.users_degree[self.users_degree == 0.0] = 1.0

        self.items_degree = np.array(self._user2item_connections.sum(axis=0)).squeeze()
        self.items_degree[self.items_degree == 0.0] = 1.0

        self._graph = None
        self._graph = self.get_sparse_graph()

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(dataset=dataset)

    def get_sparse_graph(self):
        if self._graph is None:
            adj_mat = sp.dok_matrix(
                (self._num_users + self._num_items, self._num_users + self._num_items),
                dtype=np.float32
            )

            adj_mat = adj_mat.tolil()

            R = self._user2item_connections.tolil()  # list of lists (num_users, num_items)
            adj_mat[:self._num_users, self._num_users:] = R  # (num_users, num_items)
            adj_mat[self._num_users:, :self._num_users] = R.T  # (num_items, num_users)

            adj_mat = adj_mat.todok()
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])  # TODO ????

            edges_degree = np.array(adj_mat.sum(axis=1))  # D

            d_inv = np.power(edges_degree, -0.5).flatten()  # D^(-0.5)
            d_inv[np.isinf(d_inv)] = 0.0  # fix NaNs
            d_mat = sp.diags(d_inv)

            # D^(-0.5) @ A @ D^(-0.5)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)

            norm_adj = norm_adj.tocsr()

            self._graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self._graph = self._graph.coalesce().to(DEVICE)

        return self._graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    @property
    def num_users(self):
        return self._dataset.num_users

    @property
    def num_items(self):
        return self._dataset.num_items

    @property
    def max_sequence_length(self):
        return self._dataset.max_sequence_length

    @property
    def graph(self):
        return self._graph

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length,
            'graph': self.graph
        }
