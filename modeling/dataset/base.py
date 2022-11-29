from dataset.samplers import TrainSampler, EvalSampler

from utils import MetaParent, DEVICE

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import os
import logging

logger = logging.getLogger(__name__)


class BaseDataset(metaclass=MetaParent):

    def get_samplers(self):
        raise NotImplementedError


# TODO implement
class CompositeDataset(BaseDataset, config_name='composite'):

    def __init__(self):
        pass

    def get_samplers(self):
        raise NotImplementedError


class InteractionsDataset(BaseDataset, config_name='interactions'):

    def __init__(
            self,
            train_sampler,
            test_sampler,
            num_users,
            num_items
    ):
        self._train_sampler = train_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_user_idx, max_item_idx = 0, 0

        train_dataset, train_num_interactions, train_max_user_idx, train_max_item_idx = cls._create_dataset(data_dir_path, 'train')
        max_user_idx, max_item_idx = max(max_user_idx, train_max_user_idx), max(max_item_idx, train_max_item_idx)

        test_dataset, test_num_interactions, test_max_user_idx, test_max_item_idx = cls._create_dataset(data_dir_path,'test')
        max_user_idx, max_item_idx = max(max_user_idx, test_max_user_idx), max(max_item_idx, test_max_item_idx)

        # Add zero user/item
        max_user_idx += 1
        max_item_idx += 1

        logger.info('Max user idx: {}'.format(max_user_idx))
        logger.info('Max item idx: {}'.format(max_item_idx))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (train_num_interactions + test_num_interactions) / max_user_idx / max_item_idx
        ))

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )  # TODO sanity check

        return cls(
            train_sampler=train_sampler,
            test_sampler=test_sampler,
            num_users=max_user_idx,
            num_items=max_item_idx
        )

    @staticmethod
    def _create_dataset(dir_path, part):
        max_user_idx = 0
        max_item_idx = 0

        dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        interactions_info = InteractionsDataset._create_interactions(data)
        user_interactions, item_interactions, num_interactions, _, _ = interactions_info
        max_user_idx = max(max_user_idx, interactions_info[3])
        max_item_idx = max(max_item_idx, interactions_info[4])

        dataset = []
        for user_id, item_id in zip(user_interactions, item_interactions):
            dataset.append({
                'user.ids': [user_id], 'user.length': 1,
                'item.ids': [item_id], 'item.length': 1
            })
        logger.info('{} dataset size: {}'.format(part, len(dataset)))

        return dataset, num_interactions, max_user_idx, max_item_idx

    @staticmethod
    def _create_interactions(data):
        user_interactions = []
        item_interactions = []
        num_interactions = 0

        max_user_id = 0
        max_item_id = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))

            user_interactions.extend([user_id] * len(item_ids))
            item_interactions.extend(item_ids)

            num_interactions += len(item_ids)

        return user_interactions, item_interactions, num_interactions, max_user_id, max_item_id

    def get_samplers(self):
        return self._train_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def meta(self):
        return {'num_users': self.num_users, 'num_items': self.num_items}


class GraphDataset(BaseDataset, config_name='graph'):

    def __init__(self, dataset):
        self._dataset = dataset
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        train_sampler, test_sampler = self._dataset.get_samplers()

        train_interactions, train_user_interactions, train_item_interactions = [], [], []
        test_interactions, test_user_interactions, test_item_interactions = [], [], []
        train_data_size, test_data_size = 0, 0

        for sample in train_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                train_interactions.append((user_id, item_id))
                train_user_interactions.append(user_id)
                train_item_interactions.append(item_id)

            train_data_size += len(item_ids)

        for sample in test_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                test_interactions.append((user_id, item_id))
                test_user_interactions.append(user_id)
                test_item_interactions.append(item_id)

            test_data_size += len(item_ids)

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

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
    def graph(self):
        return self._graph

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        return {'graph': self.graph, **self._dataset.meta}
