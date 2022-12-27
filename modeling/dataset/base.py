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

        test_dataset, test_num_interactions, test_max_user_idx, test_max_item_idx = cls._create_dataset(data_dir_path, 'test')
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


class SequenceDataset(BaseDataset, config_name='sequence'):

    def __init__(
            self,
            train_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):  # TODO implement amazon attributes in more general way
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_user_idx, max_item_idx, max_sequence_length = 0, 0, 0

        train_dataset, train_max_user_idx, train_max_item_idx, train_max_sequence_length = cls._create_dataset(
            data_dir_path, 'train', config['max_sequence_length']
        )
        max_user_idx, max_item_idx = max(max_user_idx, train_max_user_idx), max(max_item_idx, train_max_item_idx)
        max_sequence_length = max(max_sequence_length, train_max_sequence_length)

        test_dataset, test_max_user_idx, test_max_item_idx, test_max_sequence_length = cls._create_dataset(
            data_dir_path, 'test', config['max_sequence_length']
        )
        max_user_idx, max_item_idx = max(max_user_idx, test_max_user_idx), max(max_item_idx, test_max_item_idx)
        max_sequence_length = max(max_sequence_length, test_max_sequence_length)

        logger.info('Max user idx: {}'.format(max_user_idx))
        logger.info('Max item idx: {}'.format(max_item_idx))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (len(train_dataset) + len(test_dataset)) / max_user_idx / max_item_idx
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
            num_items=max_item_idx,
            max_sequence_length=max_sequence_length
        )

    @staticmethod
    def _create_dataset(dir_path, part, max_sequence_length=None):
        max_user_idx = 0
        max_item_idx = 0
        max_sequence_len = 0

        dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        sequence_info = SequenceDataset._create_sequences(data, max_sequence_length)
        user_sequences, item_sequences, _, _, _ = sequence_info
        max_user_idx = max(max_user_idx, sequence_info[2])
        max_item_idx = max(max_item_idx, sequence_info[3])
        max_sequence_len = max(max_sequence_len, sequence_info[4])

        dataset = []
        for user_idx, item_ids in zip(user_sequences, item_sequences):
            if len(item_ids) > 5:  # TODO fix
                dataset.append({
                    'user.ids': [user_idx], 'user.length': 1,
                    'item.ids': item_ids, 'item.length': len(item_ids)
                })

        logger.info('{} dataset size: {}'.format(part, len(dataset)))
        logger.info('{} dataset max sequence length: {}'.format(part, max_sequence_len))

        return dataset, max_user_idx, max_item_idx, max_sequence_len

    @staticmethod
    def _create_sequences(data, max_sample_len):
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]][-max_sample_len:]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return user_sequences, item_sequences, max_user_id, max_item_id, max_sequence_length

    def get_samplers(self):
        return self._train_sampler, self._test_sampler

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def meta(self):
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'max_sequence_length': self.max_sequence_length
        }


class GraphDataset(BaseDataset, config_name='graph'):

    def __init__(self, dataset, use_user_graph=False, use_item_graph=False):
        self._dataset = dataset
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph

        self._num_users = dataset._num_users + 2
        self._num_items = dataset._num_items + 2

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
        user2item_connections = csr_matrix(
            (np.ones(len(train_user_interactions)), (train_user_interactions, train_item_interactions)),
            shape=(self._num_users, self._num_items)
        )

        user2item_connections_transpose = csr_matrix(
            (np.ones(len(train_user_interactions)), (train_item_interactions, train_user_interactions)),
            shape=(self._num_items, self._num_users)
        )

        self._graph = self.get_sparse_graph_layer(user2item_connections, self._num_users, self._num_items)

        # TODO fix
        # if self._use_user_graph:
        #     # (users, user), bipartite graph
        #     user2user_connections = user2item_connections.multiply(user2item_connections_transpose)
        #     self._user_graph = self.get_sparse_graph_layer(user2user_connections, self._num_users, self._num_users)
        #
        # if self._use_item_graph:
        #     # (item, item), bipartite graph
        #     item2item_connections = user2item_connections_transpose.multiply(user2item_connections)
        #     self._item_graph = self.get_sparse_graph_layer(item2item_connections, self._num_items, self._num_items)

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False)
        )

    @staticmethod
    def get_sparse_graph_layer(sparse_matrix, fst_dim, snd_dim):
        adj_mat = sp.dok_matrix(
            (fst_dim + snd_dim, fst_dim + snd_dim),
            dtype=np.float32
        )

        adj_mat = adj_mat.tolil()

        R = sparse_matrix.tolil()  # list of lists (num_users, num_items)

        adj_mat[:fst_dim, fst_dim:] = R  # (num_users, num_items)
        adj_mat[fst_dim:, :fst_dim] = R.T  # (num_items, num_users)

        adj_mat = adj_mat.todok()
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0])  # TODO ????

        # TODO check next part of layer creation
        edges_degree = np.array(adj_mat.sum(axis=1))  # D

        d_inv = np.power(edges_degree, -0.5).flatten()  # D^(-0.5)
        d_inv[np.isinf(d_inv)] = 0.0  # fix NaNs
        d_mat = sp.diags(d_inv)  # make it square matrix

        # D^(-0.5) @ A @ D^(-0.5)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)

        norm_adj = norm_adj.tocsr()

        graph = GraphDataset._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(DEVICE)

        return graph

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
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
        meta = {'graph': self.graph, **self._dataset.meta}
        if self._use_user_graph:
            meta['user_graph'] = None
        if self._use_item_graph:
            meta['item_graph'] = None

        return meta
