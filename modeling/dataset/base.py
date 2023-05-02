from collections import defaultdict

from tqdm import tqdm

from dataset.samplers import TrainSampler, ValidationSampler, EvalSampler

from utils import MetaParent, DEVICE

import pickle
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


class SequenceDataset(BaseDataset, config_name='sequence'):

    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(config['path_to_data_dir'], config['name'])
        max_user_idx, max_item_idx, max_sequence_length = 0, 0, 0

        train_dataset, train_max_user_idx, train_max_item_idx, train_max_sequence_length = cls._create_dataset(
            data_dir_path, 'train_new', config['max_sequence_length']
        )
        max_user_idx, max_item_idx = max(max_user_idx, train_max_user_idx), max(max_item_idx, train_max_item_idx)
        max_sequence_length = max(max_sequence_length, train_max_sequence_length)

        validation_dataset, validation_max_user_idx, validation_max_item_idx, validation_max_sequence_length = cls._create_dataset(
            data_dir_path, 'validation_new', config['max_sequence_length']
        )
        max_user_idx, max_item_idx = max(max_user_idx, validation_max_user_idx), max(max_item_idx, validation_max_item_idx)
        max_sequence_length = max(max_sequence_length, validation_max_sequence_length)

        test_dataset, test_max_user_idx, test_max_item_idx, test_max_sequence_length = cls._create_dataset(
            data_dir_path, 'test_new', config['max_sequence_length']
        )
        max_user_idx, max_item_idx = max(max_user_idx, test_max_user_idx), max(max_item_idx, test_max_item_idx)
        max_sequence_length = max(max_sequence_length, test_max_sequence_length)

        logger.info('Max user idx: {}'.format(max_user_idx))
        logger.info('Max item idx: {}'.format(max_item_idx))
        logger.info('{} dataset sparsity: {}'.format(
            config['name'], (len(train_dataset) + len(test_dataset)) / max_user_idx / max_item_idx
        ))

        # TODO sanity check
        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )
        validation_sampler = ValidationSampler.create_from_config(
            config['samplers'],
            dataset=validation_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_idx,
            num_items=max_item_idx
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_idx,
            num_items=max_item_idx,
            max_sequence_length=max_sequence_length
        )

    @classmethod
    def _create_dataset(cls, dir_path, part, max_sequence_length=None):
        max_user_idx = 0
        max_item_idx = 0
        max_sequence_len = 0

        if os.path.exists(os.path.join(dir_path, '{}.pkl'.format(part))):
            with open(os.path.join(dir_path, '{}.pkl'.format(part)), 'rb') as dataset_file:
                dataset, max_user_idx, max_item_idx, max_sequence_len = pickle.load(dataset_file)
        else:
            dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
            with open(dataset_path, 'r') as f:
                data = f.readlines()

            sequence_info = cls._create_sequences(data, max_sequence_length)
            user_sequences, item_sequences, _, _, _ = sequence_info
            max_user_idx = max(max_user_idx, sequence_info[2])
            max_item_idx = max(max_item_idx, sequence_info[3])
            max_sequence_len = max(max_sequence_len, sequence_info[4])

            dataset = []
            for user_idx, item_ids in zip(user_sequences, item_sequences):
                dataset.append({
                    'user.ids': [user_idx], 'user.length': 1,
                    'item.ids': item_ids, 'item.length': len(item_ids)
                })

            logger.info('{} dataset size: {}'.format(part, len(dataset)))
            logger.info('{} dataset max sequence length: {}'.format(part, max_sequence_len))

            with open(os.path.join(dir_path, '{}.pkl'.format(part)), 'wb') as dataset_file:
                pickle.dump(
                    (dataset, max_user_idx, max_item_idx, max_sequence_len),
                    dataset_file
                )

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
        return self._train_sampler, self._validation_sampler, self._test_sampler

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

    def __init__(self, dataset, graph_dir_path, use_user_graph=False, use_item_graph=False):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph

        self._num_users = dataset.num_users + 2
        self._num_items = dataset.num_items + 2

        train_sampler, test_sampler = self._dataset.get_samplers()

        train_interactions, train_user_interactions, train_item_interactions = [], [], []
        test_interactions, test_user_interactions, test_item_interactions = [], [], []
        train_data_size, test_data_size = 0, 0

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)

        for sample in train_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                train_interactions.append((user_id, item_id))
                train_user_interactions.append(user_id)
                train_item_interactions.append(item_id)

                train_user_2_items[user_id].add(item_id)
                train_item_2_users[item_id].add(user_id)

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

        path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        if os.path.exists(path_to_graph):
            self._graph = sp.load_npz(path_to_graph)
        else:
            # (users, items), bipartite graph
            user2item_connections = csr_matrix(
                (np.ones(len(train_user_interactions)), (train_user_interactions, train_item_interactions)),
                shape=(self._num_users, self._num_items)
            )
            self._graph = self.get_sparse_graph_layer(user2item_connections, self._num_users, self._num_items,
                                                      biparite=True)
            sp.save_npz(path_to_graph, self._graph)

        self._graph = self._convert_sp_mat_to_sp_tensor(self._graph).coalesce().to(DEVICE)

        # TODO fix
        if self._use_user_graph:
            path_to_user_graph = os.path.join(graph_dir_path, 'user_graph.npz')
            if os.path.exists(path_to_user_graph):
                self._user_graph = sp.load_npz(path_to_user_graph)
            else:
                user2user_interactions_fst = []
                user2user_interactions_snd = []

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    for connected_user_id in train_item_2_users[item_id]:
                        if user_id != connected_user_id:
                            user2user_interactions_fst.append(user_id)
                            user2user_interactions_snd.append(connected_user_id)

                # (users, user) graph
                user2user_connections = csr_matrix(
                    (
                        np.ones(len(user2user_interactions_fst)),
                        (user2user_interactions_fst, user2user_interactions_snd)),
                    shape=(self._num_users, self._num_users)
                )

                self._user_graph = self.get_sparse_graph_layer(user2user_connections, self._num_users, self._num_users)
                sp.save_npz(path_to_user_graph, self._user_graph)

            self._user_graph = self._convert_sp_mat_to_sp_tensor(self._user_graph).coalesce().to(DEVICE)
        else:
            self._user_graph = None

        if self._use_item_graph:
            path_to_item_graph = os.path.join(graph_dir_path, 'item_graph.npz')
            if os.path.exists(path_to_item_graph):
                self._item_graph = sp.load_npz(path_to_item_graph)
            else:
                item2item_interactions_fst = []
                item2item_interactions_snd = []

                for user_id, item_id in tqdm(zip(self._train_user_interactions, self._train_item_interactions)):
                    for connected_item_id in train_user_2_items[user_id]:
                        if item_id != connected_item_id:
                            item2item_interactions_fst.append(item_id)
                            item2item_interactions_snd.append(connected_item_id)

                # (item, item) graph
                item2item_connections = csr_matrix(
                    (
                        np.ones(len(item2item_interactions_fst)),
                        (item2item_interactions_fst, item2item_interactions_snd)),
                    shape=(self._num_items, self._num_items)
                )
                self._item_graph = self.get_sparse_graph_layer(item2item_connections, self._num_items, self._num_items)
                sp.save_npz(path_to_item_graph, self._item_graph)

            self._item_graph = self._convert_sp_mat_to_sp_tensor(self._item_graph).coalesce().to(DEVICE)
        else:
            self._item_graph = None

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            graph_dir_path=config['graph_dir_path'],
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False)
        )

    @staticmethod
    def get_sparse_graph_layer(sparse_matrix, fst_dim, snd_dim, biparite=False):
        mat_dim_size = fst_dim + snd_dim if biparite else fst_dim

        adj_mat = sp.dok_matrix(
            (mat_dim_size, mat_dim_size),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()

        R = sparse_matrix.tolil()  # list of lists (fst_dim, snd_dim)

        if biparite:
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

        return norm_adj

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

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        meta = {
            'user_graph': self._user_graph,
            'item_graph': self._item_graph,
            'graph': self._graph,
            **self._dataset.meta
        }
        return meta


class DuorecDataset(BaseDataset, config_name='duorec'):

    def __init__(self, dataset):
        self._dataset = dataset
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, _ = self._dataset.get_samplers()

        self.target_2_sequences = defaultdict(list)
        for sample in train_sampler.dataset:
            item_ids = sample['item.ids']

            target_item = item_ids[-1]
            semantic_similar_item_ids = item_ids[:-1]

            self.target_2_sequences[target_item].append(semantic_similar_item_ids)

        train_sampler._target_2_sequences = self.target_2_sequences

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(dataset)

    @property
    def num_users(self):
        return self._dataset.num_users

    @property
    def num_items(self):
        return self._dataset.num_items

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        return self._dataset.meta
