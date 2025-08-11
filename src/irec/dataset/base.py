from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

from tqdm import tqdm

from irec.dataset.samplers import TrainSampler, EvalSampler

from irec.utils import MetaParent, DEVICE

import pickle
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetPart:
    samples: List[Dict]
    max_user_id: int
    max_item_id: int
    max_sequence_len: int

@dataclass
class EvaluationSet:
    validation: List[Dict]
    test: List[Dict]
    max_user_id: int
    max_item_id: int


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
        max_sequence_length,
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(
            config['path_to_data_dir'],
            config['name'],
        )

        train_result = cls._create_dataset(
            dir_path=data_dir_path,
            part='train',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False),
        )

        validation_result = cls._create_dataset(
            dir_path=data_dir_path,
            part='valid',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False),
        )

        test_result = cls._create_dataset(
            dir_path=data_dir_path,
            part='test',
            max_sequence_length=config['max_sequence_length'],
            use_cached=config.get('use_cached', False),
        )

        max_user_id = max(
            train_result.max_user_id, 
            validation_result.max_user_id, 
            test_result.max_user_id
        )
        max_item_id = max(
            train_result.max_item_id, 
            validation_result.max_item_id, 
            test_result.max_item_id
        )
        max_seq_len = max(
            train_result.max_sequence_len, 
            validation_result.max_sequence_len, 
            test_result.max_sequence_len
        )

        logger.info('Train dataset size: {}'.format(len(train_result.samples)))
        logger.info('Test dataset size: {}'.format(len(test_result.samples)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_seq_len))

        train_interactions = sum(
            map(lambda x: len(x['item.ids']), 
                train_result.samples)
        )  # whole user history as a sample
        valid_interactions = len(
            validation_result.samples
        )  # each new interaction as a sample
        test_interactions = len(
            test_result.samples
        )  # each new interaction as a sample

        logger.info(
            '{} dataset sparsity: {}'.format(
                config['name'],
                (train_interactions + valid_interactions + test_interactions)
                / max_user_id
                / max_item_id,
            ),
        )

        train_sampler = TrainSampler.create_from_config(
        config['samplers'],
        dataset=train_result.samples,
        num_users=max_user_id,
        num_items=max_item_id,
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=validation_result.samples,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_result.samples,
            num_users=max_user_id,
            num_items=max_item_id,
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_seq_len,
        )

    @classmethod
    def _create_dataset(
        cls,
        dir_path,
        part,
        max_sequence_length=None,
        use_cached=False,
    ):
        max_user_id = 0
        max_item_id = 0
        max_sequence_len = 0

        if use_cached and os.path.exists(
            os.path.join(dir_path, '{}.pkl'.format(part)),
        ):
            logger.info(
                f'Take cached dataset from {os.path.join(dir_path, "{}.pkl".format(part))}',
            )

            with open(
                os.path.join(dir_path, '{}.pkl'.format(part)),
                'rb',
            ) as dataset_file:
                dataset, max_user_id, max_item_id, max_sequence_len = (
                    pickle.load(dataset_file)
                )
        else:
            logger.info(
                'Cache is forecefully ignored.'
                if not use_cached
                else 'No cached dataset has been found.',
            )
            logger.info(
                f'Creating a dataset {os.path.join(dir_path, "{}.txt".format(part))}...',
            )

            dataset_path = os.path.join(dir_path, '{}.txt'.format(part))
            with open(dataset_path, 'r') as f:
                data = f.readlines()

            sequence_info = cls._create_sequences(data, max_sequence_length)
            (
                user_sequences,
                item_sequences,
                max_user_id,
                max_item_id,
                max_sequence_len,
            ) = sequence_info

            dataset = []
            for user_id, item_ids in zip(user_sequences, item_sequences):
                dataset.append(
                    {
                        'user.ids': [user_id],
                        'user.length': 1,
                        'item.ids': item_ids,
                        'item.length': len(item_ids),
                    },
                )

            logger.info('{} dataset size: {}'.format(part, len(dataset)))
            logger.info(
                '{} dataset max sequence length: {}'.format(
                    part,
                    max_sequence_len,
                ),
            )

            with open(
                os.path.join(dir_path, '{}.pkl'.format(part)),
                'wb',
            ) as dataset_file:
                pickle.dump(
                    (dataset, max_user_id, max_item_id, max_sequence_len),
                    dataset_file,
                )

        return DatasetPart(
            samples=dataset,
            max_user_id=max_user_id,
            max_item_id=max_item_id,
            max_sequence_len=max_sequence_len
        )
        

    @staticmethod
    def _create_sequences(data, max_sample_len):
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip('\n').split(' ')
            item_ids = [int(item_id) for item_id in sample[1:]][
                -max_sample_len:
            ]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_length,
        )

    def get_samplers(self):
        return (
            self._train_sampler,
            self._validation_sampler,
            self._test_sampler,
        )

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
            'max_sequence_length': self.max_sequence_length,
        }


class GraphDataset(BaseDataset, config_name='graph'):
    def __init__(
        self,
        dataset,
        graph_dir_path,
        use_train_data_only=True,
        use_user_graph=False,
        use_item_graph=False,
        neighborhood_size=None
    ):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph
        self._neighborhood_size = neighborhood_size

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = (
            dataset.get_samplers()
        )

        (
            train_interactions,
            train_user_interactions,
            train_item_interactions,
        ) = [], [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        for sample in train_sampler.dataset:
            user_id = sample['user.ids'][0]
            item_ids = sample['item.ids']

            for item_id in item_ids:
                if (user_id, item_id) not in visited_user_item_pairs:
                    train_interactions.append((user_id, item_id))
                    train_user_interactions.append(user_id)
                    train_item_interactions.append(item_id)

                    train_user_2_items[user_id].add(item_id)
                    train_item_2_users[item_id].add(user_id)

                    visited_user_item_pairs.add((user_id, item_id))

        # TODO create separated function
        if not self._use_train_data_only:
            for sample in validation_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

            for sample in test_sampler.dataset:
                user_id = sample['user.ids'][0]
                item_ids = sample['item.ids']

                for item_id in item_ids:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        if os.path.exists(path_to_graph):
            self._graph = sp.load_npz(path_to_graph)
        else:
            # place ones only when co-occurrence happens
            user2item_connections = csr_matrix(
                (
                    np.ones(len(train_user_interactions)),
                    (train_user_interactions, train_item_interactions),
                ),
                shape=(self._num_users + 2, self._num_items + 2),
            )  # (num_users + 2, num_items + 2), bipartite graph
            self._graph = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True,
            )
            sp.save_npz(path_to_graph, self._graph)

        self._graph = (
            self._convert_sp_mat_to_sp_tensor(self._graph)
            .coalesce()
            .to(DEVICE)
        )

        if self._use_user_graph:
            path_to_user_graph = os.path.join(graph_dir_path, 'user_graph.npz')
            if os.path.exists(path_to_user_graph):
                self._user_graph = sp.load_npz(path_to_user_graph)
            else:
                user2user_interactions_fst = []
                user2user_interactions_snd = []
                visited_user_item_pairs = set()
                visited_user_user_pairs = set()

                for user_id, item_id in tqdm(
                    zip(
                        self._train_user_interactions,
                        self._train_item_interactions,
                    ),
                ):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_user_id in train_item_2_users[item_id]:
                        if (
                            (user_id, connected_user_id)
                            in visited_user_user_pairs
                            or user_id == connected_user_id
                        ):
                            continue  # add (user, user) to graph connections pair only once
                        visited_user_user_pairs.add(
                            (user_id, connected_user_id),
                        )

                        user2user_interactions_fst.append(user_id)
                        user2user_interactions_snd.append(connected_user_id)

                # (user, user) graph
                user2user_connections = csr_matrix(
                    (
                        np.ones(len(user2user_interactions_fst)),
                        (
                            user2user_interactions_fst,
                            user2user_interactions_snd,
                        ),
                    ),
                    shape=(self._num_users + 2, self._num_users + 2),
                )
                print(self._neighborhood_size)
                if self._neighborhood_size is not None:
                    user2user_connections = self._filter_matrix_by_top_k(user2user_connections, self._neighborhood_size)

                self._user_graph = self.get_sparse_graph_layer(
                    user2user_connections,
                    self._num_users + 2,
                    self._num_users + 2,
                    biparite=False,
                )
                sp.save_npz(path_to_user_graph, self._user_graph)

            self._user_graph = (
                self._convert_sp_mat_to_sp_tensor(self._user_graph)
                .coalesce()
                .to(DEVICE)
            )
        else:
            self._user_graph = None

        if self._use_item_graph:
            path_to_item_graph = os.path.join(graph_dir_path, 'item_graph.npz')
            if os.path.exists(path_to_item_graph):
                self._item_graph = sp.load_npz(path_to_item_graph)
            else:
                item2item_interactions_fst = []
                item2item_interactions_snd = []
                visited_user_item_pairs = set()
                visited_item_item_pairs = set()

                for user_id, item_id in tqdm(
                    zip(
                        self._train_user_interactions,
                        self._train_item_interactions,
                    ),
                ):
                    if (user_id, item_id) in visited_user_item_pairs:
                        continue  # process (user, item) pair only once
                    visited_user_item_pairs.add((user_id, item_id))

                    for connected_item_id in train_user_2_items[user_id]:
                        if (
                            (item_id, connected_item_id)
                            in visited_item_item_pairs
                            or item_id == connected_item_id
                        ):
                            continue  # add (item, item) to graph connections pair only once
                        visited_item_item_pairs.add(
                            (item_id, connected_item_id),
                        )

                        item2item_interactions_fst.append(item_id)
                        item2item_interactions_snd.append(connected_item_id)

                # (item, item) graph
                item2item_connections = csr_matrix(
                    (
                        np.ones(len(item2item_interactions_fst)),
                        (
                            item2item_interactions_fst,
                            item2item_interactions_snd,
                        ),
                    ),
                    shape=(self._num_items + 2, self._num_items + 2),
                )

                if self._neighborhood_size is not None:
                    item2item_connections = self._filter_matrix_by_top_k(item2item_connections, self._neighborhood_size)

                self._item_graph = self.get_sparse_graph_layer(
                    item2item_connections,
                    self._num_items + 2,
                    self._num_items + 2,
                    biparite=False,
                )
                sp.save_npz(path_to_item_graph, self._item_graph)

            self._item_graph = (
                self._convert_sp_mat_to_sp_tensor(self._item_graph)
                .coalesce()
                .to(DEVICE)
            )
        else:
            self._item_graph = None

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config['dataset'])
        return cls(
            dataset=dataset,
            graph_dir_path=config['graph_dir_path'],
            use_user_graph=config.get('use_user_graph', False),
            use_item_graph=config.get('use_item_graph', False),
            neighborhood_size=config.get('neighborhood_size', None),
        )

    @staticmethod
    def get_sparse_graph_layer(
        sparse_matrix,
        fst_dim,
        snd_dim,
        biparite=False,
    ):
        if not biparite:
            adj_mat = sparse_matrix.tocsr()
        else:
            R = sparse_matrix.tocsr()
            
            upper_right = R
            lower_left = R.T
            
            upper_left = sp.csr_matrix((fst_dim, fst_dim))
            lower_right = sp.csr_matrix((snd_dim, snd_dim))
            
            adj_mat = sp.bmat([
                [upper_left, upper_right],
                [lower_left, lower_right]
            ])
            assert adj_mat.shape == (fst_dim + snd_dim, fst_dim + snd_dim), (
            f"Got shape {adj_mat.shape}, expected {(fst_dim+snd_dim, fst_dim+snd_dim)}"
            )
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    @staticmethod
    def _filter_matrix_by_top_k(matrix, k):
        mat = matrix.tolil()

        for i in range(mat.shape[0]):
            if len(mat.rows[i]) <= k:
                continue
            data = np.array(mat.data[i])
            
            top_k_indices = np.argpartition(data, -k)[-k:]
            mat.data[i] = [mat.data[i][j] for j in top_k_indices]
            mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]

        return mat.tocsr()
                

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
            **self._dataset.meta,
        }
        return meta


class DuorecDataset(BaseDataset, config_name='duorec'):
    def __init__(self, dataset):
        self._dataset = dataset
        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, _, _ = self._dataset.get_samplers()

        target_2_sequences = defaultdict(list)
        for sample in train_sampler.dataset:
            item_ids = sample['item.ids']

            target_item = item_ids[-1]
            semantic_similar_item_ids = item_ids[:-1]

            target_2_sequences[target_item].append(semantic_similar_item_ids)

        train_sampler._target_2_sequences = target_2_sequences

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


class ScientificDataset(BaseDataset, config_name='scientific'):
    def __init__(
        self,
        train_sampler,
        validation_sampler,
        test_sampler,
        num_users,
        num_items,
        max_sequence_length,
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(
            config['path_to_data_dir'],
            config['name'],
        )
        max_sequence_length = config['max_sequence_length']
        max_user_id, max_item_id = 0, 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        dataset_path = os.path.join(data_dir_path, '{}.txt'.format('all_data'))
        with open(dataset_path, 'r') as f:
            data = f.readlines()

        for sample in data:
            sample = sample.strip('\n').split(' ')
            user_id = int(sample[0])
            item_ids = [int(item_id) for item_id in sample[1:]]

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5

            train_dataset.append(
                {
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids[:-2][-max_sequence_length:],
                    'item.length': len(item_ids[:-2][-max_sequence_length:]),
                },
            )
            assert len(item_ids[:-2][-max_sequence_length:]) == len(
                set(item_ids[:-2][-max_sequence_length:]),
            )
            validation_dataset.append(
                {
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids[:-1][-max_sequence_length:],
                    'item.length': len(item_ids[:-1][-max_sequence_length:]),
                },
            )
            assert len(item_ids[:-1][-max_sequence_length:]) == len(
                set(item_ids[:-1][-max_sequence_length:]),
            )
            test_dataset.append(
                {
                    'user.ids': [user_id],
                    'user.length': 1,
                    'item.ids': item_ids[-max_sequence_length:],
                    'item.length': len(item_ids[-max_sequence_length:]),
                },
            )
            assert len(item_ids[-max_sequence_length:]) == len(
                set(item_ids[-max_sequence_length:]),
            )

        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))
        logger.info('Max user id: {}'.format(max_user_id))
        logger.info('Max item id: {}'.format(max_item_id))
        logger.info('Max sequence length: {}'.format(max_sequence_length))
        logger.info(
            '{} dataset sparsity: {}'.format(
                config['name'],
                (len(train_dataset) + len(test_dataset))
                / max_user_id
                / max_item_id,
            ),
        )

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_sequence_length,
        )

    def get_samplers(self):
        return (
            self._train_sampler,
            self._validation_sampler,
            self._test_sampler,
        )

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
            'max_sequence_length': self.max_sequence_length,
        }

class PreSplitDataReader:
    def __init__(self, data_dir: str, max_seq_len: int = None):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

    def read_train_data(self, part_name) -> DatasetPart:
        filepath = os.path.join(self.data_dir, part_name)
        sequences, max_user, max_item = self._read_sequences_file(
            filepath, self.max_seq_len
        )
        
        samples = [
            {
                'user.ids': [uid],
                'user.length': 1,
                'item.ids': seq,
                'item.length': len(seq)
            }
            for uid, seq in sequences.items()
        ]
        
        max_len = self.max_seq_len if self.max_seq_len is not None else -1
        
        return DatasetPart(
            samples=samples,
            max_user_id=max_user,
            max_item_id=max_item,
            max_sequence_len=max_len
        )

    def read_evaluation_data(self) -> EvaluationSet:
        valid_hist, u2, i2 = self._read_sequences_file(
            os.path.join(self.data_dir, 'valid_history.txt'), self.max_seq_len
        )
        valid_trg, u3, i3 = self._read_sequences_file(
            os.path.join(self.data_dir, 'valid_target.txt'), self.max_seq_len
        )
        validation_dataset = [
            {'user.ids': [uid], 'history': valid_hist.get(uid, []), 'target': valid_trg.get(uid, [])}
            for uid in valid_hist
        ]
        
        test_hist, u4, i4 = self._read_sequences_file(
            os.path.join(self.data_dir, 'test_history.txt'), self.max_seq_len
        )
        test_trg, u5, i5 = self._read_sequences_file(
            os.path.join(self.data_dir, 'test_target.txt'), self.max_seq_len
        )
        test_dataset = [
            {'user.ids': [uid], 'history': test_hist.get(uid, []), 'target': test_trg.get(uid, [])}
            for uid in test_hist
        ]

        max_user = max(u2, u3, u4, u5)
        max_item = max(i2, i3, i4, i5)
        
        return EvaluationSet(validation_dataset, test_dataset, max_user, max_item)

    @staticmethod
    def _read_sequences_file(filepath, max_len=None):
        sequences = {}
        max_user, max_item = 0, 0
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]
                if max_len:
                    item_ids = item_ids[-max_len:]
                sequences[user_id] = item_ids
                max_user = max(max_user, user_id)
                if item_ids:
                    max_item = max(max_item, max(item_ids))
        return sequences, max_user, max_item

class MCLSRDataset(BaseDataset, config_name='mclsr'):
    def __init__(self, train_sampler, validation_sampler, test_sampler, num_users, num_items, max_sequence_length):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        reader = PreSplitDataReader(
            data_dir=os.path.join(config['path_to_data_dir'], config['name']),
            max_seq_len=config.get('max_sequence_length')
        )
        train_data = reader.read_train_data('train_mclsr.txt')
        eval_data = reader.read_evaluation_data()
        

        num_users = max(train_data.max_user_id, eval_data.max_user_id)
        num_items = max(train_data.max_item_id, eval_data.max_item_id)
        

        user_to_all_seen_items = defaultdict(set)
        for sample in train_data.samples:
            user_to_all_seen_items[sample['user.ids'][0]].update(sample['item.ids'])
        kwargs['user_to_all_seen_items'] = user_to_all_seen_items


        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            dataset=train_data.samples,
            num_users=num_users,
            num_items=num_items,
            **kwargs
        )
        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=eval_data.validation,
            num_users=num_users,
            num_items=num_items,
            **kwargs
        )
        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            dataset=eval_data.test,
            num_users=num_users,
            num_items=num_items,
            **kwargs
        )

        return cls(
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            config.get('max_sequence_length')
        )

    def get_samplers(self):
        return (self._train_sampler, self._validation_sampler, self._test_sampler)
    
    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def meta(self):
        return {'num_users': self.num_users, 'num_items': self.num_items, 'max_sequence_length': self._max_sequence_length}
    
class SASRecDataset(BaseDataset, config_name='sasrec_comparison'):
    def __init__(self, train_sampler, validation_sampler, test_sampler, num_users, num_items, max_sequence_length):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config['path_to_data_dir'], config['name'])
        max_seq_len = config.get('max_sequence_length')

        sequence_reader = PreSplitDataReader(
            dir_path=data_dir,
            max_sequence_length=max_seq_len
        )
        eval_reader = PreSplitDataReader(
            data_dir=data_dir,
            max_seq_len=max_seq_len
        )

        train_data = sequence_reader.read_train_data('train_sasrec.txt')
        eval_data = eval_reader.read_evaluation_data()


        num_users = max(train_data.max_user_id, eval_data.max_user_id)
        num_items = max(train_data.max_item_id, eval_data.max_item_id)


        train_sampler = TrainSampler.create_from_config(
            config['train_sampler'],
            dataset=train_data.samples, 
            num_users=num_users, 
            num_items=num_items
        )
        validation_sampler = EvalSampler.create_from_config(
            config['eval_sampler'],
            dataset=eval_data.validation, 
            num_users=num_users, 
            num_items=num_items
        )
        test_sampler = EvalSampler.create_from_config(
            config['eval_sampler'],
            dataset=eval_data.test, 
            num_users=num_users, 
            num_items=num_items
        )

        return cls(train_sampler,
                   validation_sampler,
                   test_sampler,
                   num_users,
                   num_items,
                   max_seq_len
        )

    def get_samplers(self):
        return (self._train_sampler,
                self._validation_sampler, 
                self._test_sampler
        )
    
    @property
    def num_users(self): return self._num_users

    @property
    def num_items(self): return self._num_items

    @property
    def meta(self):
        return {'num_users': self.num_users,
                'num_items': self.num_items,
                'max_sequence_length': self._max_sequence_length
        }
