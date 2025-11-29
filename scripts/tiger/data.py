from collections import defaultdict
import json
from loguru import logger
import numpy as np
from pathlib import Path


import pyarrow as pa
import pyarrow.feather as feather

import torch

from irec.data.base import BaseDataset


class Dataset:
    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_items,
            max_sequence_length
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_timestamp_based(
            cls,
            train_json_path,
            validation_json_path,
            test_json_path,
            max_sequence_length,
            sampler_type,
            min_sample_len=2,
            is_extended=False
    ):
        max_item_id = 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        with open(validation_json_path, 'r') as f:
            validation_data = json.load(f)
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)

        all_users = set(train_data.keys()) | set(validation_data.keys()) | set(test_data.keys())

        for user_id_str in all_users:
            user_id = int(user_id_str)

            train_items = train_data.get(user_id_str, [])
            validation_items = validation_data.get(user_id_str, [])
            test_items = test_data.get(user_id_str, [])

            full_sequence = train_items + validation_items + test_items
            if full_sequence:
                max_item_id = max(max_item_id, max(full_sequence))

            assert len(full_sequence) >= 5, f'Core-5 dataset is used, user {user_id} has only {len(full_sequence)} items'

            if is_extended:
                # sample = [1, 2]
                # sample = [1, 2, 3]
                # sample = [1, 2, 3, 4]
                # sample = [1, 2, 3, 4, 5]
                # sample = [1, 2, 3, 4, 5, 6]
                # sample = [1, 2, 3, 4, 5, 6, 7]
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                for prefix_length in range(min_sample_len, len(train_items) + 1):
                    train_dataset.append({
                        'user.ids': [user_id],
                        'item.ids': train_items[:prefix_length],
                    })
            else:
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                train_dataset.append({
                    'user.ids': [user_id],
                    'item.ids': train_items,
                })

            # разворачиваем каждый айтем из валидации в отдельный сэмпл
            # Пример: Train=[1,2], Valid=[3,4]
            # sample = [1, 2, 3]
            # sample = [1, 2, 3, 4]

            current_history = train_items.copy()
            for item in validation_items:
                # эвал датасет сам отрезает таргет потом
                sample_sequence = current_history + [item]

                if len(sample_sequence) >= min_sample_len:
                    validation_dataset.append({
                        'user.ids': [user_id],
                        'item.ids': sample_sequence,
                    })
                current_history.append(item)

            # разворачиваем каждый айтем из теста в отдельный сэмпл
            # Пример: Train=[1,2], Valid=[3,4], Test=[5, 6]
            # sample = [1, 2, 3, 4, 5]
            # sample = [1, 2, 3, 4, 5, 6]
            current_history = train_items + validation_items

            for item in test_items:
                # эвал датасет сам отрезает таргет потом
                sample_sequence = current_history + [item]

                if len(sample_sequence) >= min_sample_len:
                    test_dataset.append({
                        'user.ids': [user_id],
                        'item.ids': sample_sequence,
                    })

                current_history.append(item)

        logger.debug(f'Train dataset size: {len(train_dataset)}')
        logger.debug(f'Validation dataset size: {len(validation_dataset)}')
        logger.debug(f'Test dataset size: {len(test_dataset)}')

        train_sampler = TrainDataset(train_dataset, sampler_type, max_sequence_length=max_sequence_length)
        validation_sampler = EvalDataset(validation_dataset, max_sequence_length=max_sequence_length)
        test_sampler = EvalDataset(test_dataset, max_sequence_length=max_sequence_length)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_items=max_item_id + 1,  # +1 added because our ids are 0-indexed
            max_sequence_length=max_sequence_length
        )

    @classmethod
    def create(cls, inter_json_path, max_sequence_length, sampler_type, is_extended=False):
        max_item_id = 0
        train_dataset, validation_dataset, test_dataset = [], [], []

        with open(inter_json_path, 'r') as f:
            user_interactions = json.load(f)

        for user_id_str, item_ids in user_interactions.items():
            user_id = int(user_id_str)

            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5, f'Core-5 dataset is used, user {user_id} has only {len(item_ids)} items'

            # sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (leave one out scheme, 8 - train, 9 - valid, 10 - test)
            if is_extended:
                # sample = [1, 2]
                # sample = [1, 2, 3]
                # sample = [1, 2, 3, 4]
                # sample = [1, 2, 3, 4, 5]
                # sample = [1, 2, 3, 4, 5, 6]
                # sample = [1, 2, 3, 4, 5, 6, 7]
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                for prefix_length in range(2, len(item_ids) - 2 + 1):
                    train_dataset.append({
                        'user.ids': [user_id],
                        'item.ids': item_ids[:prefix_length],
                    })
            else:
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                train_dataset.append({
                    'user.ids': [user_id],
                    'item.ids': item_ids[:-2],
                })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            validation_dataset.append({
                'user.ids': [user_id],
                'item.ids': item_ids[:-1],
            })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            test_dataset.append({
                'user.ids': [user_id],
                'item.ids': item_ids,
            })

        logger.debug(f'Train dataset size: {len(train_dataset)}')
        logger.debug(f'Validation dataset size: {len(validation_dataset)}')
        logger.debug(f'Test dataset size: {len(test_dataset)}')
        logger.debug(f'Max item id: {max_item_id}')

        train_sampler = TrainDataset(train_dataset, sampler_type, max_sequence_length=max_sequence_length)
        validation_sampler = EvalDataset(validation_dataset, max_sequence_length=max_sequence_length)
        test_sampler = EvalDataset(test_dataset, max_sequence_length=max_sequence_length)

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_items=max_item_id + 1,  # +1 added because our ids are 0-indexed
            max_sequence_length=max_sequence_length
        )

    def get_datasets(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length


class TrainDataset(BaseDataset):
    def __init__(self, dataset, prediction_type, max_sequence_length):
        self._dataset = dataset
        self._prediction_type = prediction_type
        self._max_sequence_length = max_sequence_length

        self._transforms = {
            'sasrec': self._all_items_transform,
            'tiger': self._last_item_transform
        }

    def _all_items_transform(self, sample):
        item_sequence = sample['item.ids'][-self._max_sequence_length:][:-1]
        next_item_sequence = sample['item.ids'][-self._max_sequence_length:][1:]
        return {
            'user.ids': np.array(sample['user.ids'], dtype=np.int64),
            'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),
            'item.ids': np.array(item_sequence, dtype=np.int64),
            'item.length': np.array([len(item_sequence)], dtype=np.int64),
            'labels.ids': np.array(next_item_sequence, dtype=np.int64),
            'labels.length': np.array([len(next_item_sequence)], dtype=np.int64)
        }

    def _last_item_transform(self, sample):
        item_sequence = sample['item.ids'][-self._max_sequence_length:][:-1]
        last_item = sample['item.ids'][-self._max_sequence_length:][-1]
        return {
            'user.ids': np.array(sample['user.ids'], dtype=np.int64),
            'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),
            'item.ids': np.array(item_sequence, dtype=np.int64),
            'item.length': np.array([len(item_sequence)], dtype=np.int64),
            'labels.ids': np.array([last_item], dtype=np.int64),
            'labels.length': np.array([1], dtype=np.int64),
        }

    def __getitem__(self, index):
        return self._transforms[self._prediction_type](self._dataset[index])

    def __len__(self):
        return len(self._dataset)


class EvalDataset(BaseDataset):
    def __init__(self, dataset, max_sequence_length):
        self._dataset = dataset
        self._max_sequence_length = max_sequence_length

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]

        item_sequence = sample['item.ids'][-self._max_sequence_length:][:-1]
        next_item = sample['item.ids'][-self._max_sequence_length:][-1]

        return {
            'user.ids': np.array(sample['user.ids'], dtype=np.int64),
            'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),
            'item.ids': np.array(item_sequence, dtype=np.int64),
            'item.length': np.array([len(item_sequence)], dtype=np.int64),
            'labels.ids': np.array([next_item], dtype=np.int64),
            'labels.length': np.array([1], dtype=np.int64),
            'visited.ids': np.array(sample['item.ids'][:-1], dtype=np.int64),
            'visited.length': np.array([len(sample['item.ids'][:-1])], dtype=np.int64),
        }


class ArrowBatchDataset(BaseDataset):
    def __init__(self, batch_dir, device='cuda', preload=False):
        self.batch_dir = Path(batch_dir)
        self.device = device

        all_files = list(self.batch_dir.glob('batch_*_len_*.arrow'))

        batch_files_map = defaultdict(list)
        for f in all_files:
            batch_id = int(f.stem.split('_')[1])
            batch_files_map[batch_id].append(f)
        
        for batch_id in batch_files_map:
            batch_files_map[batch_id].sort()
        
        self.batch_indices = sorted(batch_files_map.keys())

        if preload:
            print(f"Preloading {len(self.batch_indices)} batches...")
            self.cached_batches = []
            
            for idx in range(len(self.batch_indices)):
                batch = self._load_batch(batch_files_map[self.batch_indices[idx]])
                self.cached_batches.append(batch)                
        else:
            self.cached_batches = None
            self.batch_files_map = batch_files_map
    
    def _load_batch(self, arrow_files):
        batch = {}

        for arrow_file in arrow_files:
            table = feather.read_table(arrow_file)
            metadata = table.schema.metadata or {}
            
            for col_name in table.column_names:
                col = table.column(col_name)
                
                shape_key = f'{col_name}_shape'
                dtype_key = f'{col_name}_dtype'
                
                if shape_key.encode() in metadata:
                    shape = eval(metadata[shape_key.encode()].decode())
                    dtype = np.dtype(metadata[dtype_key.encode()].decode())
                    
                    # Проверяем тип колонки
                    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                        arr = np.array(col.to_pylist(), dtype=dtype)
                    else:
                        arr = col.to_numpy().reshape(shape).astype(dtype)
                else:
                    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                        arr = np.array(col.to_pylist())
                    else:
                        arr = col.to_numpy()
                    
                batch[col_name] = torch.from_numpy(arr.copy()).to(self.device)
        
        return batch
    
    def __len__(self):
        return len(self.batch_indices)
    
    def __getitem__(self, idx):
        if self.cached_batches is not None:
            return self.cached_batches[idx]
        else:
            batch_id = self.batch_indices[idx]
            arrow_files = self.batch_files_map[batch_id]
            return self._load_batch(arrow_files)
