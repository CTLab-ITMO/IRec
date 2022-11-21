from dataset.base import BaseDataset
from dataset.samplers import TrainSampler, EvalSampler

import os
import logging

logger = logging.getLogger(__name__)


class AmazonDataset(BaseDataset, config_name='amazon'):

    def __init__(
            self,
            dataset,
            interactions,
            max_user_idx,
            max_item_idx,
            max_sequence_length,
            train_sampler,
            validation_sampler,
            test_sampler,
            validation_size=0.1,
            test_size=0.1
    ):
        self._dataset = dataset
        self._interactions = interactions
        self._max_user_idx = max_user_idx
        self._max_item_idx = max_item_idx
        self._max_sequence_length = max_sequence_length

        train_dataset = dataset[:int(len(dataset) * (1.0 - validation_size - test_size))]
        validation_dataset = dataset[
                     int(len(dataset) * (1.0 - validation_size - test_size)):
                     int(len(dataset) * (1.0 - test_size))
                     ]
        test_dataset = dataset[int(len(dataset) * (1.0 - test_size)):]

        logger.info(f'Train dataset size: {len(train_dataset)}')
        logger.info(f'Validation dataset size: {len(validation_dataset)}')
        logger.info(f'Test dataset size: {len(test_dataset)}')

        self._train_sampler = train_sampler.with_dataset(train_dataset)
        self._validation_sampler = validation_sampler.with_dataset(validation_dataset)
        self._test_sampler = test_sampler.with_dataset(test_dataset)

    @classmethod
    def create_from_config(cls, config):
        dataset, num_users, num_items, max_sequence_length, interactions = cls._get_dataset(
            path_to_data_dir=config['path_to_data_dir'],
            dataset_prefix=config['dataset_prefix'],
            min_sample_len=config.get('min_sample_len', 5),
            max_sample_len=config.get('max_sample_len', None)
        )

        train_sampler = TrainSampler.create_from_config(
            config['samplers'],
            num_users=num_users,
            num_items=num_items
        )

        validation_sampler = EvalSampler.create_from_config(
            config['samplers'],
            num_users=num_users,
            num_items=num_items
        )

        test_sampler = EvalSampler.create_from_config(
            config['samplers'],
            num_users=num_users,
            num_items=num_items
        )

        return cls(
            dataset=dataset,
            interactions=interactions,
            max_user_idx=num_users,
            max_item_idx=num_items,
            max_sequence_length=max_sequence_length,
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler
        )

    @staticmethod
    def _get_dataset(
            path_to_data_dir,
            dataset_prefix,
            min_sample_len=5,
            max_sample_len=None
    ):
        max_user_idx = 0
        max_item_idx = 0
        max_sequence_length = 0

        logger.info(f'Amazon {dataset_prefix} dataset creation...')

        dataset = []
        interactions = []  # TODO

        with open(os.path.join(path_to_data_dir, f'{dataset_prefix}.txt'), 'r') as f:
            user_sequences_id = f.readlines()
        with open(os.path.join(path_to_data_dir, f'{dataset_prefix}_timestamps.txt'), 'r') as f_ts:
            user_sequences_timestamp = f_ts.readlines()

        for ids in user_sequences_id:
            user_id, item_ids = ids.split(' ', 1)
            max_user_idx = max(max_user_idx, int(user_id))
            max_item_idx = max(max_item_idx, max([int(item_id) for item_id in item_ids.split(' ')]))
            max_sequence_length = max(max_sequence_length, len(item_ids.split(' ')))

        logger.info(f'Max user idx: {max_user_idx}')
        logger.info(f'Max item idx: {max_item_idx}')
        logger.info(f'Max sequence length: {max_sequence_length}')

        for ids, timestamps in zip(user_sequences_id, user_sequences_timestamp):
            user_id, item_ids = ids.split(' ', 1)
            user_id = int(user_id)

            item_ids = [int(item_id) for item_id in item_ids.split(' ')]
            _, item_timestamps = timestamps.split(' ', 1)

            for item_id in item_ids:
                interactions.append([user_id, item_id])

            item_timestamps = [int(item_timestamp) for item_timestamp in item_timestamps.split(' ')]

            assert len(item_ids) == len(item_timestamps)

            # Sort sample by timestamp
            item_ids, item_timestamps = zip(*sorted(list(zip(item_ids, item_timestamps)), key=lambda x: x[1]))
            item_ids = list(item_ids)
            item_timestamps = list(item_timestamps)

            # Append into dataset
            for separation_idx in range(min_sample_len, len(item_ids)):
                sequence = item_ids[:separation_idx]
                if max_sample_len is not None:
                    sequence = sequence[-max_sample_len:]

                sample = {
                    'user_id': user_id,
                    'timestamp': item_timestamps[separation_idx],

                    'sample.length': len(sequence),
                    'sample.ids': sequence,

                    'answer.length': 1,
                    'answer.ids': [item_ids[separation_idx]]
                }

                dataset.append(sample)

        # Sort dataset by timestamp
        dataset = sorted(dataset, key=lambda x: x['timestamp'])

        logger.info(f'Amazon {dataset_prefix} dataset has been created!')
        logger.info(f'Dataset size: {len(dataset)}')

        return dataset, max_user_idx, max_item_idx, max_sequence_length, interactions

    @property
    def dataset(self):
        return self._dataset

    @property
    def interactions(self):
        return self._interactions

    @property
    def num_users(self):
        return self._max_user_idx

    @property
    def num_items(self):
        return self._max_item_idx

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler
