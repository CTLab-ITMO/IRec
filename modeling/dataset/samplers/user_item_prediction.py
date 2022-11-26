from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np


class UserItemPredictionTrainSampler(TrainSampler, config_name='user_item_prediction'):

    def __init__(self, num_users, num_items, negative_sampler):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler

    def with_dataset(self, dataset):
        self._negative_sampler = self._negative_sampler.with_dataset(dataset)
        return super().with_dataset(dataset)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(config['negative_sampler'], **kwargs)
        return cls(
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        sequence = sample['sample.ids']
        answer = sample['answer.ids']

        positive = np.random.choice(sequence + answer)
        negative = np.random.choice(self._negative_sampler.generate_negative_samples(sequence, answer))

        return {
            'user.ids': [sample['user_id']],
            'user.length': 1,

            'positive.ids': [positive],
            'positive.length': 1,

            'negative.ids': [negative],
            'negative.length': 1,

            'timestamp': sample['timestamp']
        }


class UserItemPredictionEvalSampler(EvalSampler, config_name='user_item_prediction'):

    def __init__(self, num_users, num_items, negative_sampler):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler

    def with_dataset(self, dataset):
        self._negative_sampler = self._negative_sampler.with_dataset(dataset)
        return super().with_dataset(dataset)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(config['negative_sampler'], **kwargs)
        return cls(
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        sequence = sample['sample.ids']
        answer = sample['answer.ids']

        negatives = self._negative_sampler.generate_negative_samples(sequence, answer)

        candidates = answer + negatives
        labels = [1] * len(answer) + [0] * len(negatives)

        return {
            'user.ids': [sample['user_id']],
            'user.length': 1,

            'timestamp': sample['timestamp'],

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }
