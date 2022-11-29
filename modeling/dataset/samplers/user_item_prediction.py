from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np


class PositiveNegativeTrainSampler(TrainSampler, config_name='pos_neg'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=1):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(
            {'type': config['negative_sampler_type']},
            **kwargs
        )

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives', 1)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        assert len(sample['user.ids']) == 1
        positive = sample['item.ids'][0]
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'positive.ids': [positive],
            'positive.length': 1,

            'negative.ids': [np.random.choice(negatives)],
            'negative.length': 1
        }


class UserItemPredictionEvalSampler(EvalSampler, config_name='pos_neg'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(
            {'type': config['negative_sampler_type']},
            **kwargs
        )

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives', 100)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        assert len(sample['user.ids']) == 1
        positive = sample['item.ids']
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        candidates = positive + negatives
        labels = [1] * len(positive) + [0] * len(negatives)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }
