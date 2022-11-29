from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np
from collections import defaultdict


def _get_seen_items(dataset):
    seen = defaultdict(list)

    for sample in dataset:
        user_id = sample['user.ids'][0]
        items = sample['item.ids']
        seen[user_id].extend(items)

    return seen


class PositiveNegativeTrainSampler(TrainSampler, config_name='pos_neg'):

    def __init__(self, dataset, num_users, num_items, negative_sampler):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._seen_items = _get_seen_items(dataset)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(config['negative_sampler'], **kwargs)
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        assert len(sample['user.ids']) == 1
        user = sample['user.ids'][0]
        positive = sample['item.ids'][0]
        negative = np.random.choice(self._negative_sampler.generate_negative_samples(user, self._seen_items[user]))

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'positive.ids': [positive],
            'positive.length': 1,

            'negative.ids': [negative],
            'negative.length': 1
        }


class UserItemPredictionEvalSampler(EvalSampler, config_name='pos_neg'):

    def __init__(self, dataset, num_users, num_items, negative_sampler):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._seen_items = _get_seen_items(dataset)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config(config['negative_sampler'], **kwargs)
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        assert len(sample['user.ids']) == 1
        user = sample['user.ids'][0]
        positive = sample['item.ids']
        negatives = self._negative_sampler.generate_negative_samples(user, self._seen_items[user])

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
