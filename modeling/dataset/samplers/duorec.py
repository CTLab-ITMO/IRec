import random

from dataset.samplers.base import TrainSampler, ValidationSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy


class DuorecTrainSampler(TrainSampler, config_name='duorec'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids']

        target_item = item_sequence[-1]
        item_sequence = item_sequence[:-1]

        # There is a probability of sampling the same sequence, but it is not a problem
        semantic_similar_sequence = random.choice(self._target_2_sequences[target_item])

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': [target_item],
            'positive.length': 1,

            'semantic_similar_item.ids': semantic_similar_sequence,
            'semantic_similar_item.length': len(semantic_similar_sequence)
        }


class DuoRecValidationSampler(ValidationSampler, config_name='duorec'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives', 100)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids']

        positive = item_sequence[-1]
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)
        item_sequence = item_sequence[:-1]

        candidates = [positive] + negatives
        labels = [1] + [0] * len(negatives)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }


class DuoRecEvalSampler(EvalSampler, config_name='duorec'):
    pass
