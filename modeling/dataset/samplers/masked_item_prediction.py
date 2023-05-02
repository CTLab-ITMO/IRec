from dataset.samplers.base import TrainSampler, ValidationSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np


class MaskedItemPredictionTrainSampler(TrainSampler, config_name='masked_item_prediction'):

    def __init__(self, dataset, num_users, num_items, mask_prob=0.0):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = self._num_items + 1
        self._mask_prob = mask_prob

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            mask_prob=config.get('mask_prob', 0.0)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids']

        masked_sequence = []
        labels = []

        for item in item_sequence:
            prob = np.random.rand()

            if prob < self._mask_prob:
                prob /= self._mask_prob

                masked_sequence.append(self._mask_item_idx)
                labels.append(item)
            else:
                masked_sequence.append(item)
                labels.append(0)

        # Mask last item
        masked_sequence[-1] = self._mask_item_idx
        labels[-1] = item_sequence[-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': masked_sequence,
            'item.length': len(masked_sequence),

            'labels.ids': labels,
            'labels.length': len(labels)
        }


class MaskedItemPredictionValidationSampler(ValidationSampler, config_name='masked_item_prediction'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = self._num_items + 1
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
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        sequence = item_sequence[:-1] + [self._mask_item_idx]
        candidates = [item_sequence[-1]] + negatives
        labels = [1] + [0] * len(negatives)

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': sequence,
            'item.length': len(sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }


class MaskedItemPredictionEvalSampler(EvalSampler, config_name='masked_item_prediction'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset, num_users, num_items)
        self._mask_item_idx = self._num_items + 1

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        item_sequence = sample['item.ids']
        labels = [item_sequence[-1]]
        sequence = item_sequence[:-1] + [self._mask_item_idx]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': sequence,
            'item.length': len(sequence),

            'labels.ids': labels,
            'labels.length': len(labels)
        }
