from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy
import numpy as np


class NextItemPredictionTrainSampler(TrainSampler, config_name='next_item_prediction'):

    def __init__(self, negative_sampler):
        super().__init__()
        self._negative_sampler = negative_sampler

    def with_dataset(self, dataset):
        self._negative_sampler = self._negative_sampler.with_dataset(dataset)
        return super().with_dataset(dataset)

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        negative_sampler = BaseNegativeSampler.create_from_config(
            config['negative_sampler'], num_users=num_users, num_items=num_items
        )
        return cls(negative_sampler=negative_sampler)

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        sequence = sample['sample.ids']
        answer = sample['answer.ids']

        next_item_sequence = sequence + answer
        next_item_sequence = next_item_sequence[1:]

        negative_sequence = []

        for idx in range(len(sequence)):
            negatives = self._negative_sampler.generate_negative_samples(sequence, answer)
            negative_sequence.append(np.random.choice(negatives))

        assert len(sequence) == len(next_item_sequence) == len(negative_sequence)

        return {
            'user.ids': [sample['user_id']],
            'user.length': 1,

            'timestamp': sample['timestamp'],

            'sample.ids': sequence,
            'sample.length': len(sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),

            'positive_labels.ids': [1] * len(next_item_sequence),
            'positive_labels.length': len(next_item_sequence),

            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),

            'negative_labels.ids': [0] * len(negative_sequence),
            'negative_labels.length': len(negative_sequence),
        }


class NextItemPredictionEvalSampler(EvalSampler, config_name='next_item_prediction'):

    def __init__(self, negative_sampler):
        super().__init__()
        self._negative_sampler = negative_sampler

    def with_dataset(self, dataset):
        self._negative_sampler = self._negative_sampler.with_dataset(dataset)
        return super().with_dataset(dataset)

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        negative_sampler = BaseNegativeSampler.create_from_config(
            config['negative_sampler'], num_users=num_users, num_items=num_items
        )
        return cls(negative_sampler=negative_sampler)

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

            'sample.ids': sequence,
            'sample.length': len(sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }
