from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class BertTrainSampler(TrainSampler, config_name='bert'):

    def __init__(
            self,
            num_users,
            num_items,
            mask_prob=0.25
    ):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = num_items + 1
        self._mask_prob = mask_prob

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            num_users=num_users,
            num_items=num_items,
            mask_prob=config.get('mask_prob', 0.25)
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]

        sequence = sample['sample.ids']
        answer = sample['answer.ids']

        masked_sequence = []
        labels = []

        for item in sequence:
            prob = np.random.rand()

            if prob < self._mask_prob:
                prob /= self._mask_prob

                if prob < 0.8:
                    masked_sequence.append(self._mask_item_idx)
                elif prob < 0.9:
                    masked_sequence.append(np.random.randint(1, self._num_items))
                else:
                    masked_sequence.append(item)

                labels.append(item)
            else:
                masked_sequence.append(item)
                labels.append(0)

        masked_sequence = masked_sequence + [self._mask_item_idx]
        labels = labels + answer

        sample['sample.ids'] = masked_sequence
        sample['sample.length'] = len(masked_sequence)

        sample['labels.ids'] = labels
        sample['labels.length'] = len(labels)

        return {
            'user_id': sample['user_id'],
            'timestamp': sample['timestamp'],

            'sample.ids': masked_sequence,
            'sample.length': len(masked_sequence),

            'labels.ids': labels,
            'labels.length': len(labels)
        }


class BertEvalSampler(EvalSampler, config_name='bert'):

    def __init__(
            self,
            num_users,
            num_items,
            negative_sampler
    ):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._mask_item_idx = num_items + 1
        self._negative_sampler = negative_sampler

    def with_dataset(self, dataset):
        self._negative_sampler = self._negative_sampler.with_dataset(dataset)
        return super().with_dataset(dataset)

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        negative_sampler = BaseNegativeSampler.create_from_config(
            config['negative_sampler'], num_users=num_users, num_items=num_items
        )

        return cls(
            num_users=num_users,
            num_items=num_items,
            negative_sampler=negative_sampler
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]

        sequence = sample['sample.ids']
        answer = sample['answer.ids']

        negatives = self._negative_sampler.generate_negative_samples(sequence, answer)

        sequence = sequence + [self._mask_item_idx]
        candidates = answer + negatives
        labels = [1] * len(answer) + [0] * len(negatives)

        return {
            'user_id': sample['user_id'],
            'timestamp': sample['timestamp'],

            'sample.ids': sequence,
            'sample.length': len(sequence),

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),
        }
