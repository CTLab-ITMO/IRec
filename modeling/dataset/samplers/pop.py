from dataset.samplers.base import TrainSampler, ValidationSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import copy


class PopTrainSampler(TrainSampler, config_name='pop'):
    pass


class PopValidationSampler(ValidationSampler, config_name='pop'):

    def __init__(self, dataset, negative_sampler, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives

        self._item_2_count = {}
        for sample in dataset:
            items = sample['item.ids'][:-1]
            for item in items:
                if item not in self._item_2_count:
                    self._item_2_count[item] = 0
                self._item_2_count[item] += 1

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives', 100)
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        positive = sample['item.ids'][-1]
        negatives = self._negative_sampler.generate_negative_samples(sample, self._num_negatives)

        candidates = [positive] + negatives
        labels = [1] + [0] * len(negatives)

        candidates_counts = [self._item_2_count[candidate] for candidate in candidates]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'candidates.ids': candidates,
            'candidates.length': len(candidates),

            'labels.ids': labels,
            'labels.length': len(labels),

            'candidates_counts.ids': candidates_counts,
            'candidates_counts.length': len(candidates_counts)
        }


class PopEvalSampler(EvalSampler, config_name='pop'):

    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset, num_users, num_items)

        self._item_2_count = {}
        for i in range(1, num_items + 1):
            self._item_2_count[i] = 0

        for sample in dataset:
            # items = sample['item.ids'][:-1]
            items = sample['item.ids']
            for item in items:
                self._item_2_count[item] += 1

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])
        labels = [sample['item.ids'][-1]]
        candidates_counts = [0] + [
            self._item_2_count[item_id] for item_id in range(1, self._num_items + 1)
        ]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'labels.ids': labels,
            'labels.length': len(labels),

            'candidates_counts.ids': candidates_counts,
            'candidates_counts.length': len(candidates_counts)
        }
