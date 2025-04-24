from dataset.samplers.base import TrainSampler, EvalSampler
from dataset.negative_samplers.base import BaseNegativeSampler

import torch
import copy
from collections import defaultdict
import numpy as np


class DatasetAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.item_freq = None  # Словарь для хранения частот
        self.all_items_count = 0

    def precompute_frequencies(self):
        """Подсчет частот встречаемости item_id в dataset"""
        freq = defaultdict(int)

        # Проходим по всем элементам dataset
        for sample in self.dataset:
            for item_id in sample['item.ids']:
                freq[item_id] += 1
            self.all_items_count += len(sample['item.ids'])

        self.item_freq = freq

    def get_frequency(self, item_ids):
        """Возвращает список частот для каждого item_id из списка item_ids"""
        if not isinstance(item_ids, list):
            raise TypeError("item_ids должен быть списком")

        # Для каждого item_id в списке возвращаем его частоту или 0, если его нет
        return [self.item_freq.get(item_id, 0) for item_id in item_ids]


class NextItemPredictionTrainSampler(TrainSampler, config_name='next_item_prediction'):

    def __init__(self, dataset, num_users, num_items, negative_sampler, num_negatives=0):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._negative_sampler = negative_sampler
        self._num_negatives = num_negatives
        self._frequency_counter = DatasetAnalyzer(dataset)
        self._frequency_counter.precompute_frequencies()

    @classmethod
    def create_from_config(cls, config, **kwargs):
        negative_sampler = BaseNegativeSampler.create_from_config({'type': config['negative_sampler_type']}, **kwargs)

        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            negative_sampler=negative_sampler,
            num_negatives=config.get('num_negatives_train', 0)
        )

    # here add how often each element in batch is occur
    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item_sequence = sample['item.ids'][1:]

        if self._num_negatives == 0:
            return {
                'user.ids': sample['user.ids'],
                'user.length': sample['user.length'],

                'item.ids': item_sequence,
                'item.length': len(item_sequence),

                'item_counts.ids': self._frequency_counter.get_frequency(item_sequence),
                'item_counts.length': len(item_sequence),

                'positive.ids': next_item_sequence,
                'positive.length': len(next_item_sequence),

                'positive_counts.ids': self._frequency_counter.get_frequency(next_item_sequence),
                'positive_counts.length': len(next_item_sequence),

                'counts.ids': [self._frequency_counter.all_items_count],
                'counts.length': 1,
            }
        else:
            negative_sequence = self._negative_sampler.generate_negative_samples(
                sample, self._num_negatives
            )

            return {
                'user.ids': sample['user.ids'],
                'user.length': sample['user.length'],

                'item.ids': item_sequence,
                'item.length': len(item_sequence),

                'positive.ids': next_item_sequence,
                'positive.length': len(next_item_sequence),

                'negative.ids': negative_sequence,
                'negative.length': len(negative_sequence)
            }


class NextItemPredictionEvalSampler(EvalSampler, config_name='next_item_prediction'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )
