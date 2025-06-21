from irec.dataset.samplers.base import TrainSampler, EvalSampler

import copy
import numpy as np


class MCLSRTrainSampler(TrainSampler, config_name='mclsr'):
    def __init__(self, dataset, num_users, num_items, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._num_negatives = num_negatives
        self._all_items = np.arange(1, num_items + 1)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        num_negatives = config.get('num_negatives_train', 100)
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            num_negatives=num_negatives,
        )

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        seen_items = set(item_sequence + [next_item])
        
        negatives = []
        while len(negatives) < self._num_negatives:
            candidates = np.random.choice(self._all_items, size=self._num_negatives * 2) 
            
            for item_id in candidates:
                if item_id not in seen_items:
                    negatives.append(item_id)
                    if len(negatives) == self._num_negatives:
                        break

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': [next_item],
            'labels.length': 1,
            'negatives.ids': negatives,
            'negatives.length': len(negatives),
        }


class MCLSRPredictionEvalSampler(EvalSampler, config_name='mclsr'):
    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )
