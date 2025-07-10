from irec.dataset.samplers.base import TrainSampler, EvalSampler

import copy
import numpy as np
from collections import defaultdict


class MCLSRTrainSampler(TrainSampler, config_name='mclsr'):
    def __init__(self, dataset, num_users, num_items, num_negatives=100):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._num_negatives = num_negatives
        self._all_items = list(range(1, num_items + 1))
        self._user_to_all_seen_items = defaultdict(set)
        for sample in self._dataset:
            user_id = sample['user.ids'][0]
            self._user_to_all_seen_items[user_id].update(sample['item.ids'])

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

        user_id = sample['user.ids'][0]
        item_sequence = sample['item.ids'][:-1]
        positive_item = sample['item.ids'][-1]

        seen_items = self._user_to_all_seen_items[user_id]
        
        negatives = []
        while len(negatives) < self._num_negatives:
            candidates = np.random.choice(self._all_items, size=self._num_negatives * 2, replace=False) 
            
            unseen_candidates = [item for item in candidates if item not in seen_items]
            negatives.extend(unseen_candidates)
            
        negatives = negatives[:self._num_negatives]

        return {
            'user.ids': [user_id],
            'user.length': sample['user.length'],
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': [positive_item],
            'labels.length': 1,
            'negatives.ids': negatives,
            'negatives.length': len(negatives),
        }


class MCLSRPredictionEvalSampler(EvalSampler, config_name='mclsr'):
    def __init__(self, dataset, num_users, num_items):
        super().__init__(dataset, num_users, num_items)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
        )
    
    def __getitem__(self, index):
        sample = self._dataset[index]
        history_sequence = sample['history']
        target_items = sample['target']

        return {
            'user.ids': sample['user.ids'],
            'user.length': 1,
            'item.ids': history_sequence,
            'item.length': len(history_sequence),
            'labels.ids': target_items,
            'labels.length': len(target_items),
        }
