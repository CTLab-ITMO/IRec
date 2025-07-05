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
        positive_item = sample['item.ids'][-1]

        seen_items = set(item_sequence)
        seen_items.add(positive_item)
        
        negatives = []
        while len(negatives) < self._num_negatives:
            random_item_id = np.random.choice(self._all_items) 
            
            if random_item_id not in seen_items:
                negatives.append(random_item_id)

        return {
            'user.ids': sample['user.ids'],
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
