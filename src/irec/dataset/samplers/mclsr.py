from irec.dataset.samplers.base import TrainSampler, EvalSampler

from collections import defaultdict
import random


class MCLSRTrainSampler(TrainSampler, config_name='mclsr'):
    def __init__(self, dataset, num_users, num_items, user_to_all_seen_items, num_negatives, **kwargs):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._num_negatives = num_negatives
        self._all_items_set = set(range(1, num_items + 1))
        self._user_to_all_seen_items = user_to_all_seen_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        num_negatives = config['num_negatives_train']
        print(num_negatives)
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            num_negatives=num_negatives,
            user_to_all_seen_items=kwargs['user_to_all_seen_items'],
        )


    def __getitem__(self, index):
        sample = self._dataset[index]

        user_id = sample['user.ids'][0]
        item_sequence = sample['item.ids'][:-1]
        positive_item = sample['item.ids'][-1]

        user_seen = self._user_to_all_seen_items[user_id]

        unseen_items = list(self._all_items_set - user_seen)
        
        negatives = random.sample(unseen_items, self._num_negatives)
        

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
