from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np
from tqdm import trange
from collections import defaultdict


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            sample_size
    ):
        super().__init__(
            dataset=dataset,
            num_users=num_users,
            num_items=num_items,
            sample_size=sample_size
        )

        print('Precomputing negatives...')
        self._unseen_items = self._compute_negatives()
        print('Precomputing negatives done!')

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            sample_size=config['sample_size']
        )

    def _compute_negatives(self):
        unseen_items = defaultdict(set)

        for user_id in trange(self._num_users + 1):
            for item_id in range(self._num_items + 1):
                unseen_items[user_id].add(item_id)

        for sample in self._dataset:
            user_id = sample['item.ids'][0]
            for item_id in sample['item.ids']:
                if item_id in unseen_items[user_id]:
                    unseen_items[user_id].remove(item_id)

        return unseen_items

    def generate_negative_samples(self, user_id, items):  # TODO implement caching
        return np.random.choice(list(self._unseen_items[user_id]), self._sample_size)
