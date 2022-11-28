from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np
from tqdm import trange
from collections import defaultdict


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    def __init__(
            self,
            num_users,
            num_items,
            sample_size
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            sample_size=sample_size
        )

        print('Precomputing negatives...')
        self._unseen_items = self._compute_negatives()
        print('Precomputing negatives done!')

    def _compute_negatives(self):
        unseen_items = defaultdict(set)

        for user_id in trange(self._num_users + 1):
            for item_id in range(self._num_items + 1):
                unseen_items[user_id].add(item_id)

        for sample in self._dataset:
            user_id = sample['item.ids'][0]
            for item_id in sample['item.ids']:
                unseen_items[user_id].remove(item_id)

        return unseen_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            sample_size=config['sample_size']
        )

    def generate_negative_samples(self, _):
        return np.random.choice(self._unseen_items, self._sample_size)
