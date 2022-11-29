from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def generate_negative_samples(self, sample, num_negatives):
        user_id = sample['user.ids'][0]
        negatives = []
        while len(negatives) < num_negatives:
            negative_idx = np.random.randint(1, self._num_items)
            if negative_idx not in self._seen_items[user_id]:
                negatives.append(negative_idx)

        return negatives
