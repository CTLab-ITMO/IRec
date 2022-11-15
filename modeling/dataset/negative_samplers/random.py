from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            num_users=num_users,
            num_items=num_items,
            sample_size=config['sample_size']
        )

    def generate_negative_samples(self, sequence, answer):
        seen = set(sequence + answer)

        negative_samples = []

        for _ in range(self._sample_size):
            item = np.random.randint(1, self._num_items)

            while item in seen:
                item = np.random.randint(1, self._num_items)

            negative_samples.append(item)
            seen.add(item)

        return negative_samples
