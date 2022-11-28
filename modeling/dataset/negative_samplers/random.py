from dataset.negative_samplers.base import BaseNegativeSampler

import numpy as np


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            sample_size=config['sample_size']
        )

    def generate_negative_samples(self, items):
        seen = set(items)
        negative_samples = []

        while len(negative_samples) < self._sample_size:
            item = np.random.randint(1, self._num_items)

            if item not in seen:
                negative_samples.append(item)
                seen.add(item)  # TODO maybe remove ???

        return negative_samples
