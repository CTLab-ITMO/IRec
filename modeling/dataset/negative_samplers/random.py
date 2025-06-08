from collections import defaultdict

import numpy as np
from dataset.negative_samplers.base import BaseNegativeSampler
from tqdm import tqdm


class RandomNegativeSampler(BaseNegativeSampler, config_name='random'):
    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs["dataset"],
            num_users=kwargs["num_users"],
            num_items=kwargs["num_items"],
        )

    def generate_negative_samples(self, sample, num_negatives):
        user_id = sample["user.ids"][0]
        negatives = set()

        while len(negatives) < num_negatives:
            candidate = np.random.randint(1, self._num_items + 1)
            if candidate not in self._seen_items[user_id]:
                negatives.add(candidate)

        return list(negatives)
