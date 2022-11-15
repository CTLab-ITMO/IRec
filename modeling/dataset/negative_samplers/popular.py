from dataset.negative_samplers.base import BaseNegativeSampler

from collections import Counter


class PopularNegativeSampler(BaseNegativeSampler, config_name='popular'):

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

        self._popular_items = self._items_by_popularity()

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            num_users=num_users,
            num_items=num_items,
            sample_size=config['sample_size']
        )

    def _items_by_popularity(self):
        popularity = Counter()

        for sample in self._dataset:
            popularity.update(sample['sequence'])

        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items

    def generate_negative_samples(self, sequence, answer):
        seen = set(sequence + answer)
        negative_samples = []

        popularity_idx = 0
        while len(negative_samples) < self._sample_size:
            item = self._popular_items[popularity_idx]
            popularity_idx += 1

            if item not in seen:
                negative_samples.append(item)
                seen.add(item)

        return negative_samples
