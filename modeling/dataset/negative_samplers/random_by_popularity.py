from dataset.negative_samplers.base import BaseNegativeSampler

from collections import Counter
import torch


class RandomByPopularityNegativeSampler(BaseNegativeSampler, config_name='random_by_popularity'):

    def __init__(
            self,
            dataset,
            num_users,
            num_items
    ):
        super().__init__(
            dataset=dataset,
            num_users=num_users,
            num_items=num_items
        )

        self._item_popularity = self._compute_item_popularity()

    @classmethod
    def create_from_config(cls, _, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items']
        )

    def _compute_item_popularity(self):
        popularity = Counter()

        for sample in self._dataset:
            for item_id in sample['item.ids']:
                popularity[item_id] += 1

        # Convert to tensor for efficient sampling
        items = list(popularity.keys())
        weights = torch.tensor(list(popularity.values()), dtype=torch.float32)
        return {"items": items, "weights": weights}

    def generate_negative_samples(self, sample, num_negatives):
        user_id = sample['user.ids'][0]
        seen_items = set(self._seen_items[user_id])  # Convert to set for faster lookup

        items = self._item_popularity["items"]
        weights = self._item_popularity["weights"]

        negatives = []
        while len(negatives) < num_negatives:
            sampled_indices = torch.multinomial(weights, num_samples=num_negatives, replacement=False)
            sampled_items = [items[idx] for idx in sampled_indices]

            for item in sampled_items:
                if item not in seen_items and len(negatives) < num_negatives:
                    negatives.append(item)

        return negatives
