from numpy import setdiff1d, random

from dataset.negative_samplers import BaseNegRatingsNegativeSampler
from torch import tensor


class NegativeRatingsNegativeSampler(BaseNegRatingsNegativeSampler, config_name='negative_ratings_negative_sampler'):

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            dataset=kwargs['dataset'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            items_popularity=kwargs['items_popularity'],
            positive_domain=kwargs['positive_domain']
        )

    def generate_negative_samples(self, sample, num_negatives):
        users_items = sample['item.ids']
        none_interactions = setdiff1d(self._all_items, users_items)
        probabilities = self._items_popularity[none_interactions] / self._items_popularity[none_interactions].sum()
        temp = (tensor(random.choice(none_interactions, num_negatives, replace=True, p=probabilities))).long()
        return temp
