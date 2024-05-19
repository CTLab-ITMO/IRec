from utils import MetaParent
from numpy import arange
from collections import defaultdict


class BaseNegRatingsNegativeSampler(metaclass=MetaParent):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            items_popularity,
            negative_items_popularity,
            positive_domain
    ):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._all_items = arange(self._num_items)
        self._items_popularity = items_popularity
        self._negative_items_popularity = negative_items_popularity
        self._positive_domain = positive_domain

        self._seen_items = defaultdict(set)
        for sample in self._dataset[self._positive_domain]:
            user_id = sample['user.ids'][0]
            items = list(sample['item.ids'])
            self._seen_items[user_id].update(items)

    def generate_negative_samples(self, sample, num_negatives):
        raise NotImplementedError
