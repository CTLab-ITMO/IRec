from utils import MetaParent


class BaseNegativeSampler(metaclass=MetaParent):

    def __init__(
            self,
            dataset,
            num_users,
            num_items,
            sample_size,
    ):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items
        self._sample_size = sample_size

    def generate_negative_samples(self, user_id, items):
        raise NotImplementedError
