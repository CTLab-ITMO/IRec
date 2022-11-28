from utils import MetaParent


class BaseNegativeSampler(metaclass=MetaParent):

    def __init__(
            self,
            num_users,
            num_items,
            sample_size,
    ):
        self._dataset = None
        self._num_users = num_users
        self._num_items = num_items
        self._sample_size = sample_size

    def generate_negative_samples(self, items):
        raise NotImplementedError
