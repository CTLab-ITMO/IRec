from utils import MetaParent

import copy


class TrainSampler(metaclass=MetaParent):

    def __init__(self):
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError
    

class MultiDomainTrainSampler(TrainSampler):

    def __init__(self, target_domain, other_domains):
        super().__init__()
        self._target_domain = target_domain
        self._other_domains = other_domains

    @property
    def dataset(self, domain):
        return self._dataset[domain]

    def __len__(self, domain=None):
        if domain is None:
            return len(self._dataset[self._target_domain])
        return len(self._dataset[domain])

    def __getitem__(self, index):
        raise NotImplementedError


class ValidationSampler(metaclass=MetaParent):

    def __init__(self):
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError
    

class MultiDomainValidationSampler(ValidationSampler):

    def __init__(self, target_domain, other_domains):
        super().__init__()
        self._target_domain = target_domain
        self._other_domains = other_domains

    @property
    def dataset(self, domain):
        return self._dataset[domain]

    def __len__(self, domain=None):
        if domain is None:
            return len(self._dataset[self._target_domain])
        return len(self._dataset[domain])

    def __getitem__(self, index):
        raise NotImplementedError


class EvalSampler(metaclass=MetaParent):

    def __init__(self, dataset, num_users, num_items):
        super().__init__()
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = copy.deepcopy(self._dataset[index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [next_item],
            'labels.length': 1
        }


class MultiDomainEvalSampler(EvalSampler):

    def __init__(self, 
                 dataset, 
                 num_users, 
                 num_items, 
                 target_domain, 
                 other_domains
    ):
        super().__init__(dataset, num_users, num_items)
        self._target_domain = target_domain
        self._other_domains = other_domains

    def __len__(self, domain=None):
        if domain is None:
            return len(self._dataset[self._target_domain])
        return len(self._dataset[domain])

    def __getitem__(self, index, domain):
        sample = copy.deepcopy(self._dataset[domain][index])

        item_sequence = sample['item.ids'][:-1]
        next_item = sample['item.ids'][-1]

        return {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'labels.ids': [next_item],
            'labels.length': 1
        }
