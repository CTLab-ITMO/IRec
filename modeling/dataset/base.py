from utils import MetaParent


class BaseDataset(metaclass=MetaParent):
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class PartialDataset(BaseDataset, config_name='partial'):
    def __init__(self, dataset, ids):
        self._dataset = dataset
        self._ids = ids

    @classmethod
    def create_from_config(cls, config, dataset):
        return cls(dataset=dataset, ids=config['idx'])

    def __getitem__(self, item):
        return self._dataset[self._ids[item]]

    def __len__(self):
        return len(self._ids)
