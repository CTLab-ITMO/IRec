from utils import MetaParent


class BaseDataset(metaclass=MetaParent):
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
