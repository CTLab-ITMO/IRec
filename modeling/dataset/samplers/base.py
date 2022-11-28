from utils import MetaParent


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


class EvalSampler(metaclass=MetaParent):

    def __init__(self):
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError
