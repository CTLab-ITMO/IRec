from utils import MetaParent


class TrainSampler(metaclass=MetaParent):

    def with_dataset(self, dataset):
        self._dataset = dataset
        return self

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError


class EvalSampler(metaclass=MetaParent):

    def with_dataset(self, dataset):
        self._dataset = dataset
        return self

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError
