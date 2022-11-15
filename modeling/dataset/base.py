from utils import MetaParent


class BaseDataset(metaclass=MetaParent):

    def get_samplers(self):
        raise NotImplementedError
