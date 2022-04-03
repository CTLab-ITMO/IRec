from utils import MetaParent


class BaseHead(metaclass=MetaParent):
    pass


class IdentityHead(BaseHead, config_name='identity'):

    def __call__(self, inputs):
        return inputs
