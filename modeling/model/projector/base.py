from utils import MetaParent


class BaseProjector(metaclass=MetaParent):
    pass


class IdentityProjector(BaseProjector, config_name='identity'):

    def __call__(self, inputs):
        return inputs
