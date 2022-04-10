from utils import MetaParent


class BaseLoss(metaclass=MetaParent):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class IdentityLoss(BaseLoss, config_name='identity'):

    def __call__(self, inputs):
        return inputs
