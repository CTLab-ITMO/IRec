from utils import MetaParent


class BaseLoss(metaclass=MetaParent):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class IdentityLoss(BaseLoss, config_name='identity'):
    def __init__(self, loss_prefix):
        self._loss_prefix = loss_prefix

    def __call__(self, inputs):
        inputs[self._loss_prefix].backward()
        inputs[self._loss_prefix] = inputs[self._loss_prefix].item()
        return inputs
