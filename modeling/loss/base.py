from utils import MetaParent


class BaseLoss(metaclass=MetaParent):
    pass


class TupleLoss(BaseLoss, config_name='tuple'):
    def __init__(self, idx):
        self._idx = idx

    def __call__(self, predict, ground_truth):
        return predict[self._idx]
