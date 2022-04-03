from utils import MetaParent


class BaseMetric(metaclass=MetaParent):
    pass


class StaticMetric(BaseMetric, config_name='static'):
    def __init__(self, value):
        self._value = value

    def __call__(self, predict, ground_truth):
        return self._value
