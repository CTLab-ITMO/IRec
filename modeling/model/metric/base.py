from utils import MetaParent


class BaseMetric(metaclass=MetaParent):
    pass


class StaticMetric(BaseMetric, config_name='static'):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs
