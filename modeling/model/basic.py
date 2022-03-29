from .base import TorchModel


class BasicModel(TorchModel, config_name='basic'):

    def __init__(self, msg):
        super().__init__()

        self._msg = msg
        print(msg)
