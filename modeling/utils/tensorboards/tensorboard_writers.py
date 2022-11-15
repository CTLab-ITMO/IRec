import time

from torch.utils.tensorboard import SummaryWriter

LOGS_DIR = '../tensorboard_logs'


class TensorboardWriter(SummaryWriter):

    def __init__(self):
        super().__init__(LOGS_DIR)


GLOBAL_TENSORBOARD_WRITER = TensorboardWriter()


class TensorboardTimer:

    def __init__(self, scope):
        super().__init__(LOGS_DIR)
        self._scope = scope

    def __enter__(self):
        self.start = int(time.time() * 10000)
        return self

    def __exit__(self, *args):
        self.end = int(time.time() * 10000)
        interval = (self.end - self.start) / 10.
        GLOBAL_TENSORBOARD_WRITER.add_scalar(self._scope, interval)
