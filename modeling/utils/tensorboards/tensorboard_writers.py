import time

from torch.utils.tensorboard import SummaryWriter

LOGS_DIR = '../tensorboard_logs'
GLOBAL_TENSORBOARD_WRITER = None


class TensorboardWriter(SummaryWriter):

    def __init__(self, experiment_name):
        super().__init__('{}/{}'.format(LOGS_DIR, experiment_name))


class TensorboardTimer:

    def __init__(self, scope):
        super().__init__(LOGS_DIR)
        self._scope = scope

    def __enter__(self):
        self.start = int(time.time() * 10000)
        return self

    def __exit__(self, *args):
        self.end = int(time.time() * 10000)
        interval = (self.end - self.start) / 10.0
        GLOBAL_TENSORBOARD_WRITER.add_scalar(self._scope, interval)
