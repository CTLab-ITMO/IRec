import time

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class TensorboardWriter(SummaryWriter):
    pass


class TensorboardTimer():
    def __init__(self, scope_name):
        pass


    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finish_time = time.time()
