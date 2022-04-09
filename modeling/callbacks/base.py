from utils import MetaParent


class BaseCallback(metaclass=MetaParent):

    def __init__(self):
        pass

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def on_step(self):
        pass
