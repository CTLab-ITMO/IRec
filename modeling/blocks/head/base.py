from utils import MetaParent

import torch.nn as nn


class BaseHead(metaclass=MetaParent):
    pass


class TorchHead(BaseHead, nn.Module):
    pass


class IdentityHead(BaseHead, config_name='identity'):

    def __call__(self, inputs):
        return inputs
