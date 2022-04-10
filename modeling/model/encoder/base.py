from utils import MetaParent

import torch


class BaseEncoder(metaclass=MetaParent):
    pass


class TorchEncoder(BaseEncoder, torch.nn.Module):
    pass
