from utils import MetaParent

import torch.nn as nn


class BaseModel(metaclass=MetaParent):
    pass


class TorchModel(nn.Module, BaseModel):
    pass
