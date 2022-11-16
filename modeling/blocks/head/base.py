from utils import MetaParent

import torch.nn as nn


class BaseHead(metaclass=MetaParent):
    pass


class IdentityHead(BaseHead, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class TorchHead(BaseHead, nn.Module):
    pass


class CompositeHead(TorchHead, config_name='composite'):

    def __init__(self, heads):
        super().__init__()
        self._heads = heads

    @classmethod
    def create_from_config(cls, config):
        heads_cfg = config['heads']
        shared_params = config['shared']

        for shared_key, shared_value in shared_params.items():
            for head_cfg in heads_cfg:
                if shared_key not in head_cfg:
                    head_cfg[shared_key] = shared_value

        return cls(heads=nn.ModuleList([
            BaseHead.create_from_config(head_cfg)
            for head_cfg in heads_cfg
        ]))

    def forward(self, inputs):
        for head in self._heads:
            inputs = head(inputs)
        return inputs
