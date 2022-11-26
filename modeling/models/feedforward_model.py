from models.base import TorchModel

from blocks.projector import BaseProjector
from blocks.encoder import BaseEncoder, CompositeEncoder

import torch
import torch.nn as nn


class FeedForwardModel(TorchModel, config_name='feedforward'):
    def __init__(
            self,
            projector,
            encoders
    ):
        super().__init__()
        self._projector = nn.Identity()
        if projector is not None:
            self._projector = projector

        self._encoders = nn.Identity()
        if encoders is not None:
            self._encoders = encoders

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projector = BaseProjector.create_from_config(config['projector'], **kwargs) if 'projector' in config else None
        encoders = CompositeEncoder(encoders=torch.nn.ModuleList([
            BaseEncoder.create_from_config(cfg, **kwargs)
            for cfg in config['encoders']
        ])) if 'encoders' in config else None

        return cls(projector=projector, encoders=encoders)

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoders(inputs)
        return inputs
