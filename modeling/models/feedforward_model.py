from models.base import TorchModel

from blocks.projector import BaseProjector
from blocks.encoder import BaseEncoder, CompositeEncoder

import torch


class FeedForwardModel(TorchModel, config_name='feedforward'):
    def __init__(
            self,
            projector,
            encoders
    ):
        super().__init__()
        self._projector = projector
        self._encoders = encoders

    @classmethod
    def create_from_config(
            cls,
            config,
            num_users=None,
            num_items=None,
            max_sequence_len=None
    ):
        projector = BaseProjector.create_from_config(
            config['projector'],
            num_users=num_users,
            num_items=num_items,
            max_sequence_len=max_sequence_len
        )

        encoders = CompositeEncoder(encoders=torch.nn.ModuleList([
            BaseEncoder.create_from_config(cfg)
            for cfg in config['encoders']
        ]))

        return cls(projector=projector, encoders=encoders)

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoders(inputs)
        return inputs
