import copy
import torch.nn as nn

from .aggregation import BaseAggregationEncoder


class EventEncoder(nn.Module):

    def __init__(self, prefix, attributes_encoder):
        super().__init__()
        self._prefix = prefix
        self._attributes_encoder = attributes_encoder

    @classmethod
    def create_from_config(cls, config):
        assert 'attributes' in config
        cfg = copy.deepcopy(config)
        cfg['type'] = config['aggregation_type']
        return cls(
            prefix=config['prefix'],
            attributes_encoder=BaseAggregationEncoder.create_from_config(cfg)
        )

    def forward(self, inputs):
        return self._attributes_encoder(inputs)
