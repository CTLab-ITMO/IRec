import torch
from torch import nn

from utils.registry import MetaParent

from .attributes import BaseAttributeEncoder


class BaseAggregationEncoder(metaclass=MetaParent):
    pass


class TorchAggregationEncoder(BaseAggregationEncoder, nn.Module):

    def __init__(self):
        super().__init__()


class SumAggregationEncoder(TorchAggregationEncoder, config_name='sum'):

    def __init__(self, encoders):
        super().__init__()
        self._attributes = []
        for attribute, encoder in encoders.items():
            self.add_module(attribute, encoder)
            self._attributes.append(attribute)

    @classmethod
    def create_from_config(cls, config):
        assert 'attributes' in config and len(config['attributes']) > 0

        attribute_encoders = dict()
        for cfg in config['attributes']:
            attribute_name = cfg['field']
            cfg['field'] = '{}.{}'.format(config['prefix'], attribute_name)
            cfg['embedding_dim'] = config['embedding_dim']
            attribute_encoders[attribute_name] = BaseAttributeEncoder.create_from_config(cfg)

        return cls(encoders=attribute_encoders)

    def forward(self, inputs):
        embeddings = None
        for attribute in self._attributes:
            attribute_embeddings = getattr(self, attribute)(inputs)
            if embeddings is None:
                embeddings = attribute_embeddings
            else:
                embeddings += attribute_embeddings

        return embeddings
