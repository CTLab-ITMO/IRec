from blocks.encoder import TorchEncoder as Encoder, Transformer

import torch
import torch.nn as nn


class BertEncoder(Encoder, config_name='bert4rec'):

    def __init__(
            self,
            prefix,
            num_layers,
            num_heads,
            hidden_size,
            output_prefix=None,
            activation='relu',
            input_dim=None,
            output_dim=None,
            dropout=0.0,
            eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._encoder = Transformer(
            prefix=prefix,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            input_dim=input_dim,
            output_dim=output_dim,
            output_prefix=output_prefix,
            initializer_range=initializer_range
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            prefix=config['prefix'],
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', config['embedding_dim'] // 64),
            hidden_size=config['embedding_dim'],
            activation=config.get('activation', 'relu'),
            output_prefix=config.get('output_prefix', None),
            input_dim=config.get('input_dim', None),
            output_dim=config.get('output_dim', None),
            dropout=config.get('dropout', 0.0),
            eps=config.get('eps', 1e-5),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        return self._encoder(inputs)
