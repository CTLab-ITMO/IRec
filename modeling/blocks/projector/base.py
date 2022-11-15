from utils import MetaParent
# from model.projector.event import EventEncoder

import torch
import torch.nn as nn


class BaseProjector(metaclass=MetaParent):
    pass


class IdentityProjector(BaseProjector, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class TorchProjector(BaseProjector, torch.nn.Module):
    pass


class BasicProjector(TorchProjector, config_name='basic'):

    def __init__(
            self,
            encoder,
            prefix,
            embedding_dim,
            output_prefix=None,
            layernorm=False,
            layernorm_eps=1e-5,
            one_directional=False,
            dropout_rate=0.0
    ):
        super().__init__()
        self._encoder = encoder
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix
        self._embedding_dim = embedding_dim
        self._one_directional = one_directional

        self._layernorm = nn.Identity()
        if layernorm:
            self._layernorm = nn.LayerNorm(self.embedding_dim, eps=layernorm_eps)
        self._dropout = nn.Dropout(p=dropout_rate)

    @classmethod
    def create_from_config(cls, config):
        encoder = EventEncoder.create_from_config(config)

        return cls(
            encoder=encoder,
            prefix=config['prefix'],
            embedding_dim=config['embedding_dim'],
            output_prefix=config.get('output_prefix', None),
            layernorm=config.get('layernorm', False),
            layernorm_eps=config.get('eps', 1e-5)
        )

    def forward(self, inputs):
        embeddings = self._encoder(inputs)

        batch_size = inputs['{}.length'.format(self._prefix)].shape[0]
        max_sequence_length = inputs['{}.length'.format(self._prefix)].max().item()

        padded_embeddings = embeddings.new_zeros(batch_size, max_sequence_length, self._embedding_dim)
        mask = BasicProjector.sequence_mask(inputs['{}.length'.format(self._prefix)], max_sequence_length)

        # if self._one_directional:
        #     mask = ~torch.tril(mask, diagonal=)  # TODO

        padded_embeddings[mask] = embeddings

        return {
            self._output_prefix: self._dropout(self._layernorm(padded_embeddings)),
            '{}.mask'.format(self._output_prefix): mask,
            '{}.length'.format(self._output_prefix): inputs['{}.length'.format(self._prefix)]
        }

    @staticmethod
    def sequence_mask(lengths, maxlen=None):
        batch_size = lengths.shape[0]
        if maxlen is None:
            maxlen = lengths.max().item()
        return torch.arange(end=maxlen, device=lengths.device)[None].tile([batch_size, 1]) < lengths[:, None]


class CompositeProjector(TorchProjector, config_name='composite'):

    def __init__(self, projectors):
        super().__init__()
        self._projectors = projectors

    @classmethod
    def create_from_config(cls, config):
        projectors_cfg = config['projectors']
        shared_params = config['shared']

        for shared_key, shared_value in shared_params.items():
            for projector_cfg in projectors_cfg:
                if shared_key not in projector_cfg:
                    projector_cfg[shared_key] = shared_value

        return cls(projectors=[
            BasicProjector.create_from_config(projector_cfg)
            for projector_cfg in projectors_cfg
        ])

    def forward(self, inputs):
        embeddings = {}

        for projector in self._projectors:
            for k, v in projector(inputs).items():
                embeddings[k] = v

        return embeddings
