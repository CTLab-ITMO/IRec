from models.base import TorchModel

from blocks.projector import BaseProjector, TorchProjector as Projector
from blocks.encoder import BaseEncoder, TorchEncoder as Encoder

from utils import DEVICE

import torch
import torch.nn as nn


class GRU4Rec(TorchModel, config_name='gru4rec'):

    def __init__(self, projector, encoder, head):
        super().__init__()
        self._projector = projector
        self._encoder = encoder
        self._head = head

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projector = BaseProjector.create_from_config(config['projector'], **kwargs)
        encoder = BaseEncoder.create_from_config(config['encoder'], **kwargs)
        return cls(projector=projector, encoder=encoder)

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoder(inputs)
        return self._head(inputs)


class GRU4RecProjector(Projector, config_name='gru4rec'):

    def __init__(
            self,
            prefix,
            num_users,
            num_items,
            max_sequence_len,
            embedding_dim,
            droupout=0.0,
            layernorm_eps=1e-5,
            output_prefix=None
    ):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        self._max_sequence_len = max_sequence_len
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._droupout = droupout

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,  # all items, zero_embedding, mask_embedding
            embedding_dim=self._embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=self._max_sequence_len,
            embedding_dim=self._embedding_dim
        )

        self._dropout = nn.Dropout(p=self._dropout_rate)
        self._layernorm = nn.Identity()
        if self._user_layernorm:
            self._layernorm = nn.LayerNorm(self._embedding_dim, eps=layernorm_eps)

    def forward(self, inputs):
        sample_embeddings = inputs['{}.ids'.format(self._prefix)]  # (all_batch_items)
        sample_embeddings = self._item_embeddings(sample_embeddings)  # (all_batch_items, emb_dim)

        if '{}.positions'.format(self._prefix) in inputs:
            if '{}.positions'.format(self._sample_prefix) in inputs:  # positional encoding
                sample_positions = inputs['{}.positions'.format(self._sample_prefix)]  # (all_batch_items)
                sample_positions = self._position_embeddings(sample_positions)  # (all_batch_items, emb_dim)
                sample_embeddings += sample_positions  # (all_batch_items, emb_dim)

        sample_embeddings = self._dropout(sample_embeddings)

        sample_lengths = inputs['{}.length'.format(self._prefix)]  # (batch_size)
        batch_size = sample_lengths.shape[0]
        max_batch_sequence_length = sample_lengths.max().item()

        padded_embeddings = torch.zeros(
            batch_size, max_batch_sequence_length, self._embedding_dim,
            dtype=torch.float, device=DEVICE
        )  # (batch_size, seq_len, emb_dim)

        mask = torch.arange(
            end=max_batch_sequence_length,
            device=DEVICE
        )[None].tile([batch_size, 1]) < sample_lengths[:, None]  # (batch_size, seq_len)

        padded_embeddings[mask] = sample_embeddings

        inputs[self._output_prefix] = padded_embeddings
        inputs['{}.mask'.format(self._output_prefix)] = mask

        return inputs


class GRU4RecEncoder(Encoder, config_name='gru4rec'):

    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        # TODO implement
