from utils import MetaParent, DEVICE
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


class CompositeProjector(TorchProjector, config_name='composite'):

    def __init__(self, projectors):
        super().__init__()
        self._projectors = projectors

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projectors_cfg = config['projectors']
        shared_params = config['shared']

        for shared_key, shared_value in shared_params.items():
            for projector_cfg in projectors_cfg:
                if shared_key not in projector_cfg:
                    projector_cfg[shared_key] = shared_value

        return cls(projectors=nn.ModuleList([
            BaseProjector.create_from_config(projector_cfg, **kwargs)
            for projector_cfg in projectors_cfg
        ]))

    def forward(self, inputs):
        for projector in self._projectors:
            inputs = projector(inputs)
        return inputs


class BasicProjector(TorchProjector, config_name='basic'):

    def __init__(
            self,
            embedding_dim,
            num_users,
            num_items,
            max_sequence_len=None,
            dropout=0.0,
            eps=1e-5,
            fields=None,
            initializer_range=0.02
    ):
        super().__init__()
        self._fields = fields or []

        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_len = max_sequence_len

        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._eps = eps

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2,
            embedding_dim=self._embedding_dim
        )

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim
        )

        self._position_embeddings = nn.Identity()
        if self._max_sequence_len is not None:  # TODO fix
            self._position_embeddings = nn.Embedding(
                num_embeddings=self._max_sequence_len,
                embedding_dim=self._embedding_dim
            )

        self._dropout = nn.Dropout(p=self._dropout)
        self._layernorm = nn.LayerNorm(self._embedding_dim, eps=self._eps)

        self._init_weights(initializer_range)

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        nn.init.trunc_normal_(
            self._user_embeddings.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

        nn.init.trunc_normal_(
            self._item_embeddings.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

        if self._max_sequence_len is not None:  # TODO fix
            nn.init.trunc_normal_(
                self._position_embeddings.weight.data,
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range
            )

        nn.init.ones_(self._layernorm.weight.data)
        nn.init.zeros_(self._layernorm.bias.data)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fields=config.get('fields', None),
            embedding_dim=config['embedding_dim'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_len=kwargs.get('max_sequence_length', None),
            dropout=config.get('dropout', 0.0),
            eps=config.get('eps', 1e-5),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        for field in self._fields:
            prefix = field['prefix']
            output_prefix = field.get('output_prefix', prefix)
            use_position = field.get('use_position', False)
            use_layernorm = field.get('use_layernorm', False)
            type = field.get('type', 'item')

            if '{}.ids'.format(prefix) in inputs:
                all_batch_items = inputs['{}.ids'.format(prefix)]  # (all_batch_items)
                all_batch_lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

                if type == 'item':
                    all_batch_embeddings = self._item_embeddings(all_batch_items)  # (all_batch_items, emb_dim)
                elif type == 'user':
                    all_batch_embeddings = self._user_embeddings(all_batch_items)  # (all_batch_items, emb_dim)
                else:
                    raise ValueError('Unknown embedding type')

                if use_position and '{}.positions'.format(prefix) in inputs:
                    all_batch_positions = inputs['{}.positions'.format(prefix)]  # (all_batch_items)
                    all_position_embeddings = self._position_embeddings(all_batch_positions)  # (all_batch_items, emb_dim)
                    all_batch_embeddings += all_position_embeddings  # (all_batch_items, emb_dim)

                all_batch_embeddings = self._dropout(all_batch_embeddings)  # (all_batch_items, emb_dim)

                batch_size = all_batch_lengths.shape[0]
                max_sequence_length = all_batch_lengths.max().item()

                padded_embeddings = torch.zeros(
                    batch_size, max_sequence_length, self._embedding_dim,
                    dtype=torch.float, device=DEVICE
                )  # (batch_size, max_seq_len, emb_dim)

                mask = torch.arange(
                    end=max_sequence_length,
                    device=DEVICE
                )[None].tile([batch_size, 1]) < all_batch_lengths[:, None]  # (batch_size, max_seq_len)

                padded_embeddings[mask] = all_batch_embeddings

                if use_layernorm:
                    padded_embeddings = self._layernorm(padded_embeddings)

                inputs[output_prefix] = padded_embeddings  # (batch_size, max_seq_len, emb_dim)
                inputs['{}.mask'.format(output_prefix)] = mask  # (batch_size, max_seq_len)

        return inputs
