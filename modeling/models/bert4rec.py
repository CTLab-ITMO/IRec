from models.base import TorchModel as Model

from blocks.projector import BaseProjector, TorchProjector as Projector
from blocks.encoder import BaseEncoder, TorchEncoder as Encoder
from blocks.head import BaseHead, TorchHead as Head

from utils import DEVICE

import torch
import torch.nn as nn


class Bert4Rec(Model, config_name='bert4rec'):
    def __init__(
            self,
            projector,
            encoder,
            head,
    ):
        super().__init__()
        self._projector = projector
        self._encoder = encoder
        self._head = head

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None, max_sequence_len=None):
        projector = BaseProjector.create_from_config(
            config['projector'],
            num_users=num_users,
            num_items=num_items,
            max_sequence_len=max_sequence_len
        )
        encoder = BaseEncoder.create_from_config(config['encoder'])
        head = BaseHead.create_from_config(config['head'], num_items=num_items)

        return cls(
            projector=projector,
            encoder=encoder,
            head=head
        )

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoder(inputs)
        return self._head(inputs)


class BertProjector(Projector, config_name='bert4rec'):

    def __init__(
            self,
            prefixes,
            num_users,
            num_items,
            max_sequence_len,
            embedding_dim,
            dropout_rate=0.0,
            output_prefixes=None,
            eps=1e-5
    ):
        super().__init__()
        self._prefixes = prefixes
        self._output_prefixes = output_prefixes or prefixes
        assert len(self._prefixes) == len(self._output_prefixes)

        self._max_sequence_len = max_sequence_len
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout_rate

        self._position_embeddings = nn.Embedding(
            num_embeddings=self._max_sequence_len,
            embedding_dim=self._embedding_dim
        )
        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,  # all items, zero_embedding, mask_embedding
            embedding_dim=self._embedding_dim
        )

        self._dropout = nn.Dropout(p=self._dropout_rate)
        self._layernorms = nn.LayerNorm(embedding_dim, eps)

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None, max_sequence_len=None):
        return cls(
            prefixes=config['prefixes'],
            num_users=num_users,
            num_items=num_items,
            max_sequence_len=max_sequence_len,
            embedding_dim=config['embedding_dim'],
            dropout_rate=config.get('dropout', 0.0),
            output_prefixes=config.get('output_prefixes', None),
            eps=config.get('eps', 1e-5)
        )

    def forward(self, inputs):
        for prefix, output_prefix in zip(self._prefixes, self._output_prefixes):
            all_sequences = inputs['{}.ids'.format(prefix)]  # (all_batch_items)
            all_item_embeddings = self._item_embeddings(all_sequences)  # (all_batch_items, emb_dim)

            if '{}.positions'.format(prefix) in inputs:  # positional embedding
                all_positions = inputs['{}.positions'.format(prefix)]  # (all_batch_items)
                all_positions = all_positions  # (all_batch_items)
                all_position_embeddings = self._position_embeddings(all_positions)  # (all_batch_items, emb_dim)
                all_item_embeddings += all_position_embeddings  # (all_batch_items, emb_dim)

            all_item_embeddings = self._dropout(all_item_embeddings)  # (all_batch_items, emb_dim)

            lengths = inputs['{}.length'.format(prefix)]  # (batch_size)
            batch_size = lengths.shape[0]
            max_sequence_length = lengths.max().item()

            padded_embeddings = torch.zeros(
                batch_size, max_sequence_length, self._embedding_dim,
                dtype=torch.float, device=DEVICE
            )  # (batch_size, max_seq_len, emb_dim)

            mask = torch.arange(
                end=max_sequence_length,
                device=DEVICE
            )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

            padded_embeddings[mask] = all_item_embeddings

            inputs[output_prefix] = self._layernorms(padded_embeddings)
            inputs['{}.mask'.format(output_prefix)] = mask

        return inputs


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_layers,
            num_heads,
            dim_feedforward,
            activation: nn.Module = nn.ReLU(),
            dropout=0.0,
            eps=1e-5
    ):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

    def forward(self, src, attention_mask):
        src = self._encoder(
            src=src,
            src_key_padding_mask=~attention_mask
        )
        return src


class BertEncoder(Encoder, config_name='bert4rec'):

    def __init__(
            self,
            prefix,
            num_layers,
            num_heads,
            hidden_size,
            output_prefix=None,
            activation: nn.Module = nn.ReLU(),
            input_dim=None,
            output_dim=None,
            dropout=0.0,
            eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        self._input_projection = nn.Identity()
        if input_dim is not None:
            self._input_projection = nn.Linear(input_dim, hidden_size)

        self._encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation=activation,
            eps=eps
        )

        self._output_projection = nn.Identity()
        if output_dim is not None:
            self._output_projection = nn.Linear(hidden_size, output_dim)

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config):
        return cls(
            prefix=config['prefix'],
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', config['embedding_dim'] // 64),
            hidden_size=config['embedding_dim'],
            output_prefix=config.get('output_prefix', None),
            input_dim=config.get('input_dim', None),
            output_dim=config.get('output_dim', None),
            dropout=config.get('dropout', 0.0),
            eps=config.get('eps', 1e-5),
            initializer_range=config.get('initializer_range', 0.02)
        )

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    def forward(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, seq_len, emb_dim)
        mask = inputs['{}.mask'.format(self._prefix)]  # (batch_size, seq_len)

        embeddings = self._encoder(src=embeddings, attention_mask=mask)

        inputs[self._output_prefix] = embeddings
        inputs['{}.mask'.format(self._output_prefix)] = mask

        return inputs


class BertHead(Head, config_name='bert4rec'):

    def __init__(
            self,
            prefix,
            labels_prefix,
            input_dim,
            output_dim,
            candidates_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._prefix = prefix
        self._labels_prefix = labels_prefix
        self._candidates_prefix = candidates_prefix
        self._output_prefix = output_prefix or prefix
        self._encoder = nn.Linear(input_dim, output_dim)

    @classmethod
    def create_from_config(cls, config, num_items=None):
        return cls(
            prefix=config['prefix'],
            labels_prefix=config['labels_prefix'],
            input_dim=config['input_dim'],
            output_dim=num_items + 2,
            candidates_prefix=config['candidates_prefix'],
            output_prefix=config.get('prefix', None)
        )

    def forward(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, max_seq_len, input_dim)
        mask = inputs['{}.mask'.format(self._prefix)]  # (batch_size, max_seq_len)

        embeddings = self._encoder(embeddings)  # (batch_size, max_seq_len, output_dim)

        if self.training:  # train mode
            inputs = self._train_postprocessing(inputs, logits=embeddings, logits_mask=mask)
        else:  # eval mode
            inputs = self._eval_postprocessing(inputs, logits=embeddings)

        return inputs

    def _train_postprocessing(self, inputs, logits, logits_mask):
        all_logits = logits[logits_mask]  # (all_events)
        inputs[self._output_prefix] = all_logits

        return inputs

    def _eval_postprocessing(self, inputs, logits):
        batch_size = logits.shape[0]

        lengths = inputs['{}.length'.format(self._prefix)]  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, logits.shape[-1]))  # (batch_size, 1, num_classes)

        last_values = logits.gather(dim=1, index=lengths)  # (batch_size, 1, num_classes)
        last_values = last_values.squeeze(1)  # (batch_size, num_classes)

        candidate_ids = inputs['{}.ids'.format(self._candidates_prefix)]  # (all_candidates)
        candidate_ids = torch.reshape(candidate_ids, (batch_size, -1))  # (batch_size, num_candidates)
        candidate_scores = last_values.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)
        inputs[self._output_prefix] = candidate_scores  # (batch_size, num_candidates)

        labels_ids = inputs['{}.ids'.format(self._labels_prefix)]  # (all_candidates)
        labels_ids = torch.reshape(labels_ids, (batch_size, -1))  # (batch_size, num_candidates)
        inputs['{}.ids'.format(self._labels_prefix)] = labels_ids

        return inputs
