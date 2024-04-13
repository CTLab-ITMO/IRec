from models.base import TorchModel

from utils import create_masked_tensor, get_activation_function

import torch
from torch import nn


class GRUModel(TorchModel):

    def __init__(
            self,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_layers,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5
    ):
        super().__init__()
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self._hidden_to_output_projection = nn.Linear(embedding_dim, num_items)
        self._activation = get_activation_function(activation)

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = torch.arange(
            start=seq_len - 1, end=-1, step=-1, device=mask.device
        )[None].tile([batch_size, 1]).long()  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        packed_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=embeddings,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        hidden = torch.zeros(
            self._num_layers, batch_size, self._embedding_dim,
            dtype=embeddings.dtype,
            device=embeddings.device,
            requires_grad=True
        )  # (num_layers, batch_size, embedding_dim)
        out, hidden = self._encoder(packed_embeddings, hidden)
        embeddings, embedding_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)
        embedding_lengths = embedding_lengths.to(lengths.device)

        assert torch.allclose(lengths, embedding_lengths)

        return embeddings, mask


class GRU4RecModel(GRUModel, config_name='gru4rec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_layers,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps
        )

        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'tanh'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            events=all_sample_events,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events
            )  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events
            )  # (all_batch_events, embedding_dim)

            positive_scores = torch.einsum(
                'ad,ad->a',
                all_sample_embeddings,
                all_positive_sample_embeddings
            )
            negative_scores = torch.einsum(
                'ad,ad->a',
                all_sample_embeddings,
                all_negative_sample_embeddings
            )

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores,
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
