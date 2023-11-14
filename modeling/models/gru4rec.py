from models.base import TorchModel

from utils import create_masked_tensor

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
            layer_norm_eps=1e-5,
            is_bidirectional=False
    ):
        super().__init__()
        self._is_bidirectional = is_bidirectional
        self._num_items = num_items

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
            bidirectional=is_bidirectional
        )

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]
        embedding_dim = embeddings.shape[2]

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
            self._num_layers, batch_size, embedding_dim,
            dtype=embeddings.dtype,
            device=embeddings.device,
            requires_grad=True
        )  # (num_layers, batch_size, embedding_dim)
        out, hidden = self._encoder(packed_embeddings, hidden)
        embeddings, embedding_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)
        embedding_lengths = embedding_lengths.to(lengths.device)

        # TODO remove this assert
        assert torch.allclose(lengths, embedding_lengths)

        return embeddings, mask


class GRU4RecModel(GRUModel, config_name='gru4rec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_layers,
            dropout=0.0,
            activation='tanh',
            layer_norm_eps=1e-5,
            is_bidirectional=False,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_bidirectional=is_bidirectional
        )

        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'tanh'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            is_bidirectional=config.get('is_bidirectional', False),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            events=all_sample_events,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

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

            return {
                'current_embeddings': all_sample_embeddings,
                'positive_embeddings': all_positive_sample_embeddings,
                'negative_embeddings': all_negative_sample_embeddings
            }
        else:  # eval mode
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events
                )  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores


class Gru4RecMCLSRModel(GRUModel, config_name='gru4rec_mclsr'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            user_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_layers,
            num_graph_layers,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            graph_dropout=0.0,
            alpha=0.5,
            initializer_range=0.02,
            is_bidirectional=False
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_bidirectional=is_bidirectional
        )

        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._user_prefix = user_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items

        self._embedding_dim = embedding_dim

        self._num_graph_layers = num_graph_layers
        self._graph_dropout = graph_dropout

        self._alpha = alpha

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )

        # Current interest learning
        self._current_interest_learning_encoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=4 * embedding_dim, out_features=1, bias=False)
        )

        # General interest learning
        self._general_interest_learning_encoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False),
            nn.Tanh()
        )

        # Cross-view contrastive learning
        self._sequential_projector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )
        self._graph_projector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._user_projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._item_projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            user_prefix=config['user_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            num_graph_layers=config['num_graph_layers'],
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02),
            is_bidirectional=config.get('is_bidirectional', False),
        )

    def _apply_graph_encoder(self, embeddings, graph):

        if self.training:  # training_mode
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            dropout_mask = torch.rand(len(values)) + self._graph_dropout
            dropout_mask = dropout_mask.int().bool()
            index = index[~dropout_mask]
            values = values[~dropout_mask] / (1.0 - self._graph_dropout)
            graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
        else:  # eval mode
            graph_dropped = graph

        for _ in range(self._num_graph_layers):
            embeddings = torch.sparse.mm(graph_dropped, embeddings)

        return embeddings

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            events=all_sample_events,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

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

            return {
                'current_embeddings': all_sample_embeddings,
                'positive_embeddings': all_positive_sample_embeddings,
                'negative_embeddings': all_negative_sample_embeddings
            }
        else:  # eval mode
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events
                )  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
