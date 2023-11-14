from models.base import SequentialTorchModel

from utils import create_masked_tensor

import torch
import torch.nn as nn


class Bert4RecModel(SequentialTorchModel, config_name='bert4rec'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='gelu',
            layer_norm_eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=False
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )

        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.nn.functional.gelu(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.einsum(
            'bsd,nd->bsn', embeddings, self._item_embeddings.weight
        )  # (batch_size, seq_len, num_items)
        embeddings += self._bias[None, None, :]  # (batch_size, seq_len, num_items)

        if self.training:  # training mode
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            embeddings = embeddings[mask]  # (all_batch_events, num_items)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)

            needed_logits = embeddings[labels_mask]  # (non_zero_events, num_items)
            needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            return {'logits': needed_logits, 'labels.ids': needed_labels}
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, num_items)

            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_ids = torch.reshape(
                    candidate_events,
                    (candidate_lengths.shape[0], candidate_lengths[0])
                )  # (batch_size, num_candidates)
                candidate_scores = last_embeddings.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)
            else:
                candidate_scores = last_embeddings  # (batch_size, num_items + 2)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores


class Bert4RecMCLSRModel(SequentialTorchModel, config_name='bert4rec_mclsr'):

    def __init__(
            self,
            sequence_prefix,
            user_prefix,
            labels_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            num_graph_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            graph_dropout=0.0,
            alpha=0.5,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=False
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix
        self._user_prefix = user_prefix

        self._num_users = num_users
        self._num_items = num_items

        self._embedding_dim = embedding_dim

        self._num_graph_layers = num_graph_layers
        self._graph_dropout = graph_dropout

        self._alpha = alpha

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )

        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

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
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_graph_layers=config['num_graph_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
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
        user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.nn.functional.gelu(embeddings)  # (batch_size, seq_len, embedding_dim)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, num_items)
        embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)

        if self.training:  # training mode
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)

            # General interest learning
            all_embeddings = torch.cat(
                [self._item_embeddings.weight, self._user_embeddings.weight],
                dim=0
            )  # (num_users + 2 + num_items + 2, embedding_dim)
            common_graph_embeddings = self._apply_graph_encoder(
                embeddings=all_embeddings,
                graph=self._graph
            )  # (num_users + 2 + num_items + 2, embedding_dim)
            common_graph_user_embeddings, common_graph_item_embeddings = torch.split(
                common_graph_embeddings,
                [self._num_users + 2, self._num_items + 2]
            )  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

            common_graph_user_embeddings = torch.gather(
                common_graph_user_embeddings,
                dim=0,
                index=user_ids[..., None].tile(1, self._embedding_dim)
            )  # (batch_size, embedding_dim)

            common_graph_item_embeddings = torch.gather(
                common_graph_item_embeddings,
                dim=0,
                index=all_sample_events[..., None].tile(1, self._embedding_dim)
            )  # (all_batch_events, embedding_dim)

            common_graph_item_embeddings, _ = create_masked_tensor(
                data=common_graph_item_embeddings,
                lengths=all_sample_lengths
            )  # (batch_size, seq_len, embedding_dim)

            graph_attention_matrix = torch.einsum(
                'bd,bsd->bs',
                self._general_interest_learning_encoder(common_graph_user_embeddings),
                common_graph_item_embeddings
            )  # (batch_size, seq_len)
            graph_attention_matrix[~mask] = -torch.inf
            graph_attention_matrix = torch.softmax(graph_attention_matrix, dim=1)  # (batch_size, seq_len)
            graph_representation = torch.einsum(
                'bs,bsd->bd', graph_attention_matrix, common_graph_item_embeddings
            )  # (batch_size, embedding_dim)

            # Downstream task
            combined_representation = \
                self._alpha * embeddings + \
                (1 - self._alpha) * torch.nn.functional.gelu(
                    self._output_projection(common_graph_item_embeddings[mask])
                )  # (all_batch_events, embedding_dim)

            combined_representation = torch.einsum(
                'ad,nd->an', combined_representation, self._item_embeddings.weight
            )  # (all_batch_events, num_items)

            needed_logits = combined_representation[labels_mask]  # (non_zero_events, num_items)
            needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            # Cross-view contrastive learning
            sequential_representation = self._sequential_projector(last_embeddings)  # (batch_size, embedding_dim)
            graph_representation = self._graph_projector(graph_representation)  # (batch_size, embedding_dim)

            # Feature-level Contrastive Learning
            user_graph_user_embeddings = self._apply_graph_encoder(
                embeddings=self._user_embeddings.weight,
                graph=self._user_graph
            )  # (num_users + 2, embedding_dim)
            user_graph_user_embeddings = torch.gather(
                user_graph_user_embeddings,
                dim=0,
                index=user_ids[..., None].tile(1, self._embedding_dim)
            )  # (batch_size, embedding_dim)

            user_graph_user_embeddings = self._user_projection(
                user_graph_user_embeddings
            )  # (batch_size, embedding_dim)
            common_graph_user_embeddings = self._user_projection(
                common_graph_user_embeddings
            )  # (batch_size, embedding_dim)

            item_graph_item_embeddings = self._apply_graph_encoder(
                embeddings=self._item_embeddings.weight,
                graph=self._item_graph
            )  # (num_items + 2, embedding_dim)
            item_graph_item_embeddings = torch.gather(
                item_graph_item_embeddings,
                dim=0,
                index=all_sample_events[..., None].tile(1, self._embedding_dim)
            )  # (all_sample_events, embedding_dim)

            item_graph_item_embeddings = self._item_projection(
                item_graph_item_embeddings
            )  # (all_batch_events, embedding_dim)
            common_graph_item_embeddings = self._item_projection(
                common_graph_item_embeddings[mask]
            )  # (all_batch_events, embedding_dim)

            return {
                # Downstream task
                'logits': needed_logits,
                'labels.ids': needed_labels,

                # Interest-level Contrastive Learning
                'sequential_representation': sequential_representation,
                'graph_representation': graph_representation,

                # Feature-level Contrastive Learning (users)
                'user_graph_user_embeddings': user_graph_user_embeddings,
                'common_graph_user_embeddings': common_graph_user_embeddings,

                # Feature-level Contrastive Learning (items)
                'item_graph_item_embeddings': item_graph_item_embeddings,
                'common_graph_item_embeddings': common_graph_item_embeddings
            }
        else:  # eval mode
            last_embeddings = torch.einsum(
                'ad,nd->an', last_embeddings, self._item_embeddings.weight
            )  # (all_batch_events, num_items)

            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_ids = torch.reshape(
                    candidate_events,
                    (candidate_lengths.shape[0], candidate_lengths[0])
                )  # (batch_size, num_candidates)
                candidate_scores = last_embeddings.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)
            else:
                candidate_scores = last_embeddings  # (batch_size, num_items + 2)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
