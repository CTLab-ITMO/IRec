from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor


class MCLSRModel(TorchModel, config_name='mclsr'):

    def __init__(
            self,
            sequence_prefix,
            user_prefix,
            labels_prefix,
            candidate_prefix,
            num_users,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_graph_layers,
            common_graph,
            user_graph,
            item_graph,
            dropout=0.0,
            layer_norm_eps=1e-5,
            graph_dropout=0.0,
            alpha=0.5,
            initializer_range=0.02
    ):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._user_prefix = user_prefix
        self._labels_prefix = labels_prefix
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

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim
        )

        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

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
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_graph_layers=config['num_graph_layers'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            dropout=config.get('dropout', 0.0),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def _apply_graph_encoder(self, embeddings, graph, use_mean=False):
        assert self.training  # Here we use graph only in training_mode

        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        dropout_mask = torch.rand(len(values)) + self._graph_dropout
        dropout_mask = dropout_mask.int().bool()
        index = index[~dropout_mask]
        values = values[~dropout_mask] / (1.0 - self._graph_dropout)
        graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)

        all_embeddings = [embeddings]
        for _ in range(self._num_graph_layers):
            new_embeddings = torch.sparse.mm(graph_dropped, all_embeddings[-1])
            all_embeddings.append(new_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)

        if use_mean:
            return torch.mean(all_embeddings, dim=1)
        else:
            return all_embeddings[-1]

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
        user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)

        embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)
        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        # Current interest learning
        # 1) get embeddings with positions
        positions = torch.arange(
            start=seq_len - 1, end=-1, step=-1, device=mask.device
        )[None].tile([batch_size, 1]).long()  # (batch_size, seq_len)
        positions_mask = positions < all_sample_lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        positioned_embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)
        positioned_embeddings = self._layernorm(positioned_embeddings)  # (batch_size, seq_len, embedding_dim)
        positioned_embeddings = self._dropout(positioned_embeddings)  # (batch_size, seq_len, embedding_dim)
        positioned_embeddings[~mask] = 0

        sequential_attention_matrix = self._current_interest_learning_encoder(
            positioned_embeddings
        ).squeeze()  # (batch_size, seq_len)
        sequential_attention_matrix[~mask] = -torch.inf
        sequential_attention_matrix = torch.softmax(sequential_attention_matrix, dim=1)  # (batch_size, seq_len)
        sequential_representation = torch.einsum(
            'bs,bsd->bd', sequential_attention_matrix, embeddings
        )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            # General interest learning
            all_embeddings = torch.cat(
                [self._user_embeddings.weight, self._item_embeddings.weight],
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

            common_graph_user_embeddings = common_graph_user_embeddings[user_ids]  # (batch_size, embedding_dim)
            common_graph_item_embeddings = common_graph_item_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)
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

            # Get final representation
            combined_representation = \
                self._alpha * sequential_representation + \
                (1 - self._alpha) * graph_representation  # (batch_size, embedding_dim)

            labels = inputs['{}.ids'.format(self._labels_prefix)]  # (batch_size)
            labels_embeddings = self._item_embeddings(labels)  # (batch_size, embedding_dim)

            # Cross-view contrastive learning
            sequential_representation = self._sequential_projector(
                sequential_representation)  # (batch_size, embedding_dim)
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
                'combined_representation': combined_representation,
                'label_representation': labels_embeddings,

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
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    sequential_representation,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    sequential_representation,
                    candidate_embeddings
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
