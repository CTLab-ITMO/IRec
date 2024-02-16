from models.base import SequentialTorchModel

from utils import create_masked_tensor

import torch
import torch.nn as nn


class SasRecModel(SequentialTorchModel, config_name='sasrec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
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
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
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
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
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
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                batch_size = candidate_lengths.shape[0]
                num_candidates = candidate_lengths[0]

                candidate_scores = torch.gather(
                    input=candidate_scores,
                    dim=1,
                    index=torch.reshape(candidate_events, [batch_size, num_candidates])
                )  # (batch_size, num_candidates)

            values, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20), (batch_size, 20)

            return indices

# class SasRecMCLSRModel(SequentialTorchModel, config_name='sasrec_mclsr'):
#
#     def __init__(
#             self,
#             sequence_prefix,
#             user_prefix,
#             positive_prefix,
#             negative_prefix,
#             candidate_prefix,
#             common_graph,
#             user_graph,
#             item_graph,
#             num_users,
#             num_items,
#             max_sequence_length,
#             embedding_dim,
#             num_heads,
#             num_layers,
#             num_hops,
#             dim_feedforward,
#             dropout=0.0,
#             activation='relu',
#             layer_norm_eps=1e-5,
#             graph_dropout=0.0,
#             alpha=0.5,
#             initializer_range=0.02
#     ):
#         super().__init__(
#             num_items=num_items,
#             max_sequence_length=max_sequence_length,
#             embedding_dim=embedding_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             layer_norm_eps=layer_norm_eps,
#             is_causal=True
#         )
#         self._sequence_prefix = sequence_prefix
#         self._positive_prefix = positive_prefix
#         self._negative_prefix = negative_prefix
#         self._user_prefix = user_prefix
#         self._candidate_prefix = candidate_prefix
#
#         self._num_users = num_users
#         self._num_items = num_items
#
#         self._embedding_dim = embedding_dim
#
#         self._num_hops = num_hops
#         self._graph_dropout = graph_dropout
#
#         self._alpha = alpha
#
#         self._graph = common_graph
#         self._user_graph = user_graph
#         self._item_graph = item_graph
#
#         self._user_embeddings = nn.Embedding(
#             num_embeddings=num_users + 2,  # add zero embedding + mask embedding
#             embedding_dim=embedding_dim
#         )
#
#         # Current interest learning
#         self._current_interest_learning_encoder = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim, bias=False),
#             nn.Tanh(),
#             nn.Linear(in_features=4 * embedding_dim, out_features=1, bias=False)
#         )
#
#         # General interest learning
#         self._general_interest_learning_encoder = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False),
#             nn.Tanh()
#         )
#
#         # Cross-view contrastive learning
#         self._sequential_projector = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
#             nn.ELU(),
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
#         )
#         self._graph_projector = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
#             nn.ELU(),
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
#         )
#
#         self._user_projection = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
#             nn.ELU(),
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
#         )
#
#         self._item_projection = nn.Sequential(
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
#             nn.ELU(),
#             nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
#         )
#
#         self._init_weights(initializer_range)
#
#     @classmethod
#     def create_from_config(cls, config, **kwargs):
#         return cls(
#             sequence_prefix=config['sequence_prefix'],
#             user_prefix=config['user_prefix'],
#             positive_prefix=config['positive_prefix'],
#             negative_prefix=config['negative_prefix'],
#             candidate_prefix=config['candidate_prefix'],
#             common_graph=kwargs['graph'],
#             user_graph=kwargs['user_graph'],
#             item_graph=kwargs['item_graph'],
#             num_users=kwargs['num_users'],
#             num_items=kwargs['num_items'],
#             max_sequence_length=kwargs['max_sequence_length'],
#             embedding_dim=config['embedding_dim'],
#             num_heads=config['num_heads'],
#             num_layers=config['num_layers'],
#             num_hops=config['num_hops'],
#             dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
#             dropout=config.get('dropout', 0.0),
#             activation=config.get('activation', 'relu'),
#             layer_norm_eps=config.get('layer_norm_eps', 1e-5),
#             graph_dropout=config.get('graph_dropout', 0.0),
#             initializer_range=config.get('initializer_range', 0.02)
#         )
#
#     def _apply_graph_encoder(self, embeddings, graph):
#         if self.training:  # training_mode
#             size = graph.size()
#             index = graph.indices().t()
#             values = graph.values()
#             dropout_mask = torch.rand(len(values)) + self._graph_dropout
#             dropout_mask = dropout_mask.int().bool()
#             index = index[~dropout_mask]
#             values = values[~dropout_mask] / (1.0 - self._graph_dropout)
#             graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
#         else:  # eval mode
#             graph_dropped = graph
#
#         for _ in range(self._num_hops):
#             embeddings = torch.sparse.mm(graph_dropped, embeddings)
#
#         return embeddings
#
#     def forward(self, inputs):
#         all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
#         all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
#         user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
#
#         embeddings, mask = self._apply_sequential_encoder(
#             all_sample_events, all_sample_lengths
#         )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
#         last_embedding = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
#
#         if self.training:  # training mode
#             all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
#             all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)
#
#             all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
#             all_positive_sample_embeddings = self._item_embeddings(
#                 all_positive_sample_events
#             )  # (all_batch_events, embedding_dim)
#             all_negative_sample_embeddings = self._item_embeddings(
#                 all_negative_sample_events
#             )  # (all_batch_events, embedding_dim)
#
#             # General interest learning
#             all_embeddings = torch.cat(
#                 [self._item_embeddings.weight, self._user_embeddings.weight],
#                 dim=0
#             )  # (num_users + 2 + num_items + 2, embedding_dim)
#             common_graph_embeddings = self._apply_graph_encoder(
#                 embeddings=all_embeddings,
#                 graph=self._graph
#             )  # (num_users + 2 + num_items + 2, embedding_dim)
#             common_graph_user_embeddings, common_graph_item_embeddings = torch.split(
#                 common_graph_embeddings,
#                 [self._num_users + 2, self._num_items + 2]
#             )  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)
#
#             common_graph_user_embeddings = \
#                 common_graph_user_embeddings[user_ids]  # (batch_size, embedding_dim)
#             common_graph_item_embeddings = \
#                 common_graph_item_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)
#
#             common_graph_item_embeddings, _ = create_masked_tensor(
#                 data=common_graph_item_embeddings,
#                 lengths=all_sample_lengths
#             )  # (batch_size, seq_len, embedding_dim)
#
#             graph_attention_matrix = torch.einsum(
#                 'bd,bsd->bs',
#                 self._general_interest_learning_encoder(common_graph_user_embeddings),
#                 common_graph_item_embeddings
#             )  # (batch_size, seq_len)
#             graph_attention_matrix[~mask] = -torch.inf
#             graph_attention_matrix = torch.softmax(graph_attention_matrix, dim=1)  # (batch_size, seq_len)
#
#             graph_representation = torch.einsum(
#                 'bs,bsd->bd', graph_attention_matrix, common_graph_item_embeddings
#             )  # (batch_size, embedding_dim)
#
#             # Downstream task
#             combined_representation = \
#                 self._alpha * all_sample_embeddings + \
#                 (1 - self._alpha) * common_graph_item_embeddings[mask]  # (all_batch_events, embedding_dim)
#
#             # Cross-view contrastive learning
#             sequential_representation = self._sequential_projector(last_embedding)  # (batch_size, embedding_dim)
#             graph_representation = self._graph_projector(graph_representation)  # (batch_size, embedding_dim)
#
#             # Feature-level Contrastive Learning
#             user_graph_user_embeddings = self._apply_graph_encoder(
#                 embeddings=self._user_embeddings.weight,
#                 graph=self._user_graph
#             )  # (num_users + 2, embedding_dim)
#             user_graph_user_embeddings = torch.gather(
#                 user_graph_user_embeddings,
#                 dim=0,
#                 index=user_ids[..., None].tile(1, self._embedding_dim)
#             )  # (batch_size, embedding_dim)
#
#             user_graph_user_embeddings = self._user_projection(
#                 user_graph_user_embeddings
#             )  # (batch_size, embedding_dim)
#             common_graph_user_embeddings = self._user_projection(
#                 common_graph_user_embeddings
#             )  # (batch_size, embedding_dim)
#
#             item_graph_item_embeddings = self._apply_graph_encoder(
#                 embeddings=self._item_embeddings.weight,
#                 graph=self._item_graph
#             )  # (num_items + 2, embedding_dim)
#             item_graph_item_embeddings = torch.gather(
#                 item_graph_item_embeddings,
#                 dim=0,
#                 index=all_sample_events[..., None].tile(1, self._embedding_dim)
#             )  # (all_sample_events, embedding_dim)
#
#             item_graph_item_embeddings = self._item_projection(
#                 item_graph_item_embeddings
#             )  # (all_batch_events, embedding_dim)
#             common_graph_item_embeddings = self._item_projection(
#                 common_graph_item_embeddings[mask]
#             )  # (all_batch_events, embedding_dim)
#
#             return {
#                 # Downstream task (sequential)
#                 'current_embeddings': combined_representation,
#                 'positive_embeddings': all_positive_sample_embeddings,
#                 'negative_embeddings': all_negative_sample_embeddings,
#
#                 # Interest-level Contrastive Learning
#                 'sequential_representation': sequential_representation,
#                 'graph_representation': graph_representation,
#
#                 # Feature-level Contrastive Learning (users)
#                 'user_graph_user_embeddings': user_graph_user_embeddings,
#                 'common_graph_user_embeddings': common_graph_user_embeddings,
#
#                 # Feature-level Contrastive Learning (items)
#                 'item_graph_item_embeddings': item_graph_item_embeddings,
#                 'common_graph_item_embeddings': common_graph_item_embeddings
#             }
#         else:  # eval mode
#             if '{}.ids'.format(self._candidate_prefix) in inputs:
#                 candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
#                 candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)
#                 candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)
#                 candidate_embeddings, _ = create_masked_tensor(
#                     data=candidate_embeddings,
#                     lengths=candidate_lengths
#                 )  # (batch_size, num_candidates, embedding_dim)
#                 candidate_scores = torch.einsum(
#                     'bd,bnd->bn',
#                     last_embedding,
#                     candidate_embeddings
#                 )  # (batch_size, num_candidates)
#             else:
#                 candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
#                 candidate_scores = torch.einsum(
#                     'bd,nd->bn',
#                     last_embedding,
#                     candidate_embeddings
#                 )  # (batch_size, num_items)
#                 candidate_scores[:, 0] = -torch.inf
#                 candidate_scores[:, self._num_items + 1:] = -torch.inf
#
#             return candidate_scores
#
#
# class GraphSasRecModel(SequentialTorchModel, config_name='graph_sasrec'):
#
#     def __init__(
#             self,
#             sequence_prefix,
#             user_prefix,
#             positive_prefix,
#             negative_prefix,
#             candidate_prefix,
#             common_graph,
#             user_graph,
#             item_graph,
#             num_users,
#             num_items,
#             max_sequence_length,
#             embedding_dim,
#             num_heads,
#             num_layers,
#             num_hops,
#             dim_feedforward,
#             dropout=0.0,
#             norm_first=True,
#             activation='relu',
#             layer_norm_eps=1e-5,
#             graph_dropout=0.0,
#             initializer_range=0.02
#     ):
#         super().__init__(
#             num_items=num_items,
#             max_sequence_length=max_sequence_length,
#             embedding_dim=embedding_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=activation,
#             layer_norm_eps=layer_norm_eps,
#             is_causal=True
#         )
#         self._sequence_prefix = sequence_prefix
#         self._positive_prefix = positive_prefix
#         self._negative_prefix = negative_prefix
#         self._user_prefix = user_prefix
#         self._candidate_prefix = candidate_prefix
#
#         self._num_users = num_users
#         self._num_items = num_items
#
#         self._embedding_dim = embedding_dim
#
#         self._num_hops = num_hops
#         self._graph_dropout = graph_dropout
#
#         self._graph = common_graph
#         self._user_graph = user_graph
#         self._item_graph = item_graph
#
#         self._mha = MultiheadAttention(
#             embed_dim=embedding_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             bias=True,
#             add_bias_kv=False,
#             add_zero_attn=False,
#             batch_first=True,
#         )
#
#         self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, embedding_dim)
#         self.activation = get_activation_function(activation)
#
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
#         self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self._output_projection = nn.Linear(
#             in_features=2 * embedding_dim,
#             out_features=embedding_dim,
#         )
#
#         self._bias = nn.Parameter(
#             data=torch.zeros(num_items + 2),
#             requires_grad=True
#         )
#
#         self._init_weights(initializer_range)
#
#     @classmethod
#     def create_from_config(cls, config, **kwargs):
#         return cls(
#             sequence_prefix=config['sequence_prefix'],
#             user_prefix=config['user_prefix'],
#             positive_prefix=config['positive_prefix'],
#             negative_prefix=config['negative_prefix'],
#             candidate_prefix=config['candidate_prefix'],
#             common_graph=kwargs['graph'],
#             user_graph=kwargs['user_graph'],
#             item_graph=kwargs['item_graph'],
#             num_users=kwargs['num_users'],
#             num_items=kwargs['num_items'],
#             max_sequence_length=kwargs['max_sequence_length'],
#             embedding_dim=config['embedding_dim'],
#             num_heads=config['num_heads'],
#             num_layers=config['num_layers'],
#             num_hops=config['num_hops'],
#             dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
#             dropout=config.get('dropout', 0.0),
#             activation=config.get('activation', 'relu'),
#             layer_norm_eps=config.get('layer_norm_eps', 1e-5),
#             graph_dropout=config.get('graph_dropout', 0.0),
#             initializer_range=config.get('initializer_range', 0.02)
#         )
#
#     def _ca_block(self, q, k, v, attn_mask=None, key_padding_mask=None):
#         x = self._mha(
#             q, k, v,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask,
#             need_weights=False
#         )[0]  # (batch_size, seq_len, embedding_dim)
#         return self.dropout1(x)  # (batch_size, seq_len, embedding_dim)
#
#     def _ff_block(self, x):
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
#
#     def _apply_graph_encoder(self, embeddings, graph):
#         if self.training:  # training_mode
#             size = graph.size()
#             index = graph.indices().t()
#             values = graph.values()
#             dropout_mask = torch.rand(len(values)) + self._graph_dropout
#             dropout_mask = dropout_mask.int().bool()
#             index = index[~dropout_mask]
#             values = values[~dropout_mask] / (1.0 - self._graph_dropout)
#             graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
#         else:  # eval mode
#             graph_dropped = graph
#
#         for _ in range(self._num_hops):
#             embeddings = torch.sparse.mm(graph_dropped, embeddings)
#
#         return embeddings
#
#     def forward(self, inputs):
#         all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
#         all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
#
#         embeddings, mask = self._apply_sequential_encoder(
#             all_sample_events, all_sample_lengths
#         )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
#
#         common_graph_embeddings = self._apply_graph_encoder(
#             embeddings=self._item_embeddings.weight,
#             graph=self._item_graph
#         )  # (num_items + 2, embedding_dim)
#
#         graph_embeddings = common_graph_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)
#
#         graph_embeddings, graph_mask = create_masked_tensor(
#             data=graph_embeddings,
#             lengths=all_sample_lengths
#         )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
#
#         if self.norm_first:
#             graph_embeddings = graph_embeddings + self.norm1(self._ca_block(
#                 q=embeddings,
#                 k=graph_embeddings,
#                 v=graph_embeddings,
#                 attn_mask=None,
#                 key_padding_mask=~mask
#             ))  # (batch_size, seq_len, embedding_dim)
#             graph_embeddings = graph_embeddings + self.norm2(self._ff_block(graph_embeddings))
#         else:
#             graph_embeddings = self.norm1(graph_embeddings + self._ca_block(
#                 q=embeddings,
#                 k=graph_embeddings,
#                 v=graph_embeddings,
#                 attn_mask=None,
#                 key_padding_mask=~mask
#             ))  # (batch_size, seq_len, embedding_dim)
#             graph_embeddings = self.norm2(graph_embeddings + self._ff_block(graph_embeddings))
#
#         embeddings = torch.cat([embeddings, graph_embeddings], dim=-1)
#         embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)
#
#         last_embedding = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
#
#         if self.training:  # training mode
#             all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
#             all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)
#
#             all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
#             all_positive_sample_embeddings = self._item_embeddings(
#                 all_positive_sample_events
#             )  # (all_batch_events, embedding_dim)
#             all_negative_sample_embeddings = self._item_embeddings(
#                 all_negative_sample_events
#             )  # (all_batch_events, embedding_dim)
#
#             return {
#                 # Downstream task (sequential)
#                 'current_embeddings': all_sample_embeddings,
#                 'positive_embeddings': all_positive_sample_embeddings,
#                 'negative_embeddings': all_negative_sample_embeddings,
#             }
#         else:  # eval mode
#             if '{}.ids'.format(self._candidate_prefix) in inputs:
#                 candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
#                 candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)
#                 candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)
#                 candidate_embeddings, _ = create_masked_tensor(
#                     data=candidate_embeddings,
#                     lengths=candidate_lengths
#                 )  # (batch_size, num_candidates, embedding_dim)
#                 candidate_scores = torch.einsum(
#                     'bd,bnd->bn',
#                     last_embedding,
#                     candidate_embeddings
#                 )  # (batch_size, num_candidates)
#             else:
#                 candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
#                 candidate_scores = torch.einsum(
#                     'bd,nd->bn',
#                     last_embedding,
#                     candidate_embeddings
#                 )  # (batch_size, num_items)
#                 candidate_scores[:, 0] = -torch.inf
#                 candidate_scores[:, self._num_items + 1:] = -torch.inf
#
#             return candidate_scores
