from torch.nn import MultiheadAttention

from models.base import SequentialTorchModel

from utils import create_masked_tensor, get_activation_function

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

        self._cls_token = nn.Parameter(torch.rand(1, 1, embedding_dim))

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

        add_cls_token = True
        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths, add_cls_token
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)
        predictions = embeddings[:, 0, :] # (batch_size, embedding_dim)

        if self.training:  # training mode
            candidates = self._item_embeddings(inputs['{}.ids'.format(self._labels_prefix)])  # (batch_size, embedding_dim)

            return {'predictions': predictions, 'candidates': candidates}
        else:  # eval mode
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                predictions,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            if '{}.ids'.format(self._candidate_prefix) in inputs: # only validation should be here
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                batch_size = candidate_lengths.shape[0]
                num_candidates = candidate_lengths[0]

                candidate_scores = torch.gather(
                    input=candidate_scores,
                    dim=1,
                    index=torch.reshape(candidate_events, [batch_size, num_candidates])
                )  # (batch_size, num_candidates)

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices


class GraphBert4RecModel(SequentialTorchModel, config_name='graph_bert4rec'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_hops,
            graph_dropout,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            norm_first=True,
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
            is_causal=False
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._common_graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph
        self._num_hops = num_hops
        self._graph_dropout = graph_dropout

        self._mha = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
        )

        self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embedding_dim)
        self.activation = get_activation_function(activation)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self._output_projection = nn.Linear(
            in_features=2 * embedding_dim,
            out_features=embedding_dim,
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
            initializer_range=config.get('initializer_range', 0.02),
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_hops=config['num_hops'],
            graph_dropout=config['graph_dropout'],
        )

    def _ca_block(self, q, k, v, attn_mask=None, key_padding_mask=None):
        x = self._mha(
            q, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]  # (batch_size, seq_len, embedding_dim)
        return self.dropout1(x)  # (batch_size, seq_len, embedding_dim)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

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

        for _ in range(self._num_hops):
            embeddings = torch.sparse.mm(graph_dropped, embeddings)

        return embeddings

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            events=all_sample_events,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        common_graph_embeddings = self._apply_graph_encoder(
            embeddings=self._item_embeddings.weight,
            graph=self._item_graph
        )  # (num_items + 2, embedding_dim)

        graph_embeddings = common_graph_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)

        graph_embeddings, graph_mask = create_masked_tensor(
            data=graph_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.norm_first:
            graph_embeddings = graph_embeddings + self._ca_block(
                embeddings,
                graph_embeddings,
                graph_embeddings,
                attn_mask=None,
                key_padding_mask=~mask
            )  # (batch_size, seq_len, embedding_dim)
            graph_embeddings = graph_embeddings + self._ff_block(self.norm2(graph_embeddings))
        else:
            graph_embeddings = self.norm1(graph_embeddings + self._ca_block(
                embeddings,
                graph_embeddings,
                graph_embeddings,
                attn_mask=None,
                key_padding_mask=~mask
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings = self.norm2(graph_embeddings + self._ff_block(graph_embeddings))

        embeddings = torch.cat([embeddings, graph_embeddings], dim=-1)
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
