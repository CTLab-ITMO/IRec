from models.base import TorchModel, SequentialTorchModel

from utils import DEVICE, create_masked_tensor, get_activation_function

import torch
import torch.nn as nn


class SasRecModel(SequentialTorchModel, config_name='sasrec'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            labels_prefix,
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
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
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
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

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events)  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events)  # (all_batch_events, embedding_dim)

            positive_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_positive_sample_embeddings)  # (all_batch_events)
            negative_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_negative_sample_embeddings)  # (all_batch_events)

            return {'positive_scores': positive_scores, 'negative_scores': negative_scores}
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_embeddings, candidate_mask = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )

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

            return candidate_scores


class SasRecMCLSRModel(TorchModel, config_name='sasrec_mclsr'):

    def __init__(
            self,
            sequence_prefix,
            user_prefix,
            positive_prefix,
            negative_prefix,
            labels_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
            max_sequence_length,
            sequence_length,
            embedding_dim,
            num_encoder_heads,
            num_encoder_layers,
            num_graph_layers,
            dim_feedforward,
            encoder_dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            keep_prob=1.0,
            graph_dropout=0.0,
            alpha=0.5,
            initializer_range=0.02
    ):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._user_prefix = user_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items
        self._sequence_length = sequence_length

        self._embedding_dim = embedding_dim
        self._num_encoder_heads = num_encoder_heads
        self._num_encoder_layers = num_encoder_layers
        self._dim_feedforward = dim_feedforward
        self._encoder_dropout = encoder_dropout
        self._activation = activation
        self._layer_norm_eps = layer_norm_eps

        self._num_graph_layers = num_graph_layers
        self._keep_prob = keep_prob
        self._graph_dropout = graph_dropout

        self._alpha = alpha  # TODO fix

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,
            embedding_dim=embedding_dim
        )

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(encoder_dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_encoder_layers)

        self._weights_1 = nn.Parameter(
            data=torch.rand(4 * embedding_dim, embedding_dim), requires_grad=True
        )

        self._weights_2 = nn.Parameter(
            data=torch.rand(4 * embedding_dim), requires_grad=True
        )

        self._weights_3 = nn.Parameter(
            data=torch.rand(embedding_dim, embedding_dim), requires_grad=True
        )

        # Contrastive learning weights
        self._user_sequential = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._item_sequential = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.Sigmoid(),
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
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_encoder_heads=config['num_encoder_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_graph_layers=config['num_graph_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            encoder_dropout=config.get('encoder_dropout', 0.0),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            keep_prob=config.get('keep_prob', 1.0),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def computer(self, use_users=True, use_items=True):
        all_embeddings = []

        if use_users:
            users_embeddings = self._user_embeddings.weight
            all_embeddings.append(users_embeddings)

        if use_items:
            items_embeddings = self._item_embeddings.weight
            all_embeddings.append(items_embeddings)

        all_embeddings = torch.cat(all_embeddings)

        embeddings = [all_embeddings]

        if self._graph_dropout:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + self._keep_prob
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / self._keep_prob
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for layer in range(self._num_graph_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        light_out = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        user_final_embeddings, item_final_embeddings = torch.split(
            light_out,
            [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def compute_user_graph_encoder(self):
        all_embeddings = torch.cat([self._user_embeddings.weight])
        embeddings = [all_embeddings]

        if self._graph_dropout:  # drop some edges
            if self.training:  # training_mode
                size = self._user_graph.size()
                index = self._user_graph.indices().t()
                values = self._user_graph.values()
                random_index = torch.rand(len(values)) + self._keep_prob
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / self._keep_prob
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._user_graph
        else:
            graph_dropped = self._user_graph

        for layer in range(self._num_graph_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        return torch.mean(torch.stack(embeddings, dim=1), dim=1)

    def compute_item_graph_encoder(self):
        all_embeddings = torch.cat([self._item_embeddings.weight])
        embeddings = [all_embeddings]

        if self._graph_dropout:  # drop some edges
            if self.training:  # training_mode
                size = self._item_graph.size()
                index = self._item_graph.indices().t()
                values = self._item_graph.values()
                random_index = torch.rand(len(values)) + self._keep_prob
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / self._keep_prob
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._item_graph
        else:
            graph_dropped = self._item_graph

        for layer in range(self._num_graph_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        return torch.mean(torch.stack(embeddings, dim=1), dim=1)

    def _get_embeddings(self, inputs, prefix, ego_embeddings, final_embeddings):
        ids = inputs['{}.ids'.format(prefix)]  # (batch_size)
        lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

        final_embeddings = final_embeddings[ids]  # (batch_size, emb_dim)
        ego_embeddings = ego_embeddings(ids)  # (batch_size, emb_dim)

        padded_embeddings, mask = create_masked_tensor(final_embeddings, lengths)
        padded_ego_embeddings, ego_mask = create_masked_tensor(ego_embeddings, lengths)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = torch.tile(
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device).unsqueeze(0),
            dims=[batch_size, 1]
        ).long()  # (batch_size, seq_len)
        position_embeddings = self._position_embeddings(positions)  # (batch_size, seq_len, embedding_dim)
        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        causal_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).bool().to(DEVICE)  # (seq_len, seq_len)

        embeddings = self._encoder(
            src=embeddings,
            mask=~causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        # Current interest learning part TODO try other options
        sequence_logits = torch.einsum(
            'n,bsn->bs',
            self._weights_2,
            torch.tanh(torch.einsum('nd,bsd->bsn', self._weights_1, embeddings))
        )  # (batch_size, max_seq_len)

        sequence_logits[~mask] = -torch.inf
        attention_probits = torch.softmax(sequence_logits, dim=1)  # (batch_size, max_seq_len)

        current_interest_embedding = torch.einsum(
            'bs,bsd->bd', attention_probits, embeddings
        )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            training_output = {'current_interest_embeddings': current_interest_embedding}

            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events)  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events)  # (all_batch_events, embedding_dim)

            positive_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_positive_sample_embeddings)  # (all_batch_events)
            negative_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_negative_sample_embeddings)  # (all_batch_events)

            training_output['encoder_positive_scores'] = positive_scores  # (all_batch_events)
            training_output['encoder_negative_scores'] = negative_scores  # (all_batch_events)

            # General interest learning part
            user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
            all_user_final_embeddings, all_item_final_embeddings = self.computer()
            user_embeddings = all_user_final_embeddings[user_ids]  # (batch_size, embedding_dim)
            item_embeddings = all_item_final_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)

            positive_embeddings, _, positive_mask = self._get_embeddings(
                inputs, self._positive_prefix, self._item_embeddings, all_item_final_embeddings
            )

            negative_embeddings, _, negative_mask = self._get_embeddings(
                inputs, self._negative_prefix, self._item_embeddings, all_item_final_embeddings
            )

            positive_scores = torch.einsum('bd,bsd->bs', user_embeddings, positive_embeddings)  # (batch_size, seq_len)
            negative_scores = torch.einsum('bd,bsd->bs', user_embeddings, negative_embeddings)  # (batch_size, seq_len)

            training_output['graph_positive_scores'] = positive_scores[positive_mask]  # (all_batch_events)
            training_output['graph_negative_scores'] = negative_scores[negative_mask]  # (all_batch_events)

            all_item_final_embeddings, _ = create_masked_tensor(
                data=item_embeddings, lengths=all_sample_lengths
            )  # (batch_size, max_seq_len, embedding_dim)

            # TODO try linear layer
            graph_logits = torch.einsum(
                'bd,bsd->bs',
                torch.tanh(torch.einsum('bd,da->ba', user_embeddings, self._weights_3)),
                all_item_final_embeddings
            )  # (batch_size, max_seq_len)

            graph_logits[~mask] = -torch.inf
            attention_graph_probits = torch.softmax(graph_logits, dim=1)  # (batch_size, max_seq_len)

            global_interest_embedding = torch.einsum(
                'bs,bsd->bd', attention_graph_probits, all_item_final_embeddings
            )  # (batch_size, embedding_dim)
            training_output['global_interest_embeddings'] = global_interest_embedding

            # Training part
            combined_embedding = self._alpha * current_interest_embedding + (
                        1 - self._alpha) * global_interest_embedding
            training_output['combined_embedding'] = combined_embedding  # (batch_size, embedding_dim)

            return training_output

        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

            candidate_embeddings = self._item_embeddings(
                candidate_events)  # (batch_size, num_candidates, embedding_dim)

            candidate_embeddings, candidate_mask = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )

            lengths = torch.sum(mask, dim=-1)  # (batch_size)

            lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
            last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

            lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
            lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

            candidate_scores = torch.einsum(
                'bd,bnd->bn',
                last_embeddings,
                candidate_embeddings
            )  # (batch_size, num_candidates)

            return candidate_scores
