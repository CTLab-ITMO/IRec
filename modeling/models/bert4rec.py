from models.base import TorchModel, SequentialTorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor, get_activation_function


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

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim
        embeddings = torch.nn.functional.gelu(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = torch.einsum(
            'bsd,nd->bsn', embeddings, self._item_embeddings.weight
        )  # (batch_size, seq_len, num_items)
        embeddings += self._bias[None, None, :]  # (batch_size, seq_len, num_items)

        if self.training:  # training mode
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            embeddings = embeddings[mask]  # (all_batch_events, num_items)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)

            needed_logits = embeddings[labels_mask]  # (non_zero_events)
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

            return candidate_scores


class Bert4RecMCLSRModel(TorchModel, config_name='bert4rec_mclsr'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            candidate_prefix,
            user_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
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
            num_embeddings=num_items + 2,
            embedding_dim=embedding_dim
        )

        self._positional_embeddings = nn.Embedding(
            num_embeddings=sequence_length,
            embedding_dim=embedding_dim
        )

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

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=num_items + 1
        )

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

    def computer(self):
        all_embeddings = torch.cat([self._user_embeddings.weight, self._item_embeddings.weight])
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

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        all_sample_embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=all_sample_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        # encoder part
        embeddings = self._encoder(
            src=embeddings,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        item_scores = self._output_projection(embeddings)  # (batch_size, seq_len, num_items)

        # Current interest learning part
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

            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            item_scores = item_scores[mask]  # (all_batch_events, num_items)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)
            needed_logits = item_scores[labels_mask]  # (non_zero_events)
            needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            training_output['logits'] = needed_logits
            training_output['labels.ids'] = needed_labels

            # General interest learning part
            user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
            all_user_final_embeddings, all_item_final_embeddings = self.computer()
            user_embeddings = all_user_final_embeddings[user_ids]  # (batch_size, embedding_dim)
            item_embeddings = all_item_final_embeddings[all_sample_events]  # (all_batch_events, embedding_dim)

            # Feature-level contrastive learning part
            # if self._user_graph is not None:
            #     user_graph_final_embeddings = self.compute_user_graph_encoder()[user_ids]  # (batch_size, embedding_dim)
            #     training_output['user_final_embeddings'] = self._user_sequential(user_final_embeddings)  # (batch_size, embedding_dim)
            #     training_output['user_graph_final_embeddings'] = self._user_sequential(user_graph_final_embeddings)  # (batch_size, embedding_dim)

            # if self._item_graph is not None:
            #     item_graph_final_embeddings = self.compute_item_graph_encoder()[sequence_ids]  # (all_batch_items, embedding_dim)
            #     training_output['item_graph_embeddings'] = self._item_sequential(item_final_embeddings)  # (all_batch_items, embedding_dim)
            #     training_output['item_graph_final_embeddings'] = self._item_sequential(item_graph_final_embeddings)  # (all_batch_items, embedding_dim)

            all_item_final_embeddings, _ = create_masked_tensor(
                data=item_embeddings, lengths=all_sample_lengths
            )  # (batch_size, max_seq_len, embedding_dim)

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

            item_scores[~mask] = 0

            lengths = torch.sum(mask, dim=-1)  # (batch_size)
            lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
            last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

            lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
            lengths = torch.tile(lengths, (1, 1, item_scores.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = item_scores.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)
            candidate_ids = torch.reshape(candidate_events,
                                          (candidate_lengths.shape[0], -1))  # (batch_size, num_candidates)
            candidate_scores = last_embeddings.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)

            return candidate_scores
