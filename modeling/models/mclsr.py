from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor, DEVICE


class MCLSRModel(TorchModel, config_name='mclsr'):
    INF = 1e9

    def __init__(
            self,
            sequence_prefix,
            user_prefix,
            positive_prefix,
            negative_prefix,
            candidates_prefix,
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
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix  # unused
        self._user_prefix = user_prefix
        self._candidates_prefix = candidates_prefix

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

        self._init_weights(initializer_range)

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

        nn.init.trunc_normal_(
            self._positional_embeddings.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            user_prefix=config['user_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidates_prefix=config['candidates_prefix'],
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

        if self._dropout:  # drop some edges
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

        for layer in range(self._num_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        return torch.mean(torch.stack(embeddings, dim=1), dim=1)

    def compute_item_graph_encoder(self):
        all_embeddings = torch.cat([self._item_embeddings.weight])
        embeddings = [all_embeddings]

        if self._dropout:  # drop some edges
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

        for layer in range(self._num_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        return torch.mean(torch.stack(embeddings, dim=1), dim=1)

    def forward(self, inputs):
        sequence_ids = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        sequence_length = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
        sequence_embeddings = self._item_embeddings(sequence_ids)  # (all_batch_events, embedding_dim)
        sequence_embeddings, sequence_mask = create_masked_tensor(
            data=sequence_embeddings, lengths=sequence_length
        )  # (batch_size, max_seq_len, embedding_dim), (batch_size, max_seq_len)

        # Current interest learning part
        position_embedding = torch.flip(torch.arange(0, torch.max(sequence_length).item(), device=DEVICE), dims=[0])  # (max_seq_len)
        position_embedding = torch.reshape(position_embedding, (1, -1))  # (1, max_seq_len)
        position_embedding = torch.tile(
            position_embedding,
            dims=(sequence_length.shape[0], 1)
        )  # (batch_size, max_seq_len)
        position_embedding = self._positional_embeddings(position_embedding)  # (batch_size, max_seq_len, embedding_dim)

        enriched_sequence_embeddings = sequence_embeddings[sequence_mask] + position_embedding[sequence_mask]
        enriched_sequence_embeddings, _ = create_masked_tensor(
            data=enriched_sequence_embeddings, lengths=sequence_length
        )  # (batch_size, max_seq_len, embedding_dim)

        sequence_logits = torch.einsum(
            'n,bsn->bs',
            self._weights_2,
            torch.tanh(torch.einsum('nd,bsd->bsn', self._weights_1, enriched_sequence_embeddings))
        )  # (batch_size, max_seq_len)

        # TODO ask
        sequence_logits[~sequence_mask] = -MCLSRModel.INF
        attention_probits = torch.softmax(sequence_logits, dim=1)  # (batch_size, max_seq_len)

        current_interest_embedding = torch.einsum(
            'bs,bsd->bd', attention_probits, sequence_embeddings
        )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            training_output = {}

            # General interest learning part
            user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
            user_final_embeddings, item_final_embeddings = self.computer()
            user_final_embeddings = user_final_embeddings[user_ids]  # (batch_size, embedding_dim)
            item_final_embeddings = item_final_embeddings[sequence_ids]  # (all_batch_events, embedding_dim)

            item_final_embeddings, _ = create_masked_tensor(
                data=item_final_embeddings, lengths=sequence_length
            )  # (batch_size, max_seq_len, embedding_dim)

            graph_logits = torch.einsum(
                'bd,bsd->bs',
                torch.tanh(torch.einsum('bd,da->ba', user_final_embeddings, self._weights_3)),
                item_final_embeddings
            )  # (batch_size, max_seq_len)

            graph_logits[~sequence_mask] = -MCLSRModel.INF
            attention_graph_probits = torch.softmax(graph_logits, dim=1)  # (batch_size, max_seq_len)

            global_interest_embedding = torch.einsum(
                'bs,bsd->bd', attention_graph_probits, item_final_embeddings
            )  # (batch_size, embedding_dim)

            # Feature-level contrastive learning part
            if self._user_graph is not None:
                user_graph_final_embeddings = self.compute_user_graph_encoder()[user_ids]  # (batch_size, embedding_size)
                training_output['user_final_embeddings'] = self._user_sequential(user_final_embeddings)
                training_output['user_graph_final_embeddings'] = self._user_sequential(user_graph_final_embeddings)

            if self._item_graph is not None:
                item_graph_final_embeddings = self.compute_item_graph_encoder()  # (all_batch_items, embedding_size)
                training_output['item_graph_embeddings'] = self._item_sequential(item_final_embeddings)  # (all_batch_items, embedding_size)
                training_output['item_graph_final_embeddings'] = self._item_sequential(item_graph_final_embeddings)  # (all_batch_items, embedding_size)

            # Training part
            combined_embedding = self._alpha * current_interest_embedding + (1 - self._alpha) * global_interest_embedding
            training_output['combined_embedding'] = combined_embedding  # (batch_size, embedding_dim)

            # (in-batch setting) TODO make random
            positive_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_candidates)
            positive_lengths = inputs['{}.length'.format(self._positive_prefix)]  # (batch_size)
            next_events = positive_events[torch.cumsum(positive_lengths, dim=0).long() - 1]  # (batch_size)
            next_embeddings = self._item_embeddings(next_events)  # (batch_size, embedding_dim)
            training_output['next_item_embedding'] = next_embeddings  # (batch_size, embedding_dim)

            return training_output

        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidates_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidates_prefix)]  # (batch_size)
            candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)

            candidate_embeddings, _ = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )  # (batch_size, num_candidates, embedding_dim)

            candidate_scores = torch.einsum(
                'bd,bnd->bn',
                current_interest_embedding,
                candidate_embeddings
            )  # (batch_size, num_candidates)

            return candidate_scores
