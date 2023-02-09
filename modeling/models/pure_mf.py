from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor


class PureMF(TorchModel, config_name='pure_mf'):

    def __init__(
            self,
            user_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            num_users,
            num_items,
            embedding_dim,
            initializer_range
    ):
        super().__init__()

        self._user_prefix = user_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2,
            embedding_dim=self._embedding_dim
        )

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            user_prefix=config['user_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            initializer_range=config.get('initializer_range', 0.02)
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

    def forward(self, inputs):
        user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
        user_embeddings = self._user_embeddings(user_ids)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            all_positive = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_positive_embeddings = self._item_embeddings(all_positive)  # (all_batch_events, embedding_dim)
            positive_lengths = inputs['{}.length'.format(self._positive_prefix)]  # (batch_size)

            all_negative = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)
            all_negative_embeddings = self._item_embeddings(all_negative)  # (all_batch_events, embedding_dim)
            negative_lengths = inputs['{}.length'.format(self._negative_prefix)]  # (batch_size)

            positive_embeddings, positive_mask = create_masked_tensor(all_positive_embeddings, positive_lengths)
            negative_embeddings, negative_mask = create_masked_tensor(all_negative_embeddings, negative_lengths)

            positive_scores = torch.einsum('bd,bsd->bs', user_embeddings, positive_embeddings)  # (batch_size, seq_len)
            negative_scores = torch.einsum('bd,bsd->bs', user_embeddings, negative_embeddings)  # (batch_size, seq_len)

            positive_scores = positive_scores[positive_mask]  # (all_batch_events)
            negative_scores = negative_scores[negative_mask]  # (all_batch_events)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores
            }
        else:
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)
            candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)

            candidate_embeddings, _ = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )  # (batch_size, num_candidates, embedding_dim)

            all_candidates_scores = torch.einsum('bd,bcd->bc', user_embeddings, candidate_embeddings)
            return all_candidates_scores


class PureMFMCLSRModel(TorchModel, config_name='pure_mf_mclsr'):

    def __init__(
            self,
            user_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
            embedding_dim,
            num_graph_layers,
            keep_prob=1.0,
            graph_dropout=0.0,
            alpha=0.5,
            initializer_range=0.02
    ):
        super().__init__()

        self._user_prefix = user_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim

        self._num_graph_layers = num_graph_layers
        self._keep_prob = keep_prob
        self._graph_dropout = graph_dropout

        self._alpha = alpha  # TODO fix

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2,
            embedding_dim=self._embedding_dim
        )

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim
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
            user_prefix=config['user_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_graph_layers=config['num_graph_layers'],
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

    def forward(self, inputs):
        user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)
        user_embeddings = self._user_embeddings(user_ids)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            training_output = {'current_interest_embeddings': user_embeddings}

            all_positive = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_positive_embeddings = self._item_embeddings(all_positive)  # (all_batch_events, embedding_dim)
            positive_lengths = inputs['{}.length'.format(self._positive_prefix)]  # (batch_size)

            all_negative = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)
            all_negative_embeddings = self._item_embeddings(all_negative)  # (all_batch_events, embedding_dim)
            negative_lengths = inputs['{}.length'.format(self._negative_prefix)]  # (batch_size)

            positive_embeddings, positive_mask = create_masked_tensor(all_positive_embeddings, positive_lengths)
            negative_embeddings, negative_mask = create_masked_tensor(all_negative_embeddings, negative_lengths)

            positive_scores = torch.einsum('bd,bsd->bs', user_embeddings, positive_embeddings)  # (batch_size, seq_len)
            negative_scores = torch.einsum('bd,bsd->bs', user_embeddings, negative_embeddings)  # (batch_size, seq_len)

            positive_scores = positive_scores[positive_mask]  # (all_batch_events)
            negative_scores = negative_scores[negative_mask]  # (all_batch_events)

            training_output['positive_scores'] = positive_scores
            training_output['negative_scores'] = negative_scores

            user_final_embeddings, _ = self.computer()
            user_final_embeddings = user_final_embeddings[user_ids]  # (batch_size, embedding_dim)

            training_output['global_interest_embeddings'] = user_final_embeddings

            return training_output
        else:
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)
            candidate_embeddings = self._item_embeddings(candidate_events)  # (all_batch_candidates, embedding_dim)

            candidate_embeddings, _ = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )  # (batch_size, num_candidates, embedding_dim)

            all_candidates_scores = torch.einsum('bd,bcd->bc', user_embeddings, candidate_embeddings)
            return all_candidates_scores
