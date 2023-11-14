from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor


class LightGCNModel(TorchModel, config_name='light_gcn'):

    def __init__(
            self,
            user_prefix,
            positive_prefix,
            negative_prefix,
            candidates_prefix,
            graph,
            num_users,
            num_items,
            embedding_dim,
            num_layers,
            keep_prob=1.0,
            dropout=0.0,
            initializer_range=0.02
    ):
        super().__init__()
        self._user_prefix = user_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidates_prefix = candidates_prefix

        self._graph = graph

        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._keep_prob = keep_prob
        self._dropout = dropout

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
            candidates_prefix=config['candidates_prefix'],
            graph=kwargs['graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            keep_prob=config.get('keep_prob', 1.0),
            dropout=config.get('dropout', 0.0),
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

    def _apply_graph_encoder(self):
        all_embeddings = torch.cat([self._user_embeddings.weight, self._item_embeddings.weight], dim=0)
        embeddings = [all_embeddings]

        if self._dropout:  # drop some edges
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

        for layer in range(self._num_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        light_out = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        user_final_embeddings, item_final_embeddings = torch.split(
            light_out, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def _get_embeddings(self, inputs, prefix, ego_embeddings, final_embeddings):
        ids = inputs['{}.ids'.format(prefix)]  # (batch_size)
        lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

        final_embeddings = final_embeddings[ids]  # (batch_size, emb_dim)
        ego_embeddings = ego_embeddings(ids)  # (batch_size, emb_dim)

        padded_embeddings, mask = create_masked_tensor(final_embeddings, lengths)
        padded_ego_embeddings, ego_mask = create_masked_tensor(ego_embeddings, lengths)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def forward(self, inputs):
        all_final_user_embeddings, all_final_item_embeddings = \
            self._apply_graph_encoder()  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        user_embeddings, user_ego_embeddings, user_mask = self._get_embeddings(
            inputs, self._user_prefix, self._user_embeddings, all_final_user_embeddings
        )
        user_embeddings = user_embeddings[user_mask]  # (batch_size, embedding_dim)

        if self.training:  # training mode
            positive_embeddings, _, positive_mask = self._get_embeddings(
                inputs, self._positive_prefix, self._item_embeddings, all_final_item_embeddings
            )

            negative_embeddings, _, negative_mask = self._get_embeddings(
                inputs, self._negative_prefix, self._item_embeddings, all_final_item_embeddings
            )

            positive_scores = torch.einsum('bd,bsd->bs', user_embeddings, positive_embeddings)  # (batch_size, seq_len)
            negative_scores = torch.einsum('bd,bsd->bs', user_embeddings, negative_embeddings)  # (batch_size, seq_len)

            positive_scores = positive_scores[positive_mask]  # (all_batch_events)
            negative_scores = negative_scores[negative_mask]  # (all_batch_events)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores,
                'positive_embeddings': positive_embeddings[positive_mask],
                'negative_embeddings': negative_embeddings[negative_mask],
                'user_embeddings': user_embeddings
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
                    user_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    user_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
