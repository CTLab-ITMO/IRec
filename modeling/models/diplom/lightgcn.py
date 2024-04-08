from models.base import TorchModel

from utils import create_masked_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNModel(TorchModel, config_name='light_gcn'):

    def __init__(
            self,
            user_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            graph,
            num_users,
            num_items,
            embedding_dim,
            num_layers,
            dropout=0.0,
            initializer_range=0.02
    ):
        super().__init__()
        self._user_prefix = user_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix
        self._graph = graph
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._dropout_rate = dropout

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
            graph=kwargs['graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def _apply_graph_encoder(self):
        ego_embeddings = torch.cat((self._user_embeddings.weight, self._item_embeddings.weight), dim=0)
        all_embeddings = [ego_embeddings]

        if self._dropout_rate > 0:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + (1 - self._dropout_rate)
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / (1 - self._dropout_rate)
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for i in range(self._num_layers):
            ego_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=-1)
        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def _get_embeddings(self, inputs, prefix, ego_embeddings, final_embeddings):
        ids = inputs['{}.ids'.format(prefix)]  # (all_batch_events)
        lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

        final_embeddings = final_embeddings[ids]  # (all_batch_events, embedding_dim)
        ego_embeddings = ego_embeddings(ids)  # (all_batch_events, embedding_dim)

        padded_embeddings, mask = create_masked_tensor(
            final_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        padded_ego_embeddings, ego_mask = create_masked_tensor(
            ego_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def forward(self, inputs):
        all_final_user_embeddings, all_final_item_embeddings = \
            self._apply_graph_encoder()  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        user_embeddings, user_ego_embeddings, user_mask = self._get_embeddings(
            inputs, self._user_prefix, self._user_embeddings, all_final_user_embeddings
        )
        user_embeddings = user_embeddings[user_mask]  # (all_batch_events, embedding_dim)

        if self.training:  # training mode
            positive_embeddings, _, positive_mask = self._get_embeddings(
                inputs, self._positive_prefix, self._item_embeddings, all_final_item_embeddings
            )  # (batch_size, seq_len, embedding_dim)
            negative_embeddings, _, negative_mask = self._get_embeddings(
                inputs, self._negative_prefix, self._item_embeddings, all_final_item_embeddings
            )  # (batch_size, seq_len, embedding_dim)

            # b - batch_size, s - seq_len, d - embedding_dim
            positive_scores = torch.einsum(
                'bd,bsd->bs',
                user_embeddings,
                positive_embeddings
            )  # (batch_size, seq_len)
            negative_scores = torch.einsum(
                'bd,bsd->bs',
                user_embeddings,
                negative_embeddings
            )  # (batch_size, seq_len)

            positive_scores = positive_scores[positive_mask]  # (all_batch_events)
            negative_scores = negative_scores[negative_mask]  # (all_batch_events)

            return {
                'positive_scores': positive_scores,
                'negative_scores': negative_scores,
                'item_embeddings': torch.cat((self._user_embeddings.weight, self._item_embeddings.weight), dim=0)
            }
        else:  # eval mode
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                user_embeddings,
                all_final_item_embeddings
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

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
