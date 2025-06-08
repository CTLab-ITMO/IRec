from models.base import TorchModel

from utils import create_masked_tensor, DEVICE

import torch
import torch.nn as nn
import torch.nn.functional as F


class NgcfModel(TorchModel, config_name='ngcf'):

    def __init__(
            self,
            user_prefix,
            positive_prefix,
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
        self._graph = graph
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._dropout_rate = dropout

        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        for i in range(self._num_layers):
            self.dropout_list.append(nn.Dropout(dropout))
            self.GC_Linear_list.append(nn.Linear(embedding_dim, embedding_dim))
            self.Bi_Linear_list.append(nn.Linear(embedding_dim, embedding_dim))

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
            graph=kwargs['graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

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
            side_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=-1)
        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def forward(self, inputs):
        all_final_user_embeddings, all_final_item_embeddings = \
            self._apply_graph_encoder()  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        user_embeddings, user_ego_embeddings, user_mask = self._get_embeddings(
            inputs, self._user_prefix, self._user_embeddings, all_final_user_embeddings
        )
        user_embeddings = user_embeddings[user_mask]  # (all_batch_events, embedding_dim)

        if self.training:  # training mode
            positive_item_ids = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            positive_item_lengths = inputs['{}.length'.format(self._positive_prefix)]  # (batch_size)

            batch_size = positive_item_lengths.shape[0]
            max_sequence_length = positive_item_lengths.max().item()

            mask = torch.arange(
                end=max_sequence_length,
                device=DEVICE
            )[None].tile([batch_size, 1]) < positive_item_lengths[:, None]  # (batch_size, max_seq_len)

            positive_user_ids = torch.arange(
                batch_size,
                device=DEVICE
            )[None].tile([max_sequence_length, 1]).T  # (batch_size, max_seq_len)
            positive_user_ids = positive_user_ids[mask]  # (all_batch_items)
            user_embeddings = user_embeddings[positive_user_ids]  # (all_batch_items, embedding_dim)

            all_scores = torch.einsum(
                'ad,nd->an',
                user_embeddings,
                all_final_item_embeddings
            )  # (all_batch_items, num_items + 2)

            negative_mask = torch.zeros(self._num_items + 2, dtype=torch.bool, device=DEVICE)  # (num_items + 2)
            negative_mask[positive_item_ids] = 1

            positive_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=positive_item_ids[..., None]
            )  # (all_batch_items, 1)

            all_scores = torch.scatter_add(
                input=all_scores,
                dim=1,
                index=positive_item_ids[..., None],
                src=torch.ones_like(positive_item_ids[..., None]).float()
            )  # (all_batch_items, num_items + 2)

            return {
                'positive_scores': positive_scores,
                'negative_scores': all_scores,
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

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
