from blocks.encoder.base import TorchEncoder

from utils import maybe_to_list, DEVICE

import torch
import torch.nn as nn


class LightGCN(TorchEncoder, config_name='light_gcn'):

    def __init__(
            self,
            user_prefix,
            item_prefix,
            num_users,
            num_items,
            embedding_dim,
            num_layers,
            keep_prob,
            dropout,
            graph
    ):
        super().__init__()

        self._user_prefix = maybe_to_list(user_prefix)
        self._item_prefix = maybe_to_list(item_prefix)
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._keep_prob = keep_prob
        self._dropout = dropout
        self._graph = graph

        self._activation = nn.Sigmoid()

        self.user_embeddings = torch.nn.Embedding(
            num_embeddings=self._num_users,
            embedding_dim=self._embedding_dim
        )
        self.item_embeddings = torch.nn.Embedding(
            num_embeddings=self._num_items,
            embedding_dim=self._embedding_dim
        )

        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            user_prefix=config['user_prefix'],
            item_prefix=config['item_prefix'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            keep_prob=config['keep_prob'],
            dropout=config['dropout'],
            graph=kwargs['graph']
        )

    def computer(self):
        users_embeddings = self.user_embeddings.weight
        items_embeddings = self.item_embeddings.weight
        all_embeddings = torch.cat([users_embeddings, items_embeddings])

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

        embeddings = torch.stack(embeddings, dim=1)
        light_out = torch.mean(embeddings, dim=1)
        user_final_embeddings, item_final_embeddings = torch.split(light_out, [self._num_users, self._num_items])
        return user_final_embeddings, item_final_embeddings

    def tmp_function(self, lengths, data):  # TODO add to utils
        batch_size = lengths.shape[0]
        max_sequence_length = lengths.max().item()

        padded_embeddings = torch.zeros(
            batch_size, max_sequence_length, self._embedding_dim,
            dtype=torch.float, device=DEVICE
        )  # (batch_size, max_seq_len, emb_dim)

        mask = torch.arange(
            end=max_sequence_length,
            device=DEVICE
        )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, max_seq_len)

        padded_embeddings[mask] = data

        return padded_embeddings, mask

    def forward(self, inputs):
        all_final_user_embeddings, all_final_item_embeddings = self.computer()  # (user_num, emb_dim), (items_num, emb_dim)

        for user_prefix in self._user_prefix:
            if '{}.ids'.format(user_prefix) in inputs:
                user_ids = inputs['{}.ids'.format(user_prefix)]  # (batch_size)
                user_lengths = inputs['{}.length'.format(user_prefix)]  # (batch_size)

                user_embeddings = all_final_user_embeddings[user_ids]  # (batch_size, emb_dim)
                user_ego_embeddings = self.user_embeddings(user_ids)  # (batch_size, emb_dim)

                padded_embeddings, mask = self.tmp_function(user_lengths, user_embeddings)
                padded_ego_embeddings, ego_mask = self.tmp_function(user_lengths, user_ego_embeddings)

                inputs[user_prefix] = padded_embeddings  # (batch_size, max_seq_len, emb_dim)
                inputs['{}_ego'.format(user_prefix)] = padded_ego_embeddings  # (batch_size, max_seq_len, emb_dim)

                inputs['{}.mask'.format(user_prefix)] = mask  # (batch_size, max_seq_len)
                inputs['{}_ego.mask'.format(user_prefix)] = ego_mask  # (batch_size, max_seq_len)

        for item_prefix in self._item_prefix:
            if '{}.ids'.format(item_prefix) in inputs:
                item_ids = inputs['{}.ids'.format(item_prefix)].long()  # (batch_size)
                item_lengths = inputs['{}.length'.format(item_prefix)]  # (batch_size)

                item_embeddings = all_final_item_embeddings[item_ids]  # (batch_size, emb_dim)
                item_ego_embeddings = self.item_embeddings(item_ids)  # (batch_size, emb_dim)

                padded_embeddings, mask = self.tmp_function(item_lengths, item_embeddings)
                padded_ego_embeddings, ego_mask = self.tmp_function(item_lengths, item_ego_embeddings)

                inputs[item_prefix] = padded_embeddings  # (batch_size, max_seq_len, emb_dim)
                inputs['{}_ego'.format(item_prefix)] = padded_ego_embeddings  # (batch_size, max_seq_len, emb_dim)

                inputs['{}.mask'.format(item_prefix)] = mask  # (batch_size, max_seq_len)
                inputs['{}_ego.mask'.format(item_prefix)] = ego_mask  # (batch_size, max_seq_len)

        return inputs
