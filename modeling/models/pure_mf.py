from models.base import TorchModel

import torch
import torch.nn as nn


class PureMF(TorchModel, config_name='pure_mf'):

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._embedding_dim = embedding_dim

        self._user_embeddings = torch.nn.Embedding(
            num_embeddings=self._num_users,
            embedding_dim=self.latent_dim
        )

        self._item_embeddings = torch.nn.Embedding(
            num_embeddings=self._num_items,
            embedding_dim=self.latent_dim
        )

        self.f = nn.Sigmoid()

    def forward(self, inputs):
        users = inputs[self._user_prefix]
        items = inputs[self._item_preifx]

        user_embeddings = self._user_embeddings(users)
        item_embeddings = self._item_embeddings(items)

        scores = torch.sum(user_embeddings * item_embeddings, dim=1)

        return self.f(scores)
