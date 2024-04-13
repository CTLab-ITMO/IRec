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
            num_users,
            num_items,
            embedding_dim,
            initializer_range
    ):
        super().__init__()

        self._user_prefix = user_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

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
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            initializer_range=config.get('initializer_range', 0.02)
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
            candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                user_embeddings,
                candidate_embeddings
            )  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
