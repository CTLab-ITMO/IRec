from blocks.projector import TorchProjector as Projector

from utils import DEVICE

import torch
import torch.nn as nn


class DeepFMProjector(Projector, config_name='deepfm'):

    def __init__(self, prefix, num_items, max_sequence_len, embedding_dim, hidden_size, dropout_1, dropout_2, output_prefix=None):
        super().__init__()

        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        self._embedding_dim = embedding_dim
        self._num_items = num_items

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim
        )
        self._item_weights = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=1
        )

        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_len,
            embedding_dim=self._embedding_dim
        )
        self._position_weight = nn.Embedding(
            num_embeddings=max_sequence_len,
            embedding_dim=1
        )

        self._bias_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=1
        )

        self._hidden_size = hidden_size

        self.dropout_fm_1o = nn.Dropout(p=dropout_1)
        self.dropout_fm_2o = nn.Dropout(p=dropout_2)

    def forward(self, inputs):
        idxs = inputs['{}.positions'.format(self._prefix)]  # (all_batch_items)
        vals = self._item_weights(inputs['{}.ids'.format(self._prefix)])  # (all_batch_items, 1)
        lenghts = inputs['{}.length'.format(self._prefix)]

        batch_size = lenghts.shape[0]
        max_sequence_length = lenghts.max().item()

        mask = torch.arange(
            end=max_sequence_length,
            device=DEVICE
        )[None].tile([batch_size, 1]) < lenghts[:, None]  # (batch_size, max_seq_len)

        feat_emb = self._position_embeddings(idxs)  # (all_batch_items, pos_embedding_dim)
        feat_emb = feat_emb * vals  # (all_batch_items, pos_embedding_dim)
        padded_embeddings = torch.zeros(
            batch_size, max_sequence_length, self._embedding_dim,
            dtype=torch.float, device=DEVICE
        )  # (batch_size, max_seq_len, emb_dim)

        padded_embeddings[mask] = feat_emb
        padded_embeddings = padded_embeddings[:, ::-1]  # (batch_size, max_seq_len, emb_dim)

        # first order part
        y_first_order = self._position_weights(idxs)  # (all_batch_items, 1)
        y_first_order = y_first_order * vals  # (all_batch_items, 1)

        tmp = torch.zeros(
            batch_size, max_sequence_length,
            dtype=torch.float, device=DEVICE
        )  # (batch_size, max_seq_len)
        tmp[mask] = y_first_order  # (batch_size, max_seq_len)
        y_first_order = tmp[:, ::-1]  # (batch_size, max_seq_len)
        y_first_order = self.dropout_fm_1o(y_first_order)  # (batch_size, max_seq_len)

        # second order part
        summed_features_emb_square = torch.square(torch.sum(padded_embeddings, dim=1))  # (batch_size, embedding_size)
        squared_sum_features_emb = torch.sum(torch.square(padded_embeddings), dim=1)  # (batch_size, embedding_size)
        y_second_order = 0.5 * (summed_features_emb_square - squared_sum_features_emb)  # (batch_size, embedding_size)
        y_second_order = self.dropout_fm_2o(y_second_order)  # (batch_size, embedding_size)

        # deep part
        y_deep = padded_embeddings.view(batch_size, -1)  # (batch_size, max_seq_len * embedding_size)
        y_deep = self.deep(y_deep)  # (batch_size, output_dim)

        concat_input = torch.cat([y_first_order, y_second_order, y_deep],
                                 dim=1)  # (batch_size, max_seq_len + embedding_size + output_dim)
        output = self.output(concat_input)  # (batch_size)
        out = torch.sigmoid(output)  # (batch_size)

        inputs[self._output_prefix] = out
        return inputs