from models.base import TorchModel as Model

from blocks.projector import BaseProjector, TorchProjector as Projector
from blocks.encoder import BaseEncoder, TorchEncoder as Encoder
from blocks.head import BaseHead, TorchHead as Head

from utils import DEVICE

import torch
import torch.nn as nn


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# class SasRec(Model, config_name='sasrec'):
#
#     def __init__(self, projector, encoder, head):
#         super().__init__()
#         self._projector = projector
#         self._encoder = encoder
#         self._head = head
#
#     @classmethod
#     def create_from_config(cls, config, num_users=None, num_items=None, max_sequence_len=None):
#         projector = BaseProjector.create_from_config(
#             config['projector'],
#             num_users=num_users,
#             num_items=num_items,
#             max_sequence_len=max_sequence_len
#         )
#         encoder = BaseEncoder.create_from_config(config['encoder'])
#         head = BaseHead.create_from_config(config['head'])
#
#         return cls(
#             projector=projector,
#             encoder=encoder,
#             head=head
#         )
#
#     def forward(self, inputs):
#         inputs = self._projector(inputs)
#         inputs = self._encoder(inputs)
#         return self._head(inputs)


# class SasRecProjector(Projector, config_name='sasrec'):
#
#     def __init__(
#             self,
#             sample_prefix,
#             positive_prefix,
#             negative_prefix,
#             candidates_prefix,
#             num_users,
#             num_items,
#             max_sequence_len,
#             embedding_dim,
#             dropout_rate=0.0,
#             eps=1e-5
#     ):
#         super().__init__()
#         self._sample_prefix = sample_prefix
#         self._positive_prefix = positive_prefix
#         self._negative_prefix = negative_prefix
#         self._candidates_prefix = candidates_prefix
#
#         self._max_sequence_len = max_sequence_len
#         self._num_users = num_users
#         self._num_items = num_items
#         self._embedding_dim = embedding_dim
#         self._dropout_rate = dropout_rate
#
#         self._position_embeddings = nn.Embedding(
#             num_embeddings=self._max_sequence_len,
#             embedding_dim=self._embedding_dim
#         )
#         self._item_embeddings = nn.Embedding(
#             num_embeddings=self._num_items + 2,  # all items, zero_embedding, mask_embedding
#             embedding_dim=self._embedding_dim
#         )
#
#         self._dropout = nn.Dropout(p=self._dropout_rate)
#         self._layernorms = nn.LayerNorm(embedding_dim, eps)  # TODO change projector on composite
#
#     @classmethod
#     def create_from_config(cls, config, num_users=None, num_items=None, max_sequence_len=None):
#         return cls(
#             sample_prefix=config['sample_prefix'],
#             positive_prefix=config['positive_prefix'],
#             negative_prefix=config['negative_prefix'],
#             candidates_prefix=config['candidates_prefix'],
#             num_users=num_users,
#             num_items=num_items,
#             max_sequence_len=max_sequence_len,
#             embedding_dim=config['embedding_dim'],
#             dropout_rate=config.get('dropout_rate', 0.0),
#             eps=config.get('eps', 1e-5)
#         )
#
#     def forward(self, inputs):  # TODO re-implement
#         current_embeddings = []
#         current_prefixes = []
#
#         if '{}.ids'.format(self._sample_prefix) in inputs:
#             sample_embeddings = inputs['{}.ids'.format(self._sample_prefix)]  # (all_items)
#             sample_embeddings = self._item_embeddings(sample_embeddings)  # (all_items, emb_dim)
#             sample_embeddings *= self._item_embeddings.embedding_dim ** 0.5
#
#             if '{}.positions'.format(self._sample_prefix) in inputs:  # positional embedding
#                 sample_positions = inputs['{}.positions'.format(self._sample_prefix)]  # (all_batch_items)
#                 sample_positions = self._position_embeddings(sample_positions)  # (all_batch_items, emb_dim)
#                 sample_embeddings += sample_positions  # (all_batch_items, emb_dim)
#
#             sample_embeddings = self._dropout(sample_embeddings)
#
#             current_embeddings.append(sample_embeddings)
#             current_prefixes.append(self._sample_prefix)
#
#         if '{}.ids'.format(self._positive_prefix) in inputs:
#             positive_embeddings = inputs['{}.ids'.format(self._positive_prefix)]  # (all_items)
#             positive_embeddings = self._item_embeddings(positive_embeddings)  # (all_items, emb_dim)
#             current_embeddings.append(positive_embeddings)
#             current_prefixes.append(self._positive_prefix)
#
#         if '{}.ids'.format(self._candidates_prefix) in inputs:
#             candidate_embeddings = inputs['{}.ids'.format(self._candidates_prefix)]  # (all_items)
#             candidate_embeddings = self._item_embeddings(candidate_embeddings)  # (all_items, emb_dim)
#             current_embeddings.append(candidate_embeddings)
#             current_prefixes.append(self._candidates_prefix)
#
#         if '{}.ids'.format(self._negative_prefix) in inputs:
#             negative_embeddings = inputs['{}.ids'.format(self._negative_prefix)]  # (all_items)
#             negative_embeddings = self._item_embeddings(negative_embeddings)  # (all_items, emb_dim)
#             current_embeddings.append(negative_embeddings)
#             current_prefixes.append(self._negative_prefix)
#
#         for embeddings, prefix in zip(current_embeddings, current_prefixes):
#             lengths = inputs['{}.length'.format(prefix)]  # (batch_size)
#             batch_size = lengths.shape[0]
#             max_sequence_length = lengths.max().item()
#
#             padded_embeddings = torch.zeros(
#                 batch_size, max_sequence_length, self._embedding_dim,
#                 dtype=torch.float, device=DEVICE
#             )  # (batch_size, seq_len, emb_dim)
#
#             mask = torch.arange(
#                 end=max_sequence_length,
#                 device=DEVICE
#             )[None].tile([batch_size, 1]) < lengths[:, None]  # (batch_size, seq_len)
#
#             padded_embeddings[mask] = embeddings
#
#             inputs[prefix] = padded_embeddings
#             inputs['{}.mask'.format(prefix)] = mask
#
#         return inputs


class SasRecEncoder(Encoder, config_name='sasrec'):

    def __init__(
            self,
            sample_prefix,
            embedding_dim,
            num_layers,
            num_heads,
            dropout=0.0,
            eps=1e-5
    ):
        super().__init__()
        self._sample_prefix = sample_prefix
        self._num_layers = num_layers

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(embedding_dim, eps=eps)

        for _ in range(num_layers):
            new_attn_layernorm = torch.nn.LayerNorm(embedding_dim, eps=eps)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                embedding_dim, num_heads, dropout
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(embedding_dim, eps=eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(embedding_dim, dropout)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, inputs):
        sample_embeddings = inputs[self._sample_prefix]  # (batch_size, seq_len, emb_dim)
        sample_mask = inputs['{}.mask'.format(self._sample_prefix)]  # (batch_size, seq_len)

        sample_embeddings *= sample_mask.unsqueeze(-1)  # (batch_size, seq_len, emb_dim)
        seq_len = sample_embeddings.shape[1]

        attention_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=DEVICE)
        )  # (seq_len, seq_len)

        for i in range(self._num_layers):
            sample_embeddings = torch.transpose(sample_embeddings, dim0=0, dim1=1)  # (seq_len, batch_size, emb_dim)
            Q = self.attention_layernorms[i](sample_embeddings)  # (seq_len, batch_size, emb_dim)
            mha_outputs, _ = self.attention_layers[i](
                Q, sample_embeddings, sample_embeddings,
                attn_mask=~attention_mask
            )

            sample_embeddings = Q + mha_outputs  # (seq_len, batch_size, emb_dim)
            sample_embeddings = torch.transpose(sample_embeddings, dim0=0, dim1=1)  # (batch_size, seq_len, emb_dim)

            sample_embeddings = self.forward_layernorms[i](sample_embeddings)  # (batch_size, seq_len, emb_dim)
            sample_embeddings = self.forward_layers[i](sample_embeddings)  # (batch_size, seq_len, emb_dim)
            sample_embeddings *= sample_mask.unsqueeze(-1)  # (batch_size, seq_len, emb_dim)

        sample_embeddings = self.last_layernorm(sample_embeddings)  # (batch_size, seq_len, emb_dim)

        inputs[self._sample_prefix] = sample_embeddings
        inputs['{}.mask'.format(self._sample_prefix)] = sample_mask

        return inputs


class SasRecHead(Head, config_name='sasrec'):

    def __init__(
            self,
            prefix,
            labels_prefix,
            candidates_prefix,
            positive_prefix,
            negative_prefix,
            output_prefix=None,
    ):
        super().__init__()
        self._prefix = prefix
        self._labels_prefix = labels_prefix
        self._candidates_prefix = candidates_prefix
        self._output_prefix = output_prefix or prefix

        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            prefix=config['prefix'],
            labels_prefix=config['labels_prefix'],
            candidates_prefix=config['candidates_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            output_prefix=config.get('output_prefix', None)
        )

    def forward(self, inputs):
        if self.training:  # train mode
            inputs = self._train_processing(inputs)
        else:  # eval mode
            inputs = self._eval_processing(inputs)

        return inputs

    def _train_processing(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, seq_len, emb_dim)
        mask = inputs['{}.mask'.format(self._prefix)]  # (batch_size, seq_len)
        embeddings[~mask] = 0

        positive_embeddings = inputs[self._positive_prefix]  # (batch_size, seq_len, emb_dim)
        positive_mask = inputs['{}.mask'.format(self._positive_prefix)]  # (batch_size, seq_len)

        negative_embeddings = inputs[self._negative_prefix]  # (batch_size, seq_len, emb_dim)
        negative_mask = inputs['{}.mask'.format(self._negative_prefix)]  # (batch_size, seq_len)

        pos_logits = torch.einsum('bsd,bsd->bs', embeddings, positive_embeddings)
        positive_labels = torch.ones_like(pos_logits)
        neg_logits = torch.einsum('bsd,bsd->bs', embeddings, negative_embeddings)
        negative_labels = torch.zeros_like(neg_logits)

        all_positive_logits = pos_logits[positive_mask]
        all_negative_logits = neg_logits[negative_mask]

        all_positive_labels = positive_labels[positive_mask]
        all_negative_labels = negative_labels[negative_mask]

        inputs['positive.logits'] = all_positive_logits
        inputs['positive.labels'] = all_positive_labels

        inputs['negative.logits'] = all_negative_logits
        inputs['negative.labels'] = all_negative_labels

        return inputs

    def _eval_processing(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, seq_len, emb_dim)

        lengths = inputs['{}.length'.format(self._prefix)]  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
        # TODO check that everything works here (probably yes)

        candidate_embeddings = inputs[self._candidates_prefix]  # (batch_size, num_candidates, emb_dim)
        candidate_scores = torch.sum(candidate_embeddings * last_embeddings, dim=-1)  # (batch_size, num_candidates)
        inputs[self._output_prefix] = candidate_scores  # (batch_size, num_candidates)

        labels_ids = inputs['{}.ids'.format(self._labels_prefix)]  # (all_candidates)
        labels_ids = torch.reshape(labels_ids, (embeddings.shape[0], -1))  # (batch_size, num_candidates)
        inputs['{}.ids'.format(self._labels_prefix)] = labels_ids

        return inputs
