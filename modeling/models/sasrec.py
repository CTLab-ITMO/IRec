from models.base import TorchModel
from blocks.encoder import TorchEncoder as Encoder

from utils import DEVICE, create_masked_tensor, get_activation_function

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


class SasRecModel(TorchModel, config_name='sasrec'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            candidate_prefix,
            num_items,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

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

    def forward(self, inputs):
        all_sample_events = inputs[self._sequence_prefix]  # (all_batch_events)
        all_sample_lengths = inputs['{}.lengths'.format(self._sequence_prefix)]  # (batch_size)

        all_sample_embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)
        embeddings, mask = create_masked_tensor(
            data=all_sample_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        causal_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).bool()  # (seq_len, seq_len)

        embeddings = self._encoder(
            src=embeddings,
            mask=~causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        if self.training:  # training mode
            all_positive_sample_events = inputs[self._positive_prefix]  # (all_batch_events)
            all_negative_sample_events = inputs[self._negative_prefix]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(all_positive_sample_events)  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(all_negative_sample_events)  # (all_batch_events, embedding_dim)

            positive_scores = torch.einsum('bd,bd->b', all_sample_embeddings, all_positive_sample_embeddings)  # (all_batch_events)
            negative_scores = torch.einsum('bd,bd->b', all_sample_embeddings, all_negative_sample_embeddings)  # (all_batch_events)

            return {'positive_scores': positive_scores, 'negative_scores': negative_scores}
        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.lengths'.format(self._candidate_prefix)]  # (batch_size)

            candidate_embeddings = self._item_embeddings(candidate_events)  # (batch_size, num_candidates, embedding_dim)

            candidate_embeddings, candidate_mask = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )

            embeddings[~mask] = 0

            lengths = torch.sum(mask, dim=-1)  # (batch_size)
            lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
            last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

            lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
            lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

            candidate_scores = torch.einsum('bd,bnd->bn', last_embeddings, candidate_embeddings)  # (batch_size, num_candidates)

            return candidate_scores
