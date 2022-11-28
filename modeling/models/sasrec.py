from blocks.encoder import TorchEncoder as Encoder

from utils import DEVICE

import torch


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
