from utils import MetaParent

from utils import DEVICE, create_masked_tensor, get_activation_function

import torch
import torch.nn as nn


class BaseModel(metaclass=MetaParent):
    pass


class TorchModel(nn.Module, BaseModel):

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

    @staticmethod
    def _get_last_embedding(embeddings, mask):
        lengths = torch.sum(mask, dim=-1)  # (batch_size)
        lengths = (lengths - 1)  # (batch_size)
        last_masks = mask.gather(dim=1, index=lengths[:, None])  # (batch_size, 1)
        lengths = torch.tile(lengths[:, None, None], (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
        last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)
        return last_embeddings


class SequentialTorchModel(TorchModel):

    def __init__(
            self,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            is_causal=True
    ):
        super().__init__()
        self._is_causal = is_causal
        self._num_item = num_items

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

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

    def _apply_sequential_encoder(self, events, lengths):
        embeddings = self._item_embeddings(events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = torch.arange(
            start=seq_len - 1, end=-1, step=-1, device=mask.device
        )[None].tile([batch_size, 1]).long()  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        if self._is_causal:
            causal_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).bool().to(DEVICE)  # (seq_len, seq_len)
            embeddings = self._encoder(
                src=embeddings,
                mask=~causal_mask,
                src_key_padding_mask=~mask
            )  # (batch_size, seq_len, embedding_dim)
        else:
            embeddings = self._encoder(
                src=embeddings,
                src_key_padding_mask=~mask
            )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask

    @staticmethod
    def _add_cls_token(items, lengths, cls_token_id=0):
        num_items = items.shape[0]
        batch_size = lengths.shape[0]
        num_new_items = num_items + batch_size

        new_items = torch.ones(
            num_new_items,
            dtype=items.dtype,
            device=items.device
        ) * cls_token_id  # (num_new_items)

        old_items_mask = torch.zeros_like(new_items).bool()  # (num_new_items)
        old_items_mask = ~old_items_mask.scatter(
            src=torch.ones_like(lengths).bool(),
            dim=0,
            index=torch.cat([torch.LongTensor([0]).to(DEVICE), lengths + 1]).cumsum(dim=0)[:-1]
        )  # (num_new_items)
        new_items[old_items_mask] = items
        new_length = lengths + 1

        return new_items, new_length
