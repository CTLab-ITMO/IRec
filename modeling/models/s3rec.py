from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor, get_activation_function, DEVICE


class S3RecModel(TorchModel, config_name='s3rec'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            candidate_prefix,
            positive_prefix,
            negative_prefix,
            sequence_segment_prefix,
            positive_segment_prefix,
            negative_segment_prefix,
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
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._sequence_segment_prefix = sequence_segment_prefix
        self._positive_segment_prefix = positive_segment_prefix
        self._negative_segment_prefix = negative_segment_prefix

        self._num_items = num_items
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,  # add zero embedding + mask embedding
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

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=num_items + 1
        )

        self._init_weights(initializer_range)

        # add unique dense layer for 2 losses respectively
        self.mip_norm = nn.Linear(embedding_dim, embedding_dim)
        self.sp_norm = nn.Linear(embedding_dim, embedding_dim)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            sequence_segment_prefix=config['sequence_segment_prefix'],
            positive_segment_prefix=config['positive_segment_prefix'],
            negative_segment_prefix=config['negative_segment_prefix'],
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

    def masked_item_prediction(self, sequence_embeddings, sequence_mask, target_item):
        all_items = sequence_embeddings[sequence_mask]  # (all_batch_items, emb_dim)
        score = torch.mul(self.mip_norm(all_items), target_item)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    def segment_prediction(self, context, segment):
        context = self.sp_norm(context)
        score = torch.mul(context, segment)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def pretrain(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        all_sample_embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=all_sample_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        causal_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).bool().to(DEVICE)  # (seq_len, seq_len)

        embeddings = self._encoder(
            src=embeddings,
            mask=~causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        # Masked Item Prediction
        positive_item_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
        negative_item_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

        positive_item_embeddings = self._item_embeddings(positive_item_events)  # (all_batch_events, embedding_dim)
        negative_item_embeddings = self._item_embeddings(negative_item_events)  # (all_batch_events, embedding_dim)

        positive_scores = self.masked_item_prediction(embeddings, mask, positive_item_embeddings)  # (all_batch_events)
        negative_scores = self.masked_item_prediction(embeddings, mask, negative_item_embeddings)  # (all_batch_events)

        mip_mask = (positive_item_events != 0).bool()  # (all_masked_batch_events)
        needed_positive_scores = positive_scores[mip_mask]  # (all_masked_batch_events)
        needed_negative_scores = negative_scores[mip_mask]  # (all_masked_batch_events)

        mip_distance = torch.sigmoid(needed_positive_scores - needed_negative_scores)

        # SP
        all_segment_events = inputs['{}.ids'.format(self._sequence_segment_prefix)]  # (all_batch_events)
        all_segment_lengths = inputs['{}.length'.format(self._sequence_segment_prefix)]  # (batch_size)
        all_segment_embeddings = self._item_embeddings(all_segment_events)  # (all_batch_events, embedding_dim)
        segment_embeddings, segment_mask = create_masked_tensor(
            data=all_segment_embeddings,
            lengths=all_segment_lengths
        )  # (batch_size, seq_len, embedding_dim)
        segment_causal_mask = torch.tril(
            torch.ones(segment_mask.shape[-1], segment_mask.shape[-1])
        ).bool().to(DEVICE)  # (seq_len, seq_len)
        segment_embeddings = self._encoder(
            src=segment_embeddings,
            mask=~segment_causal_mask,
            src_key_padding_mask=~segment_mask
        )  # (batch_size, seq_len, embedding_dim)
        segment_embeddings[~segment_mask] = 0
        lengths = torch.sum(segment_mask, dim=-1)  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
        last_masks = segment_mask.gather(dim=1, index=lengths)  # (batch_size, 1)
        lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, segment_embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = segment_embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
        last_segment_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

        positive_segment_events = inputs['{}.ids'.format(self._positive_segment_prefix)]  # (all_batch_events)
        positive_segment_lengths = inputs['{}.length'.format(self._positive_segment_prefix)]  # (batch_size)
        positive_segment_embeddings = self._item_embeddings(
            positive_segment_events)  # (all_batch_events, embedding_dim)
        positive_segment_embeddings, positive_segment_mask = create_masked_tensor(
            data=positive_segment_embeddings,
            lengths=positive_segment_lengths
        )  # (batch_size, seq_len, embedding_dim)
        positive_segment_causal_mask = torch.tril(
            torch.ones(positive_segment_mask.shape[-1], positive_segment_mask.shape[-1])
        ).bool().to(DEVICE)  # (seq_len, seq_len)
        positive_segment_embeddings = self._encoder(
            src=positive_segment_embeddings,
            mask=~positive_segment_causal_mask,
            src_key_padding_mask=~positive_segment_mask
        )  # (batch_size, seq_len, embedding_dim)
        positive_segment_embeddings[~positive_segment_mask] = 0
        lengths = torch.sum(positive_segment_mask, dim=-1)  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
        last_masks = positive_segment_mask.gather(dim=1, index=lengths)  # (batch_size, 1)
        lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, positive_segment_embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = positive_segment_embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
        last_positive_segment_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

        negative_segment_events = inputs['{}.ids'.format(self._negative_segment_prefix)]  # (all_batch_events)
        negative_segment_lengths = inputs['{}.length'.format(self._negative_segment_prefix)]  # (batch_size)
        negative_segment_embeddings = self._item_embeddings(
            negative_segment_events)  # (all_batch_events, embedding_dim)
        negative_segment_embeddings, negative_segment_mask = create_masked_tensor(
            data=negative_segment_embeddings,
            lengths=negative_segment_lengths
        )  # (batch_size, seq_len, embedding_dim)
        negative_segment_causal_mask = torch.tril(
            torch.ones(negative_segment_mask.shape[-1], negative_segment_mask.shape[-1])
        ).bool().to(DEVICE)  # (seq_len, seq_len)
        negative_segment_embeddings = self._encoder(
            src=negative_segment_embeddings,
            mask=~negative_segment_causal_mask,
            src_key_padding_mask=~negative_segment_mask
        )  # (batch_size, seq_len, embedding_dim)
        negative_segment_embeddings[~negative_segment_mask] = 0
        lengths = torch.sum(negative_segment_mask, dim=-1)  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
        last_masks = negative_segment_mask.gather(dim=1, index=lengths)  # (batch_size, 1)
        lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, negative_segment_embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = negative_segment_embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)
        last_negative_segment_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

        positive_segment_score = self.segment_prediction(last_segment_embeddings,
                                                         last_positive_segment_embeddings)  # (batch_size)
        negative_segment_score = self.segment_prediction(last_segment_embeddings,
                                                         last_negative_segment_embeddings)  # (batch_size)
        sp_distance = torch.sigmoid(positive_segment_score - negative_segment_score)  # (batch_size)

        return {
            'mip_distance': mip_distance,
            'mip_labels': torch.ones_like(mip_distance, dtype=torch.float32),
            'sp_distance': sp_distance,
            'sp_labels': torch.ones_like(sp_distance, dtype=torch.float32)
        }

    # same as SASRec (https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/2a81540ae18615d88ef88227b0c066e5b74781e5/models.py)
    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        all_sample_embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=all_sample_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        causal_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).bool().to(DEVICE)  # (seq_len, seq_len)

        embeddings = self._encoder(
            src=embeddings,
            mask=~causal_mask,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events)  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events)  # (all_batch_events, embedding_dim)

            positive_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_positive_sample_embeddings)  # (all_batch_events)
            negative_scores = torch.einsum('bd,bd->b', all_sample_embeddings,
                                           all_negative_sample_embeddings)  # (all_batch_events)

            return {'positive_scores': positive_scores, 'negative_scores': negative_scores}
        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

            candidate_embeddings = self._item_embeddings(
                candidate_events)  # (batch_size, num_candidates, embedding_dim)

            candidate_embeddings, candidate_mask = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )

            lengths = torch.sum(mask, dim=-1)  # (batch_size)

            lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
            last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

            lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
            lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)

            candidate_scores = torch.einsum('bd,bnd->bn', last_embeddings,
                                            candidate_embeddings)  # (batch_size, num_candidates)

            return candidate_scores
