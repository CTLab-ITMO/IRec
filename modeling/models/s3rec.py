from models.base import TorchModel

import torch
import torch.nn as nn

from utils import create_masked_tensor, get_activation_function


class S3RecModel(TorchModel, config_name='s3rec'):

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
            use_item_masking=False,
            use_subsequence_masking=False,
            layer_norm_eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix

        self._num_items = num_items
        self._embedding_dim = embedding_dim

        self._use_item_masking = use_item_masking
        self._use_subsequence_masking = use_subsequence_masking

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

    def _compute_masked_item_prediction(self):
        pass

    def _compute_sequence_prediction(self):
        pass

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        all_sample_embeddings = self._item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)
        embeddings, mask = create_masked_tensor(
            data=all_sample_embeddings,
            lengths=all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = self._encoder(
            src=embeddings,
            src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, num_items)

        if self.training:  # training mode
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            embeddings = embeddings[mask]  # (all_batch_events, num_items)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)
            needed_logits = embeddings[labels_mask]  # (non_zero_events)
            needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            return {'logits': needed_logits, 'labels.ids': needed_labels}
        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

            embeddings[~mask] = 0

            lengths = torch.sum(mask, dim=-1)  # (batch_size)
            lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
            last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

            lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
            lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
            last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

            last_embeddings = last_embeddings[last_masks]  # (batch_size, emb_dim)
            candidate_ids = torch.reshape(candidate_events,
                                          (candidate_lengths.shape[0], -1))  # (batch_size, num_candidates)
            candidate_scores = last_embeddings.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)

            return candidate_scores