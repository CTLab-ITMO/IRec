from models.base import SequentialTorchModel

import torch
import torch.nn as nn


class Bert4RecModelPopular(SequentialTorchModel, config_name='bert4rec_popular'):

    def __init__(
            self,
            sequence_prefix,
            labels_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='gelu',
            layer_norm_eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__(
            num_items=num_items,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=False
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )

        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        embeddings = self._output_projection(embeddings)  # (batch_size, seq_len, embedding_dim)

        if self.training: # training mode
            # TODO: move 'negative_item' to config
            negative_items = inputs['negative_item.ids'] # (num_negatives)

            negative_embeddings = self._item_embeddings.weight[negative_items] # (batch_size * num_negatives, embedding_dim)
            negative_embeddings = negative_embeddings.reshape(len(all_sample_lengths), int(negative_embeddings.size(0) / len(all_sample_lengths)), negative_embeddings.size(1)) # (batch_size, num_negatives, embedding_dim)
            negative_embeddings = torch.repeat_interleave(negative_embeddings, all_sample_lengths, dim=0)  # (all_batch_events, num_negatives, embedding_dim)

            embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)
            non_zero_negative_embeddings = negative_embeddings[labels_mask] # (non_zero_events, num_negatives, embedding_dim)
            non_zero_embeddings = embeddings[labels_mask] # (non_zero_events, embedding_dim)
            non_zero_labels = all_sample_labels[labels_mask] # (non_zero_events)

            non_zero_samples_logits = torch.einsum("bd,bnd->bn", non_zero_embeddings, non_zero_negative_embeddings) # (non_zero_events, num_negatives)
            # non_zero_samples_logits = non_zero_embeddings @ non_zero_negative_embeddings.T # (non_zero_events, num_negatives)
            non_zero_labels_embeddings = self._item_embeddings.weight[non_zero_labels] # (non_zero_events, embedding_dim)
            non_zero_labels_logits = (non_zero_embeddings * non_zero_labels_embeddings).sum(dim=-1).unsqueeze(1) # (non_zero_events, 1)

            needed_logits = torch.cat((non_zero_labels_logits, non_zero_samples_logits), dim=1)  # (non_zero_events, num_negatives + 1)

            needed_labels = torch.zeros(len(needed_logits), dtype=torch.long, device=needed_logits.device) # (non_zero_events)

            return {'logits': needed_logits, 'labels.ids': needed_labels}
        else: # eval mode
            embeddings = torch.einsum(
                'bsd,nd->bsn', embeddings, self._item_embeddings.weight
            )  # (batch_size, seq_len, num_items)

            candidate_scores = self._get_last_embedding(embeddings, mask)  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
