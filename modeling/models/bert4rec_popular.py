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
        # embeddings = torch.nn.functional.gelu(embeddings)  # (batch_size, seq_len, embedding_dim)
        # embeddings += self._bias[None, None, :]  # (batch_size, seq_len, num_items)

        if self.training:  # training mode
            # TODO: move 'not_masked_item' to config
            all_sample_not_masked = inputs['not_masked_item.ids'] # (all_batch_events)

            random_indices = torch.randperm(all_sample_not_masked.shape[0])[:embeddings.shape[0]] # (batch_size)
            random_in_batch_negative_ids = all_sample_not_masked[random_indices] # (batch_size)
            random_in_batch_negative_embeddings = self._item_embeddings.weight[random_in_batch_negative_ids] # (batch_size)

            embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_sample_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_batch_events)
            labels_mask = (all_sample_labels != 0).bool()  # (all_batch_events)
            non_zero_embeddings = embeddings[labels_mask] # (non_zero_events, embedding_dim)
            non_zero_labels = all_sample_labels[labels_mask] # (non_zero_events)

            # non_zero_samples_logits = torch.einsum(
            #     'bd,nd->bn', non_zero_embeddings, random_in_batch_negative_embeddings
            # )  # (non_zero_events, num_negatives=batch_size)
            non_zero_samples_logits = non_zero_embeddings @ random_in_batch_negative_embeddings.T # (non_zero_events, num_negatives=batch_size)
            non_zero_labels_embeddings = self._item_embeddings.weight[non_zero_labels] # (non_zero_events, embedding_dim)
            non_zero_labels_logits = (non_zero_embeddings @ non_zero_labels_embeddings.T).diagonal().unsqueeze(1) # (non_zero_events, 1)

            needed_logits = torch.cat((non_zero_labels_logits, non_zero_samples_logits), dim=1)  # (non_zero_events, num_negatives + 1=batch_size + 1)
            # needed_labels = all_sample_labels[labels_mask]  # (non_zero_events)

            needed_labels = torch.zeros(len(needed_logits), dtype=torch.long) # (non_zero_events)

            return {'logits': needed_logits, 'labels.ids': needed_labels}
        else:
            # eval mode
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
