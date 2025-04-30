from .base import SequentialTorchModel

import torch
import torch.nn as nn


class DuoRecModel(SequentialTorchModel, config_name='duorec'):
    def __init__(
        self,
        sequence_prefix,
        augmented_sequence_prefix,
        labels_prefix,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.0,
        activation='relu',
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        is_causal=True,
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
            is_causal=is_causal,
        )
        self._sequence_prefix = sequence_prefix
        self._augmented_sequence_prefix = augmented_sequence_prefix
        self._labels_prefix = labels_prefix

        # TODO taken from duorec github
        # self._init_weights(initializer_range)
        self._initializer_range = initializer_range
        self.apply(self._init_weights)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            augmented_sequence_prefix=config['augmented_sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation'],
            layer_norm_eps=config['layer_norm_eps'],
            initializer_range=config['initializer_range'],
        )

    # TODO taken from duorec github
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self._initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, inputs):
        all_sample_events = inputs[
            '{}.ids'.format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            '{}.length'.format(self._sequence_prefix)
        ]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events,
            all_sample_lengths,
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(
            embeddings,
            mask,
        )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            items_logits = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight,
            )  # (batch_size, num_items)
            training_output = {
                'logits': items_logits,
                'sequence_representation': last_embeddings,
            }

            # TODO remove this check
            labels = inputs[
                '{}.ids'.format(self._labels_prefix)
            ]  # (batch_size)
            assert torch.allclose(
                self._item_embeddings(labels),
                self._item_embeddings.weight[labels],
            )

            # Unsupervised Augmentation
            embeddings_, mask_ = self._apply_sequential_encoder(
                all_sample_events,
                all_sample_lengths,
            )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
            last_embeddings_ = self._get_last_embedding(
                embeddings_,
                mask_,
            )  # (batch_size, embedding_dim)
            training_output['similar_sequence_representation'] = (
                last_embeddings_
            )
            assert not torch.allclose(
                last_embeddings,
                last_embeddings_,
            ), 'Embedding must be different because of dropout'

            # Semantic Similarity
            all_sample_augmented_events = inputs[
                '{}.ids'.format(self._augmented_sequence_prefix)
            ]  # (all_batch_events)
            all_sample_augmented_lengths = inputs[
                '{}.length'.format(self._augmented_sequence_prefix)
            ]  # (batch_size)

            augmented_embeddings, augmented_mask = (
                self._apply_sequential_encoder(
                    all_sample_augmented_events,
                    all_sample_augmented_lengths,
                )
            )  # (batch_size, aug_seq_len, embedding_dim), (batch_size, aug_seq_len)
            last_augmented_embeddings = self._get_last_embedding(
                augmented_embeddings,
                augmented_mask,
            )  # (batch_size, embedding_dim)
            training_output['augmented_sequence_representation'] = (
                last_augmented_embeddings
            )

            return training_output
        else:  # eval mode
            candidate_embeddings = (
                self._item_embeddings.weight
            )  # (num_items, embedding_dim)
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                candidate_embeddings,
            )  # (batch_size, num_items)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1 :] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20,
                dim=-1,
                largest=True,
            )  # (batch_size, 20)

            return indices
