from models.base import SequentialTorchModel

import torch

from utils import create_masked_tensor


class Cl4SRecModel(SequentialTorchModel, config_name='cl4srec'):

    def __init__(
            self,
            sequence_prefix,
            fst_augmented_sequence_prefix,
            snd_augmented_sequence_prefix,
            labels_prefix,
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
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
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._fst_augmented_sequence_prefix = fst_augmented_sequence_prefix
        self._snd_augmented_sequence_prefix = snd_augmented_sequence_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            fst_augmented_sequence_prefix=config['fst_augmented_sequence_prefix'],
            snd_augmented_sequence_prefix=config['snd_augmented_sequence_prefix'],
            labels_prefix=config['labels_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation'],
            layer_norm_eps=config['layer_norm_eps'],
            initializer_range=config['initializer_range']
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            labels = inputs['{}.ids'.format(self._labels_prefix)]  # (batch_size)
            labels_embeddings = self._item_embeddings(labels)  # (batch_size, embedding_dim)

            return {
                'sequence_representation': last_embeddings,
                'labels_representation': labels_embeddings
            }

            # all_fst_aug_sample_events = inputs[
            #     '{}.ids'.format(self._fst_augmented_sequence_prefix)]  # (all_batch_events)
            # all_fst_aug_sample_lengths = inputs['{}.length'.format(self._fst_augmented_sequence_prefix)]  # (batch_size)
            # fst_aug_embeddings, fst_aug_mask = self._apply_sequential_encoder(
            #     all_fst_aug_sample_events, all_fst_aug_sample_lengths
            # )  # (batch_size, fst_aug_seq_len, embedding_dim), (batch_size, fst_aug_seq_len)
            # last_fst_aug_embeddings = self._get_last_embedding(
            #     fst_aug_embeddings, fst_aug_mask
            # )  # (batch_size, embedding_dim)
            # training_output['fst_aug_sequence_representation'] = last_fst_aug_embeddings

            # all_snd_aug_sample_events = inputs[
            #     '{}.ids'.format(self._snd_augmented_sequence_prefix)]  # (all_batch_events)
            # all_snd_aug_sample_lengths = inputs['{}.length'.format(self._snd_augmented_sequence_prefix)]  # (batch_size)
            # snd_aug_embeddings, snd_aug_mask = self._apply_sequential_encoder(
            #     all_snd_aug_sample_events, all_snd_aug_sample_lengths
            # )  # (batch_size, snd_aug_seq_len, embedding_dim), (batch_size, snd_aug_seq_len)
            # last_snd_aug_embeddings = self._get_last_embedding(
            #     snd_aug_embeddings, snd_aug_mask
            # )  # (batch_size, embedding_dim)
            # training_output['snd_aug_sequence_representation'] = last_snd_aug_embeddings
        else:  # eval mode
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events
                )  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_items)

            return candidate_scores
