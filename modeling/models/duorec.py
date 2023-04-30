from models.base import SequentialTorchModel

import torch

from utils import create_masked_tensor


class DuoRecModel(SequentialTorchModel, config_name='duorec'):

    def __init__(
            self,
            sequence_prefix,
            augmented_sequence_prefix,
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
            initializer_range=0.02,
            is_causal=True
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
            is_causal=is_causal
        )
        self._sequence_prefix = sequence_prefix
        self._augmented_sequence_prefix = augmented_sequence_prefix
        self._candidate_prefix = candidate_prefix
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
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
            initializer_range=config['initializer_range'],
            augmented_sequence_prefix=config['augmented_sequence_prefix'],
        )

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            training_output = {'sequence_representation': last_embeddings}

            # Recommendation Learning
            # TODO try all items and last items
            logits = torch.einsum(
                'bd,nd->bn', last_embeddings, self._item_embeddings.weight
            )  # (batch_size, num_items)

            training_output['next_item_logits'] = logits

            #  Unsupervised Augmentation
            embeddings_, mask_ = self._apply_sequential_encoder(
                all_sample_events, all_sample_lengths
            )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
            last_embeddings_ = self._get_last_embedding(embeddings_, mask_)  # (batch_size, embedding_dim)
            training_output['another_sequence_representation'] = last_embeddings_
            assert torch.allclose(last_embeddings_, last_embeddings)

            # Semantic Similarity
            all_sample_augmented_events = inputs[
                '{}.ids'.format(self._augmented_sequence_prefix)]  # (all_batch_events)
            all_sample_augmented_lengths = inputs[
                '{}.length'.format(self._augmented_sequence_prefix)]  # (batch_size)

            augmented_embeddings, augmented_mask = self._apply_sequential_encoder(
                all_sample_augmented_events, all_sample_augmented_lengths
            )  # (batch_size, augmented_seq_len, embedding_dim)
            last_augmented_embeddings = self._get_last_embedding(
                augmented_embeddings, augmented_mask
            )  # (batch_size, embedding_dim)

            training_output['augmented_sequence_representation'] = last_augmented_embeddings

            return training_output
        else:  # eval mode
            candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
            candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

            candidate_embeddings = self._item_embeddings(
                candidate_events
            )  # (batch_size, num_candidates, embedding_dim)

            candidate_embeddings, _ = create_masked_tensor(
                data=candidate_embeddings,
                lengths=candidate_lengths
            )

            candidate_scores = torch.einsum(
                'bd,bnd->bn',
                last_embeddings,
                candidate_embeddings
            )  # (batch_size, num_candidates)

            return candidate_scores
