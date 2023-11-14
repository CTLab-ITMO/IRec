from models.base import SequentialTorchModel

from utils import create_masked_tensor

import torch
import torch.nn as nn


class Cl4SRecModel(SequentialTorchModel, config_name='cl4srec'):

    def __init__(
            self,
            sequence_prefix,
            fst_augmented_sequence_prefix,
            snd_augmented_sequence_prefix,
            positive_prefix,
            negative_prefix,
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
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._labels_prefix = labels_prefix
        self._candidate_prefix = candidate_prefix
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            fst_augmented_sequence_prefix=config['fst_augmented_sequence_prefix'],
            snd_augmented_sequence_prefix=config['snd_augmented_sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
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
            items_logits = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items)

            # TODO remove this check
            labels = inputs['{}.ids'.format(self._labels_prefix)]  # (batch_size)
            assert torch.allclose(
                self._item_embeddings(labels),
                self._item_embeddings.weight[labels]
            )

            all_fst_aug_sample_events = inputs[
                '{}.ids'.format(self._fst_augmented_sequence_prefix)
            ]  # (all_batch_events)
            all_fst_aug_sample_lengths = inputs['{}.length'.format(self._fst_augmented_sequence_prefix)]  # (batch_size)
            fst_aug_embeddings, fst_aug_mask = self._apply_sequential_encoder(
                all_fst_aug_sample_events, all_fst_aug_sample_lengths
            )  # (batch_size, fst_aug_seq_len, embedding_dim), (batch_size, fst_aug_seq_len)
            last_fst_aug_embeddings = self._get_last_embedding(
                fst_aug_embeddings, fst_aug_mask
            )  # (batch_size, embedding_dim)

            all_snd_aug_sample_events = inputs[
                '{}.ids'.format(self._snd_augmented_sequence_prefix)
            ]  # (all_batch_events)
            all_snd_aug_sample_lengths = inputs['{}.length'.format(self._snd_augmented_sequence_prefix)]  # (batch_size)
            snd_aug_embeddings, snd_aug_mask = self._apply_sequential_encoder(
                all_snd_aug_sample_events, all_snd_aug_sample_lengths
            )  # (batch_size, snd_aug_seq_len, embedding_dim), (batch_size, snd_aug_seq_len)
            last_snd_aug_embeddings = self._get_last_embedding(
                snd_aug_embeddings, snd_aug_mask
            )  # (batch_size, embedding_dim)

            return {
                'logits': items_logits,
                'sequence_representation': last_embeddings,
                'fst_aug_sequence_representation': last_fst_aug_embeddings,
                'snd_aug_sequence_representation': last_snd_aug_embeddings
            }
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
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores


class Cl4SRecMCLSRModel(SequentialTorchModel, config_name='cl4srec_mclsr'):

    def __init__(
            self,
            sequence_prefix,
            fst_augmented_sequence_prefix,
            snd_augmented_sequence_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            common_graph,
            user_graph,
            item_graph,
            num_users,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            num_graph_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            graph_dropout=0.0,
            alpha=0.5,
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
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items

        self._embedding_dim = embedding_dim

        self._num_graph_layers = num_graph_layers
        self._graph_dropout = graph_dropout

        self._alpha = alpha

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim
        )

        # Current interest learning
        self._current_interest_learning_encoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=4 * embedding_dim, out_features=1, bias=False)
        )

        # General interest learning
        self._general_interest_learning_encoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False),
            nn.Tanh()
        )

        # Cross-view contrastive learning
        self._sequential_projector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )
        self._graph_projector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._user_projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._item_projection = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True),
            nn.ELU(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            fst_augmented_sequence_prefix=config['fst_augmented_sequence_prefix'],
            snd_augmented_sequence_prefix=config['snd_augmented_sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_graph_layers=config['num_graph_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'relu'),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def _apply_graph_encoder(self, embeddings, graph):

        if self.training:  # training_mode
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            dropout_mask = torch.rand(len(values)) + self._graph_dropout
            dropout_mask = dropout_mask.int().bool()
            index = index[~dropout_mask]
            values = values[~dropout_mask] / (1.0 - self._graph_dropout)
            graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
        else:  # eval mode
            graph_dropped = graph

        for _ in range(self._num_graph_layers):
            embeddings = torch.sparse.mm(graph_dropped, embeddings)

        return embeddings

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
        user_ids = inputs['{}.ids'.format(self._user_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (batch_size)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)
            all_negative_sample_length = inputs['{}.length'.format(self._negative_prefix)]  # (batch_size)

            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events
            )  # (batch_size, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events
            )  # (all_batch_events, embedding_dim)

            negative_embedding, negative_mask = create_masked_tensor(
                data=all_negative_sample_embeddings,
                lengths=all_negative_sample_length
            )  # (batch_size, num_negatives, embedding_dim)

            all_representations = torch.cat([
                all_positive_sample_embeddings.unsqueeze(1),  # (batch_size, 1, embedding_dim)
                negative_embedding  # (batch_size, num_negatives, embedding_dim)
            ], dim=1)  # (batch_size, num_negatives + 1, embedding_dim)

            all_fst_aug_sample_events = inputs[
                '{}.ids'.format(self._fst_augmented_sequence_prefix)]  # (all_batch_events)
            all_fst_aug_sample_lengths = inputs['{}.length'.format(self._fst_augmented_sequence_prefix)]  # (batch_size)
            fst_aug_embeddings, fst_aug_mask = self._apply_sequential_encoder(
                all_fst_aug_sample_events, all_fst_aug_sample_lengths
            )  # (batch_size, fst_aug_seq_len, embedding_dim), (batch_size, fst_aug_seq_len)
            last_fst_aug_embeddings = self._get_last_embedding(
                fst_aug_embeddings, fst_aug_mask
            )  # (batch_size, embedding_dim)

            all_snd_aug_sample_events = inputs[
                '{}.ids'.format(self._snd_augmented_sequence_prefix)]  # (all_batch_events)
            all_snd_aug_sample_lengths = inputs['{}.length'.format(self._snd_augmented_sequence_prefix)]  # (batch_size)
            snd_aug_embeddings, snd_aug_mask = self._apply_sequential_encoder(
                all_snd_aug_sample_events, all_snd_aug_sample_lengths
            )  # (batch_size, snd_aug_seq_len, embedding_dim), (batch_size, snd_aug_seq_len)
            last_snd_aug_embeddings = self._get_last_embedding(
                snd_aug_embeddings, snd_aug_mask
            )  # (batch_size, embedding_dim)

            # General interest learning
            all_embeddings = torch.cat(
                [self._item_embeddings.weight, self._user_embeddings.weight],
                dim=0
            )  # (num_users + 2 + num_items + 2, embedding_dim)
            common_graph_embeddings = self._apply_graph_encoder(
                embeddings=all_embeddings,
                graph=self._graph
            )  # (num_users + 2 + num_items + 2, embedding_dim)
            common_graph_user_embeddings, common_graph_item_embeddings = torch.split(
                common_graph_embeddings,
                [self._num_users + 2, self._num_items + 2]
            )  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

            common_graph_user_embeddings = torch.gather(
                common_graph_user_embeddings,
                dim=0,
                index=user_ids[..., None].tile(1, self._embedding_dim)
            )  # (batch_size, embedding_dim)

            common_graph_item_embeddings = torch.gather(
                common_graph_item_embeddings,
                dim=0,
                index=all_sample_events[..., None].tile(1, self._embedding_dim)
            )  # (all_batch_events, embedding_dim)

            common_graph_item_embeddings, _ = create_masked_tensor(
                data=common_graph_item_embeddings,
                lengths=all_sample_lengths
            )  # (batch_size, seq_len, embedding_dim)

            graph_attention_matrix = torch.einsum(
                'bd,bsd->bs',
                self._general_interest_learning_encoder(common_graph_user_embeddings),
                common_graph_item_embeddings
            )  # (batch_size, seq_len)
            graph_attention_matrix[~mask] = -torch.inf
            graph_attention_matrix = torch.softmax(graph_attention_matrix, dim=1)  # (batch_size, seq_len)
            graph_representation = torch.einsum(
                'bs,bsd->bd', graph_attention_matrix, common_graph_item_embeddings
            )  # (batch_size, embedding_dim)

            # Downstream task
            combined_representation = \
                self._alpha * last_embeddings + \
                (1 - self._alpha) * graph_representation  # (all_batch_events, embedding_dim)

            # Cross-view contrastive learning
            sequential_representation = self._sequential_projector(last_embeddings)  # (batch_size, embedding_dim)
            graph_representation = self._graph_projector(graph_representation)  # (batch_size, embedding_dim)

            # Feature-level Contrastive Learning
            user_graph_user_embeddings = self._apply_graph_encoder(
                embeddings=self._user_embeddings.weight,
                graph=self._user_graph
            )  # (num_users + 2, embedding_dim)
            user_graph_user_embeddings = torch.gather(
                user_graph_user_embeddings,
                dim=0,
                index=user_ids[..., None].tile(1, self._embedding_dim)
            )  # (batch_size, embedding_dim)

            user_graph_user_embeddings = self._user_projection(
                user_graph_user_embeddings
            )  # (batch_size, embedding_dim)
            common_graph_user_embeddings = self._user_projection(
                common_graph_user_embeddings
            )  # (batch_size, embedding_dim)

            item_graph_item_embeddings = self._apply_graph_encoder(
                embeddings=self._item_embeddings.weight,
                graph=self._item_graph
            )  # (num_items + 2, embedding_dim)
            item_graph_item_embeddings = torch.gather(
                item_graph_item_embeddings,
                dim=0,
                index=all_sample_events[..., None].tile(1, self._embedding_dim)
            )  # (all_sample_events, embedding_dim)

            item_graph_item_embeddings = self._item_projection(
                item_graph_item_embeddings
            )  # (all_batch_events, embedding_dim)
            common_graph_item_embeddings = self._item_projection(
                common_graph_item_embeddings[mask]
            )  # (all_batch_events, embedding_dim)

            return {
                # Downstream task
                'sequence_representation': combined_representation,
                'all_items_representation': all_representations,

                # Downstream Contrastive Learning
                'fst_aug_sequence_representation': last_fst_aug_embeddings,
                'snd_aug_sequence_representation': last_snd_aug_embeddings,

                # Interest-level Contrastive Learning
                'sequential_representation': sequential_representation,
                'graph_representation': graph_representation,

                # Feature-level Contrastive Learning (users)
                'user_graph_user_embeddings': user_graph_user_embeddings,
                'common_graph_user_embeddings': common_graph_user_embeddings,

                # Feature-level Contrastive Learning (items)
                'item_graph_item_embeddings': item_graph_item_embeddings,
                'common_graph_item_embeddings': common_graph_item_embeddings
            }
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
