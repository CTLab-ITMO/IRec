from models.base import SequentialTorchModel

import torch


class TigerModel(SequentialTorchModel, config_name='tiger'):

    def __init__(
            self,
            sequence_prefix,
            positive_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
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
        self._positive_prefix = positive_prefix

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
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
        print(inputs)
        
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)

            all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events
            )  # (all_batch_events, embedding_dim)

            all_embeddings = self._item_embeddings.weight  # (num_items + 2, embedding_dim)

            all_scores = torch.einsum(
                'ad,nd->an',
                all_sample_embeddings,
                all_embeddings
            )  # (all_batch_events, num_items + 2)
            positive_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=all_positive_sample_events[..., None]
            )  # (all_batch_items, 1)

            return {
                'positive_scores': positive_scores,
                'negative_scores': all_scores
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            _, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices.repeat(4, 1)
