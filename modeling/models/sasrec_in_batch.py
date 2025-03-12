import torch
from models.base import SequentialTorchModel


class SasRecInBatchModel(SequentialTorchModel, config_name="sasrec_in_batch"):
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
        num_in_batch_negatives=-1,
        dropout=0.0,
        activation="relu",
        layer_norm_eps=1e-9,
        initializer_range=0.02,
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
            is_causal=True,
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._num_in_batch_negatives = num_in_batch_negatives
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            positive_prefix=config["positive_prefix"],
            num_items=kwargs["num_items"],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_layers=config["num_layers"],
            num_in_batch_negatives=config.get("num_in_batch_negatives", -1),
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

    def forward(self, inputs):
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        last_embeddings = self._get_last_embedding(
            embeddings, mask
        )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            # positives
            in_batch_positive_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (all_batch_events, embedding_dim)

            # negatives
            num_in_batch_negatives = self._num_in_batch_negatives
            batch_size = all_sample_lengths.shape[0]
            random_ids = torch.randperm(in_batch_positive_events.shape[0])
            if num_in_batch_negatives == -1:
                num_in_batch_negatives = batch_size
            in_batch_negative_ids = in_batch_positive_events[random_ids][
                :num_in_batch_negatives
            ]

            in_batch_negative_embeddings = self._item_embeddings(
                in_batch_negative_ids
            )  # (num_in_batch_negatives, embedding_dim)

            return {
                "query_embeddings": last_embeddings,
                "positive_embeddings": (
                    in_batch_positive_events,
                    in_batch_positive_embeddings,
                ),
                "negative_embeddings": (
                    in_batch_negative_ids,
                    in_batch_negative_embeddings,
                ),
            }
        else:  # eval mode
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                "bd,nd->bn", last_embeddings, self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1 :] = -torch.inf

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
