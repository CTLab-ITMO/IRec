import torch
import torch.nn as nn

from models.base import SequentialTorchModel


class Bert4RecModelCLS(SequentialTorchModel, config_name="bert4rec_cls"):
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
        activation="gelu",
        layer_norm_eps=1e-5,
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
            is_causal=False,
        )
        self._sequence_prefix = sequence_prefix
        self._labels_prefix = labels_prefix

        self._output_projection = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )

        self._bias = nn.Parameter(data=torch.zeros(num_items + 2), requires_grad=True)

        self._init_weights(initializer_range)

        self._cls_token = nn.Parameter(torch.rand(embedding_dim))

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            labels_prefix=config["labels_prefix"],
            num_items=kwargs["num_items"],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_layers=config["num_layers"],
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
            events=all_sample_events, lengths=all_sample_lengths, add_cls_token=True
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        embeddings = self._output_projection(
            embeddings
        )  # (batch_size, seq_len, embedding_dim)
        predictions = embeddings[:, 0, :]  # (batch_size, embedding_dim)

        if self.training:  # training mode
            candidates = self._item_embeddings(
                inputs["{}.ids".format(self._labels_prefix)]
            )  # (batch_size, embedding_dim)

            return {"predictions": predictions, "candidates": candidates}
        else:  # eval mode
            candidate_scores = torch.einsum(
                "bd,nd->bn", predictions, self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1 :] = -torch.inf

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
