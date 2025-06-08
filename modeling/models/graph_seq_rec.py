import torch
import torch.nn as nn

from models.base import SequentialTorchModel
from utils import DEVICE, create_masked_tensor


class GraphSeqRecModel(SequentialTorchModel, config_name="graph_seq_rec"):
    def __init__(
        self,
        sequence_prefix,
        positive_prefix,
        negative_prefix,
        candidate_prefix,
        common_graph,
        user_graph,
        item_graph,
        num_hops,
        graph_dropout,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.0,
        use_ce=False,
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
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix

        self._use_ce = use_ce

        self._common_graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph
        self._num_hops = num_hops
        self._graph_dropout = graph_dropout

        self._output_projection = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim
        )

        self._bias = nn.Parameter(data=torch.zeros(num_items + 2), requires_grad=True)

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            positive_prefix=config["positive_prefix"],
            negative_prefix=config["negative_prefix"],
            candidate_prefix=config["candidate_prefix"],
            common_graph=kwargs["graph"],
            user_graph=kwargs["user_graph"],
            item_graph=kwargs["item_graph"],
            num_hops=config["num_hops"],
            graph_dropout=config["graph_dropout"],
            num_items=kwargs["num_items"],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_layers=config["num_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            use_ce=config.get("use_ce", False),
            initializer_range=config.get("initializer_range", 0.02),
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

        for _ in range(self._num_hops):
            embeddings = torch.sparse.mm(graph_dropped, embeddings)

        return embeddings

    def forward(self, inputs):
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        lengths = inputs["{}.length".format(self._sequence_prefix)]  # (batch_size)

        common_graph_embeddings = self._apply_graph_encoder(
            embeddings=self._item_embeddings.weight, graph=self._item_graph
        )  # (num_items + 2, embedding_dim)

        embeddings = common_graph_embeddings[
            all_sample_events
        ]  # (all_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        position_embeddings = self._position_embeddings(
            positions
        )  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        embeddings = (
            embeddings + position_embeddings
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        if self._is_causal:
            causal_mask = (
                torch.tril(
                    torch.tile(mask.unsqueeze(1), dims=[self._num_heads, seq_len, 1])
                )
                .bool()
                .to(DEVICE)
            )  # (seq_len, seq_len)
            embeddings = self._encoder(
                src=embeddings,
                mask=~causal_mask,
            )  # (batch_size, seq_len, embedding_dim)
        else:
            embeddings = self._encoder(
                src=embeddings, src_key_padding_mask=~mask
            )  # (batch_size, seq_len, embedding_dim)

        if self._use_ce:
            embeddings = self._output_projection(
                embeddings
            )  # (batch_size, seq_len, embedding_dim)
            embeddings = torch.nn.functional.gelu(
                embeddings
            )  # (batch_size, seq_len, embedding_dim)
            embeddings = torch.einsum(
                "bsd,nd->bsn", embeddings, self._item_embeddings.weight
            )  # (batch_size, seq_len, num_items)
            embeddings += self._bias[None, None, :]  # (batch_size, seq_len, num_items)
        else:
            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)

        if self.training:  # training mode
            if self._use_ce:
                return {"logits": embeddings[mask]}
            else:
                all_positive_sample_events = inputs[
                    "{}.ids".format(self._positive_prefix)
                ]  # (all_batch_events)
                all_negative_sample_events = inputs[
                    "{}.ids".format(self._negative_prefix)
                ]  # (all_batch_events)

                all_sample_embeddings = embeddings[
                    mask
                ]  # (all_batch_events, embedding_dim)
                all_positive_sample_embeddings = self._item_embeddings(
                    all_positive_sample_events
                )  # (all_batch_events, embedding_dim)
                all_negative_sample_embeddings = self._item_embeddings(
                    all_negative_sample_events
                )  # (all_batch_events, embedding_dim)

                return {
                    "current_embeddings": all_sample_embeddings,
                    "positive_embeddings": all_positive_sample_embeddings,
                    "negative_embeddings": all_negative_sample_embeddings,
                }
        else:  # eval mode
            if self._use_ce:
                last_embeddings = self._get_last_embedding(
                    embeddings, mask
                )  # (batch_size, num_items)

                if "{}.ids".format(self._candidate_prefix) in inputs:
                    candidate_events = inputs[
                        "{}.ids".format(self._candidate_prefix)
                    ]  # (all_batch_candidates)
                    candidate_lengths = inputs[
                        "{}.length".format(self._candidate_prefix)
                    ]  # (batch_size)

                    candidate_ids = torch.reshape(
                        candidate_events,
                        (candidate_lengths.shape[0], candidate_lengths[0]),
                    )  # (batch_size, num_candidates)
                    candidate_scores = last_embeddings.gather(
                        dim=1, index=candidate_ids
                    )  # (batch_size, num_candidates)
                else:
                    candidate_scores = last_embeddings  # (batch_size, num_items + 2)
                    candidate_scores[:, 0] = -torch.inf
                    candidate_scores[:, self._num_items + 1 :] = -torch.inf
            else:
                if "{}.ids".format(self._candidate_prefix) in inputs:
                    candidate_events = inputs[
                        "{}.ids".format(self._candidate_prefix)
                    ]  # (all_batch_candidates)
                    candidate_lengths = inputs[
                        "{}.length".format(self._candidate_prefix)
                    ]  # (batch_size)

                    candidate_embeddings = self._item_embeddings(
                        candidate_events
                    )  # (all_batch_candidates, embedding_dim)

                    candidate_embeddings, _ = create_masked_tensor(
                        data=candidate_embeddings, lengths=candidate_lengths
                    )  # (batch_size, num_candidates, embedding_dim)

                    candidate_scores = torch.einsum(
                        "bd,bnd->bn", last_embeddings, candidate_embeddings
                    )  # (batch_size, num_candidates)
                else:
                    candidate_embeddings = (
                        self._item_embeddings.weight
                    )  # (num_items, embedding_dim)
                    candidate_scores = torch.einsum(
                        "bd,nd->bn", last_embeddings, candidate_embeddings
                    )  # (batch_size, num_items)
                    candidate_scores[:, 0] = -torch.inf
                    candidate_scores[:, self._num_items + 1 :] = -torch.inf

            return candidate_scores
