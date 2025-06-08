import torch
import torch.nn as nn

from models.base import SequentialTorchModel
from utils import create_masked_tensor


class S3RecModel(SequentialTorchModel, config_name="s3rec"):
    def __init__(
        self,
        sequence_prefix,
        positive_prefix,
        negative_prefix,
        sequence_segment_prefix,
        positive_segment_prefix,
        negative_segment_prefix,
        candidate_prefix,
        num_items,
        max_sequence_length,
        is_training,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout=0.0,
        activation="relu",
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
            is_causal=is_training,
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._sequence_segment_prefix = sequence_segment_prefix
        self._positive_segment_prefix = positive_segment_prefix
        self._negative_segment_prefix = negative_segment_prefix
        self._candidate_prefix = candidate_prefix
        self._is_training = is_training
        self._mask_item_idx = num_items + 1

        self.aap_norm = nn.Linear(embedding_dim, embedding_dim)
        self.mip_norm = nn.Linear(embedding_dim, embedding_dim)
        self.map_norm = nn.Linear(embedding_dim, embedding_dim)
        self.sp_norm = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config["sequence_prefix"],
            positive_prefix=config["positive_prefix"],
            negative_prefix=config["negative_prefix"],
            sequence_segment_prefix=config["sequence_segment_prefix"],
            positive_segment_prefix=config["positive_segment_prefix"],
            negative_segment_prefix=config["negative_segment_prefix"],
            candidate_prefix=config["candidate_prefix"],
            num_items=kwargs["num_items"],
            max_sequence_length=kwargs["max_sequence_length"],
            is_training=config["is_training"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_layers=config["num_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

    def masked_item_prediction(self, sequence_embeddings, sequence_mask, target_item):
        all_items = sequence_embeddings[sequence_mask]  # (all_batch_items, emb_dim)
        score = torch.einsum("ad,ad->a", all_items, target_item)  # (all_batch_items)
        return torch.sigmoid(score)  # (all_batch_items)

    def segment_prediction(self, context, segment):
        score = torch.einsum("bd,bd->b", self.sp_norm(context), segment)  # (batch_size)
        return torch.sigmoid(score)  # (batch_size)

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

        if self._is_training:
            if self.training:  # training mode
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
        else:
            # Masked Item Prediction
            mip_mask = (
                all_sample_events == self._mask_item_idx
            ).bool()  # (all_batch_events)
            embeddings = embeddings[mask][mip_mask]  # (all_batch_events, embedding_dim)
            positive_item_events = inputs["{}.ids".format(self._positive_prefix)][
                mip_mask
            ]  # (all_batch_events)
            negative_item_events = inputs["{}.ids".format(self._negative_prefix)][
                mip_mask
            ]  # (all_batch_events)

            positive_item_embeddings = self._item_embeddings(
                positive_item_events
            )  # (all_batch_events, embedding_dim)
            negative_item_embeddings = self._item_embeddings(
                negative_item_events
            )  # (all_batch_events, embedding_dim)

            # Sequence Prediction
            all_segment_events = inputs[
                "{}.ids".format(self._sequence_segment_prefix)
            ]  # (all_batch_events)
            all_segment_lengths = inputs[
                "{}.length".format(self._sequence_segment_prefix)
            ]  # (batch_size)
            segment_embeddings, segment_mask = self._apply_sequential_encoder(
                all_segment_events, all_segment_lengths
            )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
            last_segment_embeddings = self._get_last_embedding(
                segment_embeddings, segment_mask
            )  # (batch_size, embedding_dim)

            positive_segment_events = inputs[
                "{}.ids".format(self._positive_segment_prefix)
            ]  # (all_batch_events)
            positive_segment_lengths = inputs[
                "{}.length".format(self._positive_segment_prefix)
            ]  # (batch_size)
            positive_segment_embeddings, positive_segment_mask = (
                self._apply_sequential_encoder(
                    positive_segment_events, positive_segment_lengths
                )
            )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
            last_positive_segment_embeddings = self._get_last_embedding(
                positive_segment_embeddings, positive_segment_mask
            )  # (batch_size, embedding_dim)

            negative_segment_events = inputs[
                "{}.ids".format(self._negative_segment_prefix)
            ]  # (all_batch_events)
            negative_segment_lengths = inputs[
                "{}.length".format(self._negative_segment_prefix)
            ]  # (batch_size)
            negative_segment_embeddings, negative_segment_mask = (
                self._apply_sequential_encoder(
                    negative_segment_events, negative_segment_lengths
                )
            )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
            last_negative_segment_embeddings = self._get_last_embedding(
                negative_segment_embeddings, negative_segment_mask
            )  # (batch_size, embedding_dim)

            return {
                "positive_representation": positive_item_embeddings,
                "negative_representation": negative_item_embeddings,
                "current_representation": embeddings,
                "positive_segment_representation": last_positive_segment_embeddings,
                "negative_segment_representation": last_negative_segment_embeddings,
                "current_segment_representation": last_segment_embeddings,
            }
