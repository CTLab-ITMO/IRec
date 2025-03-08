import torch
from models import SequentialTorchModel
from torch import nn
from utils import DEVICE, create_masked_tensor

from .tiger import TigerModel


class SasRecSemanticModel(SequentialTorchModel, config_name="sasrec_semantic"):
    def __init__(
        self,
        rqvae_model,
        item_id_to_semantic_id,
        item_id_to_residual,
        sequence_prefix,
        positive_prefix,
        negative_prefix,
        num_items,
        num_users,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
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
        self._negative_prefix = negative_prefix

        self._num_users = num_users

        self._codebook_sizes = rqvae_model.codebook_sizes

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(self._codebook_sizes) + 2, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 1, embedding_dim=embedding_dim
        )

        self._init_weights(initializer_range)

        self._item_id_to_semantic_embedding = nn.Parameter(
            self.get_init_item_embeddings(
                rqvae_model, item_id_to_semantic_id, item_id_to_residual
            ),
            requires_grad=False,
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, semantic_ids, residuals, _ = TigerModel.init_rqvae(config)

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=semantic_ids,
            item_id_to_residual=residuals,
            sequence_prefix=config["sequence_prefix"],
            positive_prefix=config["positive_prefix"],
            negative_prefix=config["negative_prefix"],
            num_items=kwargs["num_items"],
            num_users=kwargs["num_users"],
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

        user_embeddings = self._user_embeddings(inputs["user.ids"])

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events,
            all_sample_lengths * (len(self._codebook_sizes) + 1),
            user_embeddings=user_embeddings,
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        last_embeddings = self._get_last_embedding(
            embeddings, mask
        )  # (batch_size, embedding_dim)

        item_embeddings = self._item_id_to_semantic_embedding.sum(dim=1)

        if self.training:  # training mode
            # positives
            in_batch_positive_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)
            in_batch_positive_embeddings = item_embeddings[
                in_batch_positive_events - 1
            ]  # (all_batch_events, embedding_dim)
            positive_scores = torch.einsum(
                "bd,bd->b", last_embeddings, in_batch_positive_embeddings
            )  # (all_batch_events)

            # negatives
            in_batch_negative_events = inputs[
                "{}.ids".format(self._negative_prefix)
            ]  # (all_batch_events)
            in_batch_negative_embeddings = item_embeddings[
                in_batch_negative_events - 1
            ]  # (all_batch_events, embedding_dim)
            negative_scores = torch.einsum(
                "bd,bd->b", last_embeddings, in_batch_negative_embeddings
            )  # (all_batch_events)

            return {
                "positive_scores": positive_scores,
                "negative_scores": negative_scores,
            }
        else:  # eval mode
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                "bd,nd->bn",
                last_embeddings,
                item_embeddings,
            )  # (batch_size, num_items)

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices + 1  # tensors are 0 indexed

    def get_item_embeddings(self, events):
        embs = self._item_id_to_semantic_embedding[
            events - 1
        ]  # len(events), len(self._codebook_sizes) + 1, embedding_dim
        return embs.reshape(-1, self._embedding_dim)

    def get_init_item_embeddings(
        self, rqvae_model, item_id_to_semantic_id, item_id_to_residual
    ):
        codebooks = torch.stack([codebook for codebook in rqvae_model.codebooks])

        result = []
        for semantic_id in item_id_to_semantic_id:
            item_repr = []
            for codebook_idx, codebook_id in enumerate(semantic_id):
                item_repr.append(codebooks[codebook_idx][codebook_id])
            result.append(torch.stack(item_repr))

        semantic_embeddings = torch.stack(
            result
        )  # len(events), len(codebook_sizes), embedding_dim

        residual = item_id_to_residual.unsqueeze(1)

        # get true item embeddings
        item_embeddings = torch.cat(
            [semantic_embeddings, residual], dim=1
        )  # len(events), len(self._codebook_sizes) + 1, embedding_dim

        return item_embeddings

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // (
                len(self._codebook_sizes) + 1
            )  # 5 5 5 5 4 4 4 4 ..., +1 for residual

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            x = len(self._codebook_sizes) - x % (len(self._codebook_sizes) + 1)
            x[x == len(self._codebook_sizes)] = len(self._codebook_sizes) + 1
            # 0 1 2 4 0 1 2 4 ... # len(self._codebook_sizes) + 1 = 4 for residual
            return x

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return position_embeddings + codebook_embeddings

    def _get_position_embeddings(self, lengths, mask, position_lambda, embedding_layer):
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=DEVICE)[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        # 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0 ...

        positions = position_lambda(positions)  # (all_batch_events)

        # print(f"{positions.tolist()[:20]=}")

        assert (positions >= 0).all() and (
            positions < embedding_layer.num_embeddings
        ).all()

        position_embeddings = embedding_layer(
            positions
        )  # (all_batch_events, embedding_dim)

        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        return position_embeddings
