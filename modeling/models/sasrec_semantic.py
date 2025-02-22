import torch
from .tiger import TigerModel
from models import SequentialTorchModel
from utils import DEVICE, create_masked_tensor
from torch import nn


class SasRecSemanticModel(SequentialTorchModel, config_name="sasrec_semantic"):
    def __init__(
        self,
        rqvae_model,
        item_id_to_semantic_id,
        item_id_to_residual,
        sequence_prefix,
        positive_prefix,
        num_items,
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

        self._codebook_sizes = rqvae_model.codebook_sizes

        self._item_id_to_semantic_id = item_id_to_semantic_id
        self._item_id_to_residual = item_id_to_residual

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(self._codebook_sizes) + 2, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._init_weights(initializer_range)

        self._codebook_item_embeddings_stacked = torch.stack(
            [codebook for codebook in rqvae_model.codebooks]
        )
        self._codebook_item_embeddings_stacked.requires_grad = (
            False  # TODOPK maybe unfreeeze later
        )

        item_ids = torch.arange(1, len(item_id_to_semantic_id) + 1)

        self._item_id_to_semantic_embedding, self._item_id_to_full_embedding = (
            self.get_init_item_embeddings(item_ids)
        )

        self._item_id_to_semantic_embedding = nn.Parameter(
            self._item_id_to_semantic_embedding, requires_grad=False
        )
        self._item_id_to_full_embedding = nn.Parameter(
            self._item_id_to_full_embedding, requires_grad=False
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
            all_sample_events, all_sample_lengths * (len(self._codebook_sizes) + 1)
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)

            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)

            all_embeddings = (
                self._item_id_to_full_embedding
            )  # (num_items, embedding_dim)

            # a -- all_batch_events, n -- num_items, d -- embedding_dim
            all_scores = torch.einsum(
                "ad,nd->an", last_embeddings, all_embeddings
            )  # (batch_size, num_items)

            positive_scores = torch.gather(
                input=all_scores, dim=1, index=all_positive_sample_events[..., None]
            )  # (batch_size, 1)

            sample_ids, _ = create_masked_tensor(
                data=all_sample_events, lengths=all_sample_lengths
            )  # (batch_size, seq_len)

            negative_scores = torch.scatter(
                input=all_scores,
                dim=1,
                index=sample_ids,
                src=torch.ones_like(sample_ids) * (-torch.inf),
            )  # (all_batch_events, num_items)

            return {
                "positive_scores": positive_scores,
                "negative_scores": negative_scores,
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                "bd,nd->bn", last_embeddings, self._item_id_to_full_embedding
            )  # (batch_size, num_items)

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices

    def get_item_embeddings(self, events):
        embs = self._item_id_to_semantic_embedding[
            events - 1
        ]  # len(events), len(self._codebook_sizes) + 1, embedding_dim
        return embs.view(
            len(events) * (len(self._codebook_sizes) + 1), self._embedding_dim
        )

    def get_init_item_embeddings(self, events):
        # convert to semantic ids
        semantic_ids = self._item_id_to_semantic_id[
            events - 1
        ]  # len(events), len(codebook_sizes)

        result = []
        for semantic_id in semantic_ids:
            item_repr = []
            for codebook_idx, codebook_id in enumerate(semantic_id):
                item_repr.append(
                    self._codebook_item_embeddings_stacked[codebook_idx][codebook_id]
                )
            result.append(torch.stack(item_repr))

        semantic_embeddings = torch.stack(result)

        # get residuals
        residual = self._item_id_to_residual[events - 1]
        # text_embeddings = self._item_id_to_text_embedding[events - 1]
        # residual = text_embeddings - semantic_embeddings.sum(dim=1)
        residual = residual.unsqueeze(1)

        # get true item embeddings
        item_embeddings = torch.cat(
            [semantic_embeddings, residual], dim=1
        )  # len(events), len(self._codebook_sizes) + 1, embedding_dim

        full_embeddings = item_embeddings.sum(dim=1)

        return item_embeddings, full_embeddings

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
