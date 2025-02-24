import torch
from models.tiger import TigerModel
from models import SequentialTorchModel
from utils import DEVICE, create_masked_tensor


class SasRecFreezedModel(SequentialTorchModel, config_name="sasrec_freezed"):
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

        self._init_weights(initializer_range)

        self._codebook_item_embeddings_stacked = torch.nn.Parameter(
            torch.stack([codebook for codebook in rqvae_model.codebooks]),
            requires_grad=False,  # TODOPK compare with unfrozen codebooks
        )
        self._item_id_to_semantic_id = item_id_to_semantic_id
        self._item_id_to_residual = item_id_to_residual

        item_ids = torch.arange(1, len(item_id_to_semantic_id) + 1)
        self._item_id_to_semantic_embedding = self.get_init_item_embeddings(item_ids)
        self._item_id_to_semantic_embedding = torch.nn.Parameter(
            self._item_id_to_semantic_embedding.sum(dim=1), requires_grad=False
        )  # len(events), embedding_dim

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
        residual = residual.unsqueeze(1)

        # get true item embeddings
        item_embeddings = torch.cat(
            [semantic_embeddings, residual], dim=1
        )  # len(events), len(self._codebook_sizes) + 1, embedding_dim

        return item_embeddings

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, semantic_ids, residuals, item_ids = TigerModel.init_rqvae(config)

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

    def get_item_embeddings(self, events=None):
        if events is None:
            return self._item_id_to_semantic_embedding
        else:
            return self._item_id_to_semantic_embedding[events - 1]

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

        if self.training:  # training mode
            all_positive_sample_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)

            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)

            all_embeddings = torch.cat(
                [
                    torch.zeros(1, self._embedding_dim, device=DEVICE),
                    self._item_id_to_semantic_embedding,
                    torch.zeros(1, self._embedding_dim, device=DEVICE),
                ],
                dim=0,
            )  # (num_items + 2, embedding_dim)

            # a -- all_batch_events, n -- num_items + 2, d -- embedding_dim
            all_scores = torch.einsum(
                "ad,nd->an", last_embeddings, all_embeddings
            )  # (batch_size, num_items + 2)

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
            )  # (all_batch_events, num_items + 2)
            negative_scores[:, 0] = -torch.inf  # Padding idx
            negative_scores[:, self._num_items + 1 :] = -torch.inf  # Mask idx

            return {
                "positive_scores": positive_scores,
                "negative_scores": negative_scores,
                "sample_ids": sample_ids,
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                "bd,nd->bn", last_embeddings, self.get_item_embeddings()
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf  # Padding id
            candidate_scores[:, self._num_items + 1 :] = -torch.inf  # Mask id

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
