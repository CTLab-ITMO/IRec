import torch
from models import SequentialTorchModel
from utils import create_masked_tensor


class SasRecSemanticModel(SequentialTorchModel, config_name="sasrec_semantic"):
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
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )
        
    
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

        # item_embeddings = item_embeddings.view(-1, self._embedding_dim) # (all_batch_events, embedding_dim)

        return item_embeddings

    def forward(self, inputs):
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        all_sample_lengths = all_sample_lengths * (len(self._codebook_sizes) + 1)
        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            all_positive_sample_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)

            all_sample_embeddings = embeddings[
                mask
            ]  # (all_batch_events, embedding_dim)

            all_embeddings = (
                self._item_embeddings.weight
            )  # (num_items + 2, embedding_dim)

            # a -- all_batch_events, n -- num_items + 2, d -- embedding_dim
            all_scores = torch.einsum(
                "ad,nd->an", all_sample_embeddings, all_embeddings
            )  # (all_batch_events, num_items + 2)

            positive_scores = torch.gather(
                input=all_scores, dim=1, index=all_positive_sample_events[..., None]
            )  # (all_batch_items, 1)

            sample_ids, _ = create_masked_tensor(
                data=all_sample_events, lengths=all_sample_lengths
            )  # (batch_size, seq_len)

            sample_ids = torch.repeat_interleave(
                sample_ids, all_sample_lengths, dim=0
            )  # (all_batch_events, seq_len)

            negative_scores = torch.scatter(
                input=all_scores,
                dim=1,
                index=sample_ids,
                src=torch.ones_like(sample_ids) * (-torch.inf),
            )  # (all_batch_events, num_items + 2)
            negative_scores[:, 0] = -torch.inf  # Padding idx
            negative_scores[:, self._num_items + 1 :] = -torch.inf  # Mask idx

            last_item_mask = (
                torch.cumsum(mask.sum(dim=1), dim=0) - 1
            )  # TODO ask if correct (mask, last True in each row, index as only Trues appeared)

            return {
                "positive_scores": positive_scores,
                "negative_scores": negative_scores[last_item_mask],
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(
                embeddings, mask
            )  # (batch_size, embedding_dim)
            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                "bd,nd->bn", last_embeddings, self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf  # Padding id
            candidate_scores[:, self._num_items + 1 :] = -torch.inf  # Mask id

            _, indices = torch.topk(
                candidate_scores, k=20, dim=-1, largest=True
            )  # (batch_size, 20)

            return indices
        
    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // (len(self._codebook_sizes) + 1)  # 5 5 5 4 4 4 3 3 3 ...

        # TODO +1 for residual embedding

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            x = len(self._codebook_sizes) - x % (len(self._codebook_sizes) + 1)
            x[x == len(self._codebook_sizes)] = len(self._codebook_sizes) + 1
            # 0 1 2 3 5 0 1 2 3 5 ... # len(self._codebook_sizes) for bos, len(self._codebook_sizes) + 1 for residual
            return x

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )
        
        return position_embeddings + codebook_embeddings
