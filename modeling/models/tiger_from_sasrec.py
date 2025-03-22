from collections import defaultdict

import torch
from models import SequentialTorchModel
from torch import nn
from utils import DEVICE, create_masked_tensor

from .tiger import TigerModel


class TigerFromSasRec(SequentialTorchModel, config_name="tiger_from_sasrec"):
    def __init__(
            self,
            rqvae_model,
            item_id_to_semantic_id,
            item_ids,
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
        print("hello it is tiger from sasrec")
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix

        self._num_users = num_users

        self._codebook_sizes = [256, 256, 256, 256]

        self.sem_id_len = len(self._codebook_sizes)

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=self.sem_id_len + 1, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 1, embedding_dim=embedding_dim
        )

        self.codebooks = nn.Parameter(torch.stack([
            rqvae_model.codebooks[0],
            rqvae_model.codebooks[1],
            rqvae_model.codebooks[2],
            torch.zeros_like(rqvae_model.codebooks[2], requires_grad=False, device=DEVICE)
        ]), requires_grad=True)
        self._init_weights(initializer_range)

        self._item_id_to_semantic_id = item_id_to_semantic_id
        self.item_ids = item_ids

    @staticmethod
    def get_full_sids(sids, ids, codebook_size):
        assert sids.shape[0] == ids.shape[0]

        ids = ids.detach().to(DEVICE)
        sids = sids.detach().to(DEVICE)

        key = torch.tensor([codebook_size ** i for i in range(sids.shape[1])], device=DEVICE, requires_grad=False)

        shuffled_indices = torch.randperm(len(ids), device=DEVICE)
        shuffled_ids = ids[shuffled_indices]
        shuffled_sids = sids[shuffled_indices]

        col_tokens = torch.zeros(ids.shape, device=DEVICE, dtype=torch.long, requires_grad=False)

        hash_dict = defaultdict(int)

        for (i, sid) in enumerate(shuffled_sids):
            sid_hash = (sid * key).sum().item()
            col_tokens[i] = hash_dict[sid_hash]
            hash_dict[sid_hash] += 1

        full_sids = torch.cat([shuffled_sids, col_tokens.unsqueeze(1)], dim=1)
        unshuffled_indices = shuffled_indices.argsort()

        return (
            shuffled_ids[unshuffled_indices].detach(),
            full_sids[unshuffled_indices].detach()
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, sids, residuals, ids = TigerModel.init_rqvae(config)

        ids_tensor = torch.tensor(ids, dtype=torch.long, device=DEVICE, requires_grad=False)
        sids_tensor = sids.clone().detach()
        item_ids, semantic_ids = cls.get_full_sids(sids_tensor, ids_tensor, rqvae_model.codebook_sizes[0])

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=semantic_ids,
            item_ids=item_ids,
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
            all_sample_lengths * self.sem_id_len,
            user_embeddings=user_embeddings,
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        last_embeddings = self._get_last_embedding(
            embeddings, mask
        )  # (batch_size, embedding_dim)

        item_embeddings = self.get_init_item_embeddings().sum(dim=1)

        if self.training:  # training mode
            # positives
            in_batch_positive_events = inputs[
                "{}.ids".format(self._positive_prefix)
            ]  # (all_batch_events)
            in_batch_positive_embeddings = item_embeddings[
                in_batch_positive_events - 1
                ]  # (all_batch_events, embedding_dim)
            # print(f"last_embeddings={last_embeddings.shape} in_batch_positive_embeddings {in_batch_positive_embeddings.shape}")
            positive_scores = torch.einsum(
                "bd,bd->b", last_embeddings, in_batch_positive_embeddings
            )  # (all_batch_events)

            # TODOPK normalize in all models embeddings for stability

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
        sids = self._item_id_to_semantic_id[events - 1]
        assert sids.shape == (events.shape[0], self.sem_id_len)
        result = self._get_embeddings(sids)
        result_reshaped = (result.reshape(self.sem_id_len * events.shape[0],
                                                    self._embedding_dim))  # (4 * ..., embedding_dim)
        assert result[0].shape == result_reshaped[:4].shape
        assert torch.allclose(result[0],result_reshaped[:4])
        assert result_reshaped.shape == (4 * events.shape[0], self._embedding_dim)
        # print(f"label items, events.shape: {events.shape}, result.shape {result.shape}")
        return result_reshaped

    def calculate_full(self, sem_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embeddings(sem_ids).sum(1)  # (n, embedding_dim)

    def _get_embeddings(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: набор из sem ids (n, sem_id_len)
        :return: эмбеддинг для каждого sem_id из набора (n, sem_id_len, embedding_dim)
        """
        assert sem_ids.shape[1] == self.sem_id_len
        # [[1, 2, 3, 4], [5, 6, 7, 8]] TODO проверить еще раз
        offsets = torch.tensor([0, 256, 512, 768], device=DEVICE)
        stacked_codebooks = self.codebooks.reshape(
            self.sem_id_len * self._codebook_sizes[0], self._embedding_dim)
        assert self.codebooks[0].shape == stacked_codebooks[:256].shape
        assert torch.allclose(self.codebooks[0], stacked_codebooks[:256])
        # assert torch.allclose(stacked_codebooks[768:], torch.zeros_like(stacked_codebooks[768:]))
        sem_ids_with_offsets = sem_ids + offsets.unsqueeze(0)

        return stacked_codebooks[sem_ids_with_offsets]

    def get_init_item_embeddings(self):
        sids = self._item_id_to_semantic_id[self.item_ids - 1]
        assert sids.shape == (self.item_ids.shape[0], self.sem_id_len)
        result = self._get_embeddings(sids)
        assert result.shape == (self.item_ids.shape[0], 4, self._embedding_dim)
        return result

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // self.sem_id_len  # 5 5 5 5 4 4 4 4 ..., +1 for residual

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            x = len(self._codebook_sizes) - x % self.sem_id_len
            x[x == len(self._codebook_sizes)] = self.sem_id_len
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
