from collections import defaultdict

import torch
from torch import nn

from models import SequentialTorchModel
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

        self._codebook_sizes = rqvae_model.codebook_sizes

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(self._codebook_sizes) + 2, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 1, embedding_dim=embedding_dim
        )

        self.sem_id_len = 4
        self._codebook_item_embeddings_stacked = nn.Parameter(
            torch.stack([
                *[codebook for codebook in rqvae_model.codebooks],
                self.create_new_codebook(rqvae_model.codebooks[-1])
            ]),
            requires_grad=True
        )

        self._init_weights(initializer_range)

        self._item_id_to_semantic_id = item_id_to_semantic_id
        self.item_ids = item_ids

    @staticmethod
    def create_new_codebook(last_codebook, scale=1):

        new_codebook = torch.randn_like(last_codebook.data)

        norm_last = torch.norm(last_codebook, dim=1).mean()
        norm_new = torch.norm(new_codebook, dim=1).mean()
        scaling_factor = (norm_last / (scale * norm_new))

        return new_codebook * scaling_factor

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

    def _item_id_to_semantic_embedding(self, events):
        # TODO убрать sum потом
        return self._get_embeddings(self._item_id_to_semantic_id[events - 1]).sum(dim=1).squeeze(1)

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

        item_embeddings = self._item_id_to_semantic_embedding(self.item_ids)

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

    def _get_embeddings(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: набор из sem ids (n, sem_id_len)
        :return: эмбеддинг для каждого sem_id из набора (n, sem_id_len, embedding_dim)
        """
        assert sem_ids.shape[1] == self.sem_id_len
        # [[1, 2, 3, 4], [5, 6, 7, 8]] TODO проверить еще раз

        stacked_codebooks = self._codebook_item_embeddings_stacked.reshape(
            self.sem_id_len * self._codebook_sizes[0], self._embedding_dim)

        offsets = torch.tensor([0, 256, 512, 768], dtype=torch.long, device=DEVICE, requires_grad=False)
        sem_ids_with_offsets = sem_ids + offsets.unsqueeze(0)

        return stacked_codebooks[sem_ids_with_offsets]

    def assert_item_range(self, events):
        return torch.all((1 <= events) & (events <= self._item_id_to_semantic_id.shape[0])).item()

    def get_item_embeddings(self, events):
        assert len(events.shape) == 1
        assert self.assert_item_range(events)
        sids = self._item_id_to_semantic_id[events - 1]
        assert sids.shape == (events.shape[0], self.sem_id_len)
        sid_embs = self._get_embeddings(sids)
        assert sid_embs.shape == (events.shape[0], self.sem_id_len, self._embedding_dim)
        # # TODO убрать sum потом
        # result = sid_embs.sum(dim=1).squeeze(1)
        # assert result.shape ==  (events.shape[0], self._embedding_dim)
        # return result
        result = sid_embs.reshape(self.sem_id_len * events.shape[0],
                                  self._embedding_dim)  # (self.sem_id_len * ..., embedding_dim)
        assert result.shape == (self.sem_id_len * events.shape[0], self._embedding_dim)
        return result

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
