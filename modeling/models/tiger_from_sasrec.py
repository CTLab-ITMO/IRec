import torch
from torch import nn

from models import SequentialTorchModel
from utils import DEVICE, create_masked_tensor, get_activation_function
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

        self._codebook_sizes = [256, 256, 256]

        self.sem_id_len = len(self._codebook_sizes)

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=self.sem_id_len + 1, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._bos_weight = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros(embedding_dim),
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range,
            ),
            requires_grad=True,  # TODOPK added for bos
        )

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

        num_decoder_layers = 2

        self._decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_decoder_layers
        )

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 1, embedding_dim=embedding_dim
        )

        self.codebooks = nn.Parameter(torch.stack([
            rqvae_model.codebooks[0],
            rqvae_model.codebooks[1],
            rqvae_model.codebooks[2]
        ]), requires_grad=True)
        # эмбеддинги для каждого из айтемов, потом уменьшать
        # sasrec train==valid
        #
        self._init_weights(initializer_range)

        self._item_id_to_semantic_id = item_id_to_semantic_id
        self.item_ids = torch.tensor(item_ids, device=DEVICE)
        # if __name__ == '__main__':
        #     for (key, value) in model.named_parameters:
        #         print(key, value.shape)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, sids, residuals, ids = TigerModel.init_rqvae(config)

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=sids,
            item_ids=ids,
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

        encoder_embeddings, encoder_mask = self._apply_sequential_encoder(
            all_sample_events,
            all_sample_lengths * self.sem_id_len,
            user_embeddings=user_embeddings,
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        item_embeddings = self.get_init_item_embeddings().sum(dim=1)
        if self.training:
            label_events = inputs["{}.ids".format(self._positive_prefix)]
            label_lengths = inputs["{}.length".format(self._positive_prefix)]
            decoder_inputs = self.get_item_embeddings(label_events)  # (all_batch_events, embedding_dim)
            assert torch.all(label_lengths == 1)
            tgt_embeddings, tgt_mask = create_masked_tensor(
                data=decoder_inputs, lengths=label_lengths * self.sem_id_len
            )  # (batch_size, dec_seq_len, embedding_dim), (batch_size, dec_seq_len)

            decoder_outputs = self._apply_decoder(
                tgt_embeddings,
                tgt_mask,
                encoder_embeddings,
                encoder_mask
            )  # (batch_size, label_len, embedding_dim)
            decoder_prefix_scores = torch.einsum(
                "bsd,scd->bsc",
                decoder_outputs,
                self.codebooks,
            )
            semantic_ids = self._item_id_to_semantic_id[label_events - 1]
            return {
                "logits": decoder_prefix_scores.reshape(-1, decoder_prefix_scores.shape[2]),
                "semantic.labels.ids": semantic_ids.reshape(-1)
            }
        else:  # eval mode
            # b - batch_size, n - num_candidates, d - embedding_dim
            decoder_outputs = self._apply_decoder_autoregressive(
                encoder_embeddings, encoder_mask
            )  # (batch_size, sem_id_len, (batch_size, self.sem_id_len + 1, embedding_dim)
            decoder_outputs = decoder_outputs[:, :-1, :]

            full_embeddings = decoder_outputs.sum(dim=1)
            candidate_scores = torch.einsum(
                "bd,nd->bn",
                full_embeddings,
                item_embeddings,
            )  # (batch_size, num_items)

            _, indices = torch.topk(candidate_scores, k=20, dim=-1, largest=True)

            return indices + 1

    def get_item_embeddings(self, events):
        sids = self._item_id_to_semantic_id[events - 1]
        assert sids.shape == (events.shape[0], self.sem_id_len)
        result = self._get_embeddings(sids)
        result_reshaped = (result.reshape(self.sem_id_len * events.shape[0],
                                          self._embedding_dim))  # (4 * ..., embedding_dim)
        assert result[0].shape == result_reshaped[:self.sem_id_len].shape
        assert torch.allclose(result[0], result_reshaped[:self.sem_id_len])
        assert result_reshaped.shape == (self.sem_id_len * events.shape[0], self._embedding_dim)
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
        offsets = torch.tensor([0, 256, 512], device=DEVICE)
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
        assert result.shape == (self.item_ids.shape[0], self.sem_id_len, self._embedding_dim)
        return result

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // self.sem_id_len  # 5 5 5 5 4 4 4 4 ..., +1 for residual

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            x = len(self._codebook_sizes) - 1 - x % self.sem_id_len
            # print(f"encoder {x[:10]}")
            # 0 1 2 4 0 1 2 4 ... # len(self._codebook_sizes) + 1 = 4 for residual
            if len(x) > 9:
                assert torch.all(x[:9] == torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=DEVICE))
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

    def _apply_decoder(
            self, tgt_embeddings, tgt_mask, encoder_embeddings, encoder_mask
    ):
        # decoder_time = time.time()
        # print(f"tgt embeddings shape: {tgt_embeddings.shape}")
        # print(f"tgt mask shape: {tgt_mask.shape}")

        batch_size = tgt_embeddings.shape[0]
        bos_embeddings = self._bos_weight.unsqueeze(0).expand(
            batch_size, 1, -1
        )  # (batch_size, 1, embedding_dim)

        tgt_embeddings = torch.cat(
            [bos_embeddings, tgt_embeddings[:, :-1, :]], dim=1
        )  # remove residual by using :-1
        label_len = tgt_mask.shape[1]
        # print("shape", tgt_embeddings.shape)
        assert label_len == self.sem_id_len
        # print(f"bos {bos_embeddings}")
        lengths = torch.ones(size=[batch_size], device=DEVICE, dtype=torch.long) * label_len
        # print(f"decoder lengths {lengths.shape}, {tgt_mask.shape}")
        position_embeddings = self._decoder_pos_embeddings(lengths, tgt_mask)
        # print(f"final decoder pos embs {position_embeddings[:100, 0, :5]}")
        assert torch.allclose(position_embeddings[~tgt_mask], tgt_embeddings[~tgt_mask])

        # print("pos emb shape", position_embeddings.shape)
        tgt_embeddings = tgt_embeddings + position_embeddings

        tgt_embeddings[~tgt_mask] = 0

        causal_mask = (
            torch.tril(torch.ones(label_len, label_len)).bool().to(DEVICE)
        )  # (dec_seq_len, dec_seq_len) #TODO проверить
        # print(f"290 causal_mask {causal_mask}")
        decoder_outputs = self._decoder(
            tgt=tgt_embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
        )  # (batch_size, dec_seq_len, embedding_dim)
        # now = time.time()
        # print(f"-----decoder: {(now - decoder_time) * 1000:.2f} ms")
        return decoder_outputs

    def _apply_decoder_autoregressive(self, encoder_embeddings, encoder_mask):
        # regres_time = time.time()
        batch_size = encoder_embeddings.shape[0]
        embedding_dim = encoder_embeddings.shape[2]

        tgt_embeddings = (
            self._bos_weight.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 1, embedding_dim)
        )

        for step in range(self.sem_id_len):  # semantic_id_seq + residual
            index = self.sem_id_len if step == 0 else step - 1  # эмбеддинг bos последний потому что
            # print(f"INDEX STEP index: {index}; step:{step}")
            causal_mask = (
                torch.tril(torch.ones(step + 1, step + 1)).bool().to(DEVICE)
            )  # (dec_seq_len, dec_seq_len)
            # print(f"decoder mask {causal_mask}")

            decoder_output = self._decoder(
                tgt=tgt_embeddings,
                memory=encoder_embeddings,
                tgt_mask=~causal_mask,
                memory_key_padding_mask=~encoder_mask,
            )
            # print(f"real_dec_output {decoder_output}")

            # TODOPK ASK it is not true?
            # assert that prelast items don't change
            # assert decoder changes only last index in dim = 1

            next_token_embedding = decoder_output[:, -1, :]  # batch_size x embedding_dim

            assert step < self.sem_id_len  # TODO почему?

            codebook = self.codebooks[step]  # codebook_size x embedding_dim
            closest_semantic_ids = torch.argmax(
                torch.einsum("bd,cd->bc", next_token_embedding, codebook), dim=1
            )  # batch_size

            next_token_embedding = codebook[
                closest_semantic_ids
            ]  # batch_size x embedding_dim

            last_position_embedding = self._codebook_embeddings(
                torch.full((batch_size,), index, device=DEVICE)
            )  # (batch_size, embedding_dim)
            assert last_position_embedding.shape == tgt_embeddings[:, -1, :].shape
            assert tgt_embeddings.shape == torch.Size(
                [batch_size, step + 1, embedding_dim]
            )
            next_token_embedding += last_position_embedding
            tgt_embeddings = torch.cat(
                [tgt_embeddings, next_token_embedding.unsqueeze(1)], dim=1
            )

        # now = time.time()
        return tgt_embeddings

    def _decoder_pos_embeddings(self, lengths, mask):
        def codebook_lambda(x):  # TODO разобраться и посмотреть
            # print("decoder_pos_embeddings", self.sem_id_len, x[:10])
            non_bos = x < self.sem_id_len - 1
            # print(non_bos)
            x[non_bos] = (self.sem_id_len - 2) - x[non_bos]
            x[~non_bos] = self.sem_id_len
            # print(x[:20])
            # print(non_bos[:20])
            if len(x) > 9:
                assert torch.all(x[:9] == torch.tensor([3, 0, 1, 3, 0, 1, 3, 0, 1], device=DEVICE))
            return x  # 4, 0, 1, 2, 4, 0, 1, 2 ... sem_id_len = 4 for bos

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return codebook_embeddings
