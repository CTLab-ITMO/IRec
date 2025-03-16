import json
# import time
from collections import defaultdict

import torch
from torch import nn

from models.base import SequentialTorchModel
from utils import DEVICE, create_masked_tensor, get_activation_function
from .rqvae import RqVaeModel


class TigerModel(SequentialTorchModel, config_name="tiger"):
    def __init__(
            self,
            rqvae_model,
            item_id_to_semantic_id,
            item_ids,
            sequence_prefix,
            pred_prefix,
            positive_prefix,
            labels_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
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
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=True,  # TODO false на енкодере true на декодере
        )
        self.decoder_test = None

        self._sequence_prefix = sequence_prefix
        self._pred_prefix = pred_prefix
        self._positive_prefix = positive_prefix
        self._labels_prefix = labels_prefix

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

        self._decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_decoder_layers
        )

        self._decoder_layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._decoder_dropout = nn.Dropout(dropout)

        self._codebook_sizes = rqvae_model.codebook_sizes
        self._bos_weight = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros(embedding_dim),
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range,
            ),
            requires_grad=True,  # TODOPK added for bos
        )

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=(len(self._codebook_sizes) + 1) + 1, embedding_dim=embedding_dim
        )  # + 1 for bos token

        self._init_weights(initializer_range)

        self._codebook_item_embeddings_stacked = nn.Parameter(
            torch.stack([
                *[codebook for codebook in rqvae_model.codebooks],
                self.create_new_codebook(rqvae_model.codebooks[-1])
            ]),
            requires_grad=True
        )

        self._item_id_to_semantic_id = item_id_to_semantic_id
        self.item_ids = item_ids

    def create_new_codebook(self, last_codebook, scale=2):

        new_codebook = torch.randn_like(last_codebook.data)

        norm_last = torch.norm(last_codebook, dim=1).mean()
        norm_new = torch.norm(new_codebook, dim=1).mean()
        scaling_factor = (norm_last / (scale * norm_new))

        return new_codebook * scaling_factor

    @classmethod
    def init_rqvae(cls, config):
        rqvae_config = json.load(open(config["rqvae_train_config_path"]))
        rqvae_config["model"]["should_init_codebooks"] = False

        rqvae_model = RqVaeModel.create_from_config(rqvae_config["model"]).to(DEVICE)
        rqvae_model.load_state_dict(
            torch.load(config["rqvae_checkpoint_path"], weights_only=True)
        )
        rqvae_model.eval()
        for param in rqvae_model.parameters():
            param.requires_grad = False

        codebook_sizes = rqvae_model.codebook_sizes
        assert all([book_size == codebook_sizes[0] for book_size in codebook_sizes])

        embs_extractor = torch.load(config["embs_extractor_path"], weights_only=False)

        embs_extractor = embs_extractor.sort_index()

        item_ids = embs_extractor.index.tolist()
        # print(item_ids[:20])
        assert item_ids == list(range(1, len(item_ids) + 1))

        text_embeddings = torch.stack(embs_extractor["embeddings"].tolist()).to(DEVICE)

        semantic_ids, residuals = rqvae_model({"embeddings": text_embeddings})

        return rqvae_model, semantic_ids, residuals, item_ids

    def get_full_sids(self, sids, ids, codebook_size):
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
            hash = (sid * key).sum().item()
            col_tokens[i] = hash_dict[hash]
            hash_dict[hash] += 1

        full_sids = torch.cat([shuffled_sids, col_tokens.unsqueeze(1)], dim=1)
        unshuffled_indices = shuffled_indices.argsort()

        return (
            shuffled_ids[unshuffled_indices].detach(),
            full_sids[unshuffled_indices].detach()
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, sids, _, ids = cls.init_rqvae(config)
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=DEVICE, requires_grad=False)

        sids_tensor = sids.clone().detach()

        item_ids, semantic_ids = cls.get_full_sids(cls, sids_tensor, ids_tensor, rqvae_model.codebook_sizes[0])

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=semantic_ids,
            item_ids=item_ids,
            sequence_prefix=config["sequence_prefix"],
            pred_prefix=config["predictions_prefix"],
            positive_prefix=config["positive_prefix"],
            labels_prefix=config["labels_prefix"],
            num_items=rqvae_model.codebook_sizes[0],  # unused
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

    # semantic ids come with dedup token
    def forward(self, inputs):
        # forward_start_time = time.time()
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "{}.length".format(self._sequence_prefix)
        ]  # (batch_size)
        # print(f"all_sample_events.shape {all_sample_events.shape}")

        batch_embeddings = self.get_item_embeddings(all_sample_events)  # (all_batch_events, embedding_dim)

        encoder_embeddings, encoder_mask = self._apply_sequential_encoder(
            batch_embeddings, all_sample_lengths * (len(self._codebook_sizes) + 1)
        )  # (batch_size, enc_seq_len, embedding_dim), (batch_size, enc_seq_len)

        if self.training:
            label_events = inputs["{}.ids".format(self._positive_prefix)]
            label_lengths = inputs["{}.length".format(self._positive_prefix)]

            tgt_embeddings = self.get_item_embeddings(label_events)  # (all_batch_events, embedding_dim)

            decoder_outputs = self._apply_decoder(
                tgt_embeddings,
                label_lengths * (len(self._codebook_sizes) + 1),
                encoder_embeddings,
                encoder_mask,
            )  # (batch_size, label_len, embedding_dim)

            decoder_prefix_scores = torch.einsum(
                "bsd,scd->bsc",
                decoder_outputs,
                self._codebook_item_embeddings_stacked,
            )
            assert self.assert_item_range(label_events)
            semantic_ids = self._item_id_to_semantic_id[label_events - 1]
            # TODO не будет работать, сделать реверс shuffle из get full sids
            # now = time.time()
            # print(f"forward train: {(now - forward_start_time) * 1000:.2f} ms")
            return {
                "logits": decoder_prefix_scores.reshape(-1, decoder_prefix_scores.shape[2]),
                "semantic.labels.ids": semantic_ids.reshape(-1)
            }
        else:  # eval mode
            semantic_ids, tgt_embeddings = self._apply_decoder_autoregressive(
                encoder_embeddings, encoder_mask
            )  # (batch_size, len(self._codebook_sizes) + 1), (batch_size, len(self._codebook_sizes) + 2, embedding_dim)

            full_embeddings = tgt_embeddings[:, 1:, :].sum(dim=1)  # (batch_size, embedding_dim)
            candidate_scores = torch.einsum(
                "bd,nd->bn",
                full_embeddings,
                self.calculate_full(self._item_id_to_semantic_id),
            )  # (batch_size, num_items)

            _, indices = torch.topk(candidate_scores, k=20, dim=-1, largest=True)  # (batch_size, 20)

            assert self.assert_item_range(indices + 1)  # tensors are 0 indexed
            # now = time.time()
            # print(f"forward eval: {(now - forward_start_time) * 1000:.2f} ms")
            return indices + 1

    def _apply_sequential_encoder(self, embeddings, lengths, add_cls_token=False):
        # encoder_time = time.time()
        # print(f"embeddings.shape {embeddings.shape}")
        assert embeddings.shape == (sum(lengths), self._embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        position_embeddings = self._encoder_pos_embeddings(lengths, mask)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])

        embeddings = (
                embeddings + position_embeddings
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)

        embeddings[~mask] = 0

        embeddings = self._encoder(
            src=embeddings, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        # now = time.time()
        # print(f"-----encoder: {(now - encoder_time) * 1000:.2f} ms")
        return embeddings, mask

    def _apply_decoder(
            self, tgt_embeddings, label_lengths, encoder_embeddings, encoder_mask
    ):
        # decoder_time = time.time()
        tgt_embeddings, tgt_mask = create_masked_tensor(
            data=tgt_embeddings, lengths=label_lengths
        )  # (batch_size, dec_seq_len, embedding_dim), (batch_size, dec_seq_len)

        batch_size = tgt_embeddings.shape[0]
        bos_embeddings = self._bos_weight.unsqueeze(0).expand(
            batch_size, 1, -1
        )  # (batch_size, 1, embedding_dim)

        tgt_embeddings = torch.cat(
            [bos_embeddings, tgt_embeddings[:, :-1, :]], dim=1
        )  # remove residual by using :-1

        label_len = tgt_mask.shape[1]

        assert label_len == len(self._codebook_sizes) + 1

        position_embeddings = self._decoder_pos_embeddings(label_lengths, tgt_mask)
        assert torch.allclose(position_embeddings[~tgt_mask], tgt_embeddings[~tgt_mask])

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

    def _decoder_pos_embeddings(self, lengths, mask):
        def codebook_lambda(x):
            non_bos = x < len(self._codebook_sizes)
            x[non_bos] = (len(self._codebook_sizes) - 1) - x[non_bos]
            return x  # 3, 0, 1, 2, 3, 0, 1, 2 ... len(self._codebook_sizes) = 3 for bos

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return codebook_embeddings

    def _apply_decoder_autoregressive(self, encoder_embeddings, encoder_mask):
        # regres_time = time.time()
        batch_size = encoder_embeddings.shape[0]
        embedding_dim = encoder_embeddings.shape[2]

        tgt_embeddings = (
            self._bos_weight.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 1, embedding_dim)
        )

        semantic_ids = torch.tensor([], device=DEVICE, dtype=torch.int64, requires_grad=False)

        for step in range(len(self._codebook_sizes) + 1):  # semantic_id_seq + residual
            index = len(self._codebook_sizes) if step == 0 else step - 1

            last_position_embedding = self._codebook_embeddings(
                torch.full((batch_size,), index, device=DEVICE)
            )  # (batch_size, embedding_dim)

            assert last_position_embedding.shape == tgt_embeddings[:, -1, :].shape
            assert tgt_embeddings.shape == torch.Size(
                [batch_size, step + 1, embedding_dim]
            )

            curr_step_embeddings = tgt_embeddings.clone()  # TODO проверить клонится ли градиент, мб убрать
            curr_step_embeddings[:, -1, :] = (
                    tgt_embeddings[:, -1, :] + last_position_embedding
            )
            assert torch.allclose(
                tgt_embeddings[:, :-1, :], curr_step_embeddings[:, :-1, :]
            )
            tgt_embeddings = curr_step_embeddings

            # curr_embeddings[:, -1, :] = self._decoder_layernorm(curr_embeddings[:, -1, :])
            # curr_embeddings[:, -1, :] = self._decoder_dropout(curr_embeddings[:, -1, :])

            causal_mask = (
                torch.tril(torch.ones(step + 1, step + 1)).bool().to(DEVICE)
            )  # (dec_seq_len, dec_seq_len)

            decoder_output = self._decoder(
                tgt=tgt_embeddings,
                memory=encoder_embeddings,
                tgt_mask=~causal_mask,
                memory_key_padding_mask=~encoder_mask,
            )

            # TODOPK ASK it is not true?
            # assert that prelast items don't change
            # assert decoder changes only last index in dim = 1

            next_token_embedding = decoder_output[
                                   :, -1, :
                                   ]  # batch_size x embedding_dim

            assert step < len(self._codebook_sizes) + 1

            codebook = self._codebook_item_embeddings_stacked[
                step
            ]  # codebook_size x embedding_dim
            closest_semantic_ids = torch.argmax(
                torch.einsum("bd,cd->bc", next_token_embedding, codebook), dim=1
            )  # batch_size
            semantic_ids = torch.cat(
                [semantic_ids, closest_semantic_ids.unsqueeze(1)], dim=1
            )  # batch_size x (step + 1)
            next_token_embedding = codebook[
                closest_semantic_ids
            ]  # batch_size x embedding_dim

            tgt_embeddings = torch.cat(
                [tgt_embeddings, next_token_embedding.unsqueeze(1)], dim=1
            )
            # TODO спросить Петю про self.decoder_test
            if self.decoder_test is None or self.decoder_test.shape[1] > len(self._codebook_sizes) + 1:
                self.decoder_test = tgt_embeddings
            else:
                assert torch.allclose(self.decoder_test, tgt_embeddings[:, :-1, :], rtol=0, atol=1)
                self.decoder_test = tgt_embeddings
        # now = time.time()
        # print(f"-----regres: {(now - regres_time) * 1000:.2f} ms")
        return semantic_ids, tgt_embeddings

    def assert_item_range(self, events):
        return torch.all((1 <= events) & (events <= self._item_id_to_semantic_id.shape[0])).item()

    def get_item_embeddings(self, events):
        assert len(events.shape) == 1
        assert self.assert_item_range(events)
        sids = self._item_id_to_semantic_id[events - 1]
        assert sids.shape == (events.shape[0], len(self._codebook_sizes) + 1)
        result = self._get_embeddings(sids).reshape((len(self._codebook_sizes) + 1) * events.shape[0],
                                                    self._embedding_dim)  # (4 * ..., embedding_dim)
        assert result.shape[0] == 4 * events.shape[0]
        assert result.shape[1] == self._embedding_dim
        return result

    def calculate_full(self, sem_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embeddings(sem_ids).sum(1)  # (n, embedding_dim)

    def _get_embeddings(self, sem_ids: torch.Tensor) -> torch.Tensor:
        """
        :param sem_ids: набор из sem ids (n, len(codebook_sizes) + 1)
        :return: эмбеддинг для каждого sem_id из набора (n, len(self._codebook_sizes) + 1, embedding_dim)
        """
        assert sem_ids.shape[1] == len(self._codebook_sizes) + 1

        stacked_codebooks = self._codebook_item_embeddings_stacked.reshape(
            (len(self._codebook_sizes) + 1) * self._codebook_sizes[0], self._embedding_dim)

        offsets = torch.tensor([0, 256, 512, 768], dtype=torch.long, device=DEVICE, requires_grad=False)
        sem_ids_with_offsets = sem_ids + offsets.unsqueeze(0)

        return stacked_codebooks[sem_ids_with_offsets]

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

        assert (positions >= 0).all()
        assert (positions < embedding_layer.num_embeddings).all()

        position_embeddings = embedding_layer(
            positions
        )  # (all_batch_events, embedding_dim)

        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        return position_embeddings
