import json

import torch
from torch import nn

from models.base import SequentialTorchModel
from rqvae_utils import CollisionSolver, SimplifiedTree
from utils import DEVICE, create_masked_tensor, get_activation_function

from .rqvae import RqVaeModel


class TigerModel(SequentialTorchModel, config_name="tiger"):
    def __init__(
        self,
        rqvae_model,
        item_id_to_semantic_id,
        item_id_to_residual,
        solver,
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
            is_causal=True,
        )

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

        self._solver: CollisionSolver = solver

        self._codebook_sizes = rqvae_model.codebook_sizes
        self._bos_token_id = self._codebook_sizes[0]
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
            num_embeddings=len(self._codebook_sizes) + 2, embedding_dim=embedding_dim
        )  # + 2 for bos token & residual

        self._init_weights(initializer_range)

        self._codebook_item_embeddings_stacked = nn.Parameter(
            torch.stack([codebook for codebook in rqvae_model.codebooks]),
            requires_grad=True,
        )

        self._item_id_to_semantic_embedding = nn.Parameter(
            self.get_init_item_embeddings(item_id_to_semantic_id, item_id_to_residual),
            requires_grad=True,
        )

        self._trie = SimplifiedTree(self._codebook_item_embeddings_stacked)

        self._trie.build_tree_structure(
            item_id_to_semantic_id.to(DEVICE),
            item_id_to_residual.to(DEVICE),
            torch.arange(1, len(item_id_to_semantic_id) + 1).to(DEVICE),
            sum_with_residuals=False,
        )

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
        assert item_ids == list(range(1, len(item_ids) + 1))

        text_embeddings = torch.stack(embs_extractor["embeddings"].tolist()).to(DEVICE)

        semantic_ids, residuals = rqvae_model({"embeddings": text_embeddings})

        return rqvae_model, semantic_ids, residuals, item_ids

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model, semantic_ids, residuals, item_ids = cls.init_rqvae(config)

        solver = CollisionSolver(
            emb_dim=residuals.shape[1],
            sem_id_len=len(rqvae_model.codebook_sizes),
            codebook_size=rqvae_model.codebook_sizes[0],
        )
        solver.create_query_candidates_dict(
            torch.tensor(item_ids), semantic_ids, residuals
        )

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=semantic_ids,
            item_id_to_residual=residuals,
            solver=solver,
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
        all_sample_events = inputs[
            "{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        all_sample_lengths = all_sample_lengths * (len(self._codebook_sizes) + 1)
        encoder_embeddings, encoder_mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, enc_seq_len, embedding_dim), (batch_size, enc_seq_len)

        if self.training:
            label_events = inputs["{}.ids".format(self._positive_prefix)]
            label_lengths = inputs["{}.length".format(self._positive_prefix)]

            tgt_embeddings = self.get_item_embeddings(
                label_events
            )  # (all_batch_events, embedding_dim)

            decoder_outputs = self._apply_decoder(
                tgt_embeddings,
                label_lengths * (len(self._codebook_sizes) + 1),
                encoder_embeddings,
                encoder_mask,
            )  # (batch_size, label_len, embedding_dim)

            decoder_prefix_scores = torch.einsum(
                "bsd,scd->bsc",
                decoder_outputs[:, :-1, :],
                self._codebook_item_embeddings_stacked,
            )

            decoder_output_residual = decoder_outputs[:, -1, :]

            semantic_ids = self._item_id_to_semantic_id[
                label_events - 1
            ]  # len(events), len(codebook_sizes)
            true_residuals = self._item_id_to_residual[label_events - 1]

            true_info = self._solver.get_true_dedup_tokens(semantic_ids, true_residuals)
            pred_info = self._solver.get_pred_scores(
                semantic_ids, decoder_output_residual
            )

            return {
                "logits": decoder_prefix_scores.reshape(
                    -1, decoder_prefix_scores.shape[2]
                ),
                "semantic.labels.ids": semantic_ids.reshape(-1),
                "dedup.logits": pred_info["pred_scores"],
                "dedup.labels.ids": true_info["true_dedup_tokens"],
            }
        else:
            semantic_ids, tgt_embeddings = self._apply_decoder_autoregressive(
                encoder_embeddings, encoder_mask
            )  # (batch_size, len(self._codebook_sizes) (bos, residual)), (batch_size, len(self._codebook_sizes) + 2 (bos, residual), embedding_dim)

            # 1 4 6 -> lookup -> sum = emb (last embedding) # bs, embedding_dim
            # take all embedings (from stacked) # all_items, embedding_dim
            # take from sasrec eval (indices + 1)
            # guarantee that all items are in correct order

            residuals = tgt_embeddings[:, -1, :]
            semantic_ids = semantic_ids.to(torch.int64)

            item_ids = self._trie.query(semantic_ids, items_to_query=20)

            return item_ids

        # TODOPK (decompose tree)
        # else:  # eval mode
        #     last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
        #     # b - batch_size, n - num_candidates, d - embedding_dim
        #     candidate_scores = torch.einsum(
        #         'bd,nd->bn',
        #         last_embeddings,
        #         self._item_embeddings.weight
        #     )  # (batch_size, num_items + 2)

        #     _, indices = torch.topk(
        #         candidate_scores,
        #         k=20, dim=-1, largest=True
        #     )  # (batch_size, 20)

        #     return indices + 1

    def _apply_decoder(
        self, tgt_embeddings, label_lengths, encoder_embeddings, encoder_mask
    ):
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

        # TODOPK remove layernorm & dropout (for inference)
        # tgt_embeddings = self._decoder_layernorm(
        #     tgt_embeddings
        # )  # (batch_size, dec_seq_len, embedding_dim)
        # tgt_embeddings = self._decoder_dropout(
        #     tgt_embeddings
        # )  # (batch_size, dec_seq_len, embedding_dim)

        tgt_embeddings[~tgt_mask] = 0

        causal_mask = (
            torch.tril(torch.ones(label_len, label_len)).bool().to(DEVICE)
        )  # (dec_seq_len, dec_seq_len)

        decoder_outputs = self._decoder(
            tgt=tgt_embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
        )  # (batch_size, dec_seq_len, embedding_dim)

        return decoder_outputs

    def _apply_decoder_autoregressive(self, encoder_embeddings, encoder_mask):
        batch_size = encoder_embeddings.shape[0]
        embedding_dim = encoder_embeddings.shape[2]

        tgt_embeddings = (
            self._bos_weight.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 1, embedding_dim)
        )

        semantic_ids = torch.tensor([], device=DEVICE)

        for step in range(len(self._codebook_sizes) + 1):  # semantic_id_seq + residual
            index = len(self._codebook_sizes) if step == 0 else step - 1

            last_position_embedding = self._codebook_embeddings(
                torch.full((batch_size,), index, device=DEVICE)
            )

            assert last_position_embedding.shape == tgt_embeddings[:, -1, :].shape
            assert tgt_embeddings.shape == torch.Size(
                [batch_size, step + 1, embedding_dim]
            )

            curr_step_embeddings = tgt_embeddings.clone()
            curr_step_embeddings[:, -1, :] = (
                tgt_embeddings[:, -1, :] + last_position_embedding
            )
            assert torch.allclose(
                tgt_embeddings[:, :-1, :], curr_step_embeddings[:, :-1, :]
            )
            tgt_embeddings = curr_step_embeddings

            # curr_embeddings[:, -1, :] = self._decoder_layernorm(curr_embeddings[:, -1, :])
            # curr_embeddings[:, -1, :] = self._decoder_dropout(curr_embeddings[:, -1, :])

            decoder_output = self._decoder(
                tgt=tgt_embeddings,
                memory=encoder_embeddings,
                memory_key_padding_mask=~encoder_mask,
            )

            # TODOPK ASK it is not true?
            # assert that prelast items don't change
            # assert decoder changes only last index in dim = 1

            next_token_embedding = decoder_output[
                :, -1, :
            ]  # batch_size x embedding_dim

            if step < len(self._codebook_sizes):
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

        return semantic_ids, tgt_embeddings
 
    def get_item_embeddings(self, events):
        embs = self._item_id_to_semantic_embedding[
            events - 1
        ]  # len(events), len(self._codebook_sizes) + 1, embedding_dim
        return embs.reshape(-1, self._embedding_dim)

    def get_init_item_embeddings(self, item_id_to_semantic_id, item_id_to_residual):
        result = []
        for semantic_id in item_id_to_semantic_id:
            item_repr = []
            for codebook_idx, codebook_id in enumerate(semantic_id):
                item_repr.append(
                    self._codebook_item_embeddings_stacked[codebook_idx][codebook_id]
                )
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

    def _decoder_pos_embeddings(self, lengths, mask):
        def codebook_lambda(x):
            non_bos = x < len(self._codebook_sizes)
            x[non_bos] = (len(self._codebook_sizes) - 1) - x[non_bos]
            return x  # 3, 0, 1, 2, 3, 0, 1, 2 ... len(self._codebook_sizes) = 3 for bos

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return codebook_embeddings

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
