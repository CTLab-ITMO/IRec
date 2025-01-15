import json
import pickle

import torch
from models.collision_solver import CollisionSolver
from models.base import SequentialTorchModel
from torch import nn
from utils import DEVICE, create_masked_tensor, get_activation_function

from .rqvae import RqVaeModel


class TigerModel(SequentialTorchModel, config_name="tiger"):

    def __init__(
        self,
        trie,
        rqvae_model,
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
        semantic_id_arr,
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
        self._decoder_position_embeddings = nn.Embedding(
            num_embeddings=2,  # bos token + label item id (always 0)
            embedding_dim=embedding_dim,
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
        self._decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_decoder_layers
        )
        
        self._codebook_item_embeddings = torch.cat([codebook for codebook in rqvae_model.codebooks], dim=0)

        self._trie = trie
        self._rqvae_model = rqvae_model
        self._collision_solver = CollisionSolver(embedding_dim, len(semantic_id_arr), DEVICE)

        # TODO
        # self._collision_solver.create_query_candidates_dict(semantic_ids[:, :-1], residuals)
        # self._collision_solver.get_semantic_ids(query_prefixes, query_residuals)

        self._projection = nn.Linear(embedding_dim, semantic_id_arr[0])

        self._sequence_prefix = sequence_prefix
        self._pred_prefix = pred_prefix
        self._positive_prefix = positive_prefix
        self._labels_prefix = labels_prefix

        self._semantic_id_arr = semantic_id_arr

        self._bos_token_id = semantic_id_arr[0]
        self._bos_weight = nn.Parameter(torch.randn(embedding_dim))

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(semantic_id_arr) + 1, embedding_dim=embedding_dim
        )  # + 1 for bos token

        self._decoder_layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._decoder_dropout = nn.Dropout(dropout)

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        with open(config["trie"], "rb") as f:
            trie = pickle.load(f)

        rqvae_config = json.load(open(config["rqvae_train_config_path"]))
        semantic_id_arr = rqvae_config["model"]["codebook_sizes"]
        assert all([book_size == semantic_id_arr[0] for book_size in semantic_id_arr])

        rqvae_config["model"]["should_init_codebooks"] = False
        rqvae_model = RqVaeModel.create_from_config(rqvae_config["model"]).to(DEVICE)
        rqvae_model.load_state_dict(
            torch.load(config["rqvae_checkpoint_path"], weights_only=True)
        )
        rqvae_model.eval()

        embedding_dim = rqvae_model.encoder.weight.shape[0] # inner rqvae dim

        return cls(
            trie=trie,
            rqvae_model=rqvae_model,
            sequence_prefix=config["sequence_prefix"],
            pred_prefix=config["predictions_prefix"],
            positive_prefix=config["positive_prefix"],
            labels_prefix=config["labels_prefix"],
            num_items=semantic_id_arr[0],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=embedding_dim,
            num_heads=config.get("num_heads", int(embedding_dim // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * embedding_dim),
            semantic_id_arr=semantic_id_arr,
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

    def get_item_embeddings(self, events):
        bos_mask = (events == self._bos_token_id)
        
        codebook_events = events[~bos_mask]
        positions = torch.arange(len(codebook_events), device=events.device)
        codebook_positions = positions % len(self._semantic_id_arr)
        emb_indices = codebook_positions * self._semantic_id_arr[0] + codebook_events
        
        embeddings = torch.zeros((len(events), self._bos_weight.shape[0]), device=events.device)
        embeddings[bos_mask] = self._bos_weight
        embeddings[~bos_mask] = self._codebook_item_embeddings[emb_indices]
                
        return embeddings

    def get_logits(self, inputs, prefix, all_sample_events, all_sample_lengths):
        encoder_embeddings, encoder_mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, enc_seq_len, embedding_dim), (batch_size, enc_seq_len)

        label_events = inputs["semantic.{}.ids".format(prefix)]
        label_lengths = inputs["semantic.{}.length".format(prefix)]

        decoder_outputs = self._apply_decoder(
            label_events, label_lengths, encoder_embeddings, encoder_mask
        )  # (batch_size, label_len, embedding_dim)

        # TODOPK correct place for projection? or view -> projection
        logits = self._projection(
            decoder_outputs
        )  # (batch_size, dec_seq_len, _semantic_id_arr[0])

        return logits

    def forward(self, inputs):
        all_sample_events = inputs[
            "semantic.{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "semantic.{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        if self.training:
            logits = self.get_logits(
                inputs, self._positive_prefix, all_sample_events, all_sample_lengths
            )  # (batch_size, dec_seq_len, _semantic_id_arr[0])

            logits = logits.view(-1, self._semantic_id_arr[0])

            return {self._pred_prefix: logits}
        else:
            logits = self.get_logits(
                inputs, self._labels_prefix, all_sample_events, all_sample_lengths
            )  # batch_size, dec_seq_len, emb_dim (_semantic_id_arr[0])

            preds = logits.argmax(dim=-1)  # (batch_size, dec_seq_len)
            ids = torch.tensor(self._apply_trie(preds))
            return ids
        

    def _apply_trie(self, preds):  # TODOPK make this faster (how?)
        native_repr = [tuple(row.tolist()) for row in preds]
        # TODOPK add residual
        # add flag if item taken (take other items in up level)
        ids = []
        for semantic_id in native_repr:
            cur_result = set()
            prefixes = [semantic_id[:i] for i in range(len(semantic_id), 0, -1)]
            for prefix in prefixes:
                prefix_ids = self._trie.search_prefix(
                    prefix
                )  # todo handle collisions (not overwrite)
                for id in prefix_ids:
                    cur_result.add(id)
                    if len(cur_result) >= 20:
                        break
                if len(cur_result) >= 20:
                    break

            cur_result = list(cur_result)
            while len(cur_result) < 20:
                cur_result.append(0)  # solve empty event if shortest prefix
            ids.append(cur_result)
        return ids

    def _prepend_bos(self, label_events, label_lengths):
        batch_size = len(label_lengths)
        label_events = label_events.view(batch_size, -1)
        bos_tokens = torch.full(
            (batch_size, 1), self._bos_token_id, device=label_events.device
        )
        label_events = torch.cat(
            [bos_tokens, label_events], dim=1
        )  # (batch_size, dec_seq_len + 1)
        label_lengths = label_lengths + 1
        return label_events.view(-1), label_lengths

    def _apply_decoder(
        self, label_events, label_lengths, encoder_embeddings, encoder_mask
    ):
        label_events, label_lengths = self._prepend_bos(label_events, label_lengths)

        tgt_embeddings = self.get_item_embeddings(
            label_events
        )  # (all_batch_events, embedding_dim)

        tgt_embeddings, tgt_mask = create_masked_tensor(
            data=tgt_embeddings, lengths=label_lengths
        )  # (batch_size, dec_seq_len + 1, embedding_dim), (batch_size, dec_seq_len + 1)

        label_len = tgt_mask.shape[1]

        assert label_len == len(self._semantic_id_arr) + 1  # TODOPK

        position_embeddings = self._decoder_pos_embeddings(label_lengths, tgt_mask)
        assert torch.allclose(position_embeddings[~tgt_mask], tgt_embeddings[~tgt_mask])

        tgt_embeddings = tgt_embeddings + position_embeddings

        tgt_embeddings = self._decoder_layernorm(
            tgt_embeddings
        )  # (batch_size, dec_seq_len + 1, embedding_dim)
        tgt_embeddings = self._decoder_dropout(
            tgt_embeddings
        )  # (batch_size, dec_seq_len + 1, embedding_dim)

        tgt_embeddings[~tgt_mask] = 0

        causal_mask = (
            torch.tril(torch.ones(label_len, label_len)).bool().to(DEVICE)
        )  # (dec_seq_len + 1, dec_seq_len + 1)

        decoder_outputs = self._decoder(
            tgt=tgt_embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
            tgt_key_padding_mask=~tgt_mask,
        )  # (batch_size, dec_seq_len + 1, embedding_dim)

        decoder_outputs = decoder_outputs[:, 1:, :]  # remove bos token

        return decoder_outputs

    def _decoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // len(self._semantic_id_arr)  #  1 0 0 0 1 0 0 0 ...

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._decoder_position_embeddings
        )

        def codebook_lambda(x):
            non_bos = x < len(self._semantic_id_arr)
            x[non_bos] = (len(self._semantic_id_arr) - 1) - x[non_bos]
            return x  # 3, 0, 1, 2, 3, 0, 1, 2 ...

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return position_embeddings + codebook_embeddings

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // len(self._semantic_id_arr)  # 5 5 5 4 4 4 3 3 3 ...

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            return (len(self._semantic_id_arr) - 1) - x % len(
                self._semantic_id_arr
            )  # 0 1 2 3 0 1 2 3 ...

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        return position_embeddings + codebook_embeddings

    def _get_position_embeddings(self, lengths, mask, position_lambda, embedding_layer):
        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None]
            .tile([batch_size, 1])
            .long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)

        positions = position_lambda(positions)  # (all_batch_events)

        position_embeddings = embedding_layer(
            positions
        )  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        return position_embeddings
