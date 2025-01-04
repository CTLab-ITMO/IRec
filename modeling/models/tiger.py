import json
import pickle

import torch
from models.base import SequentialTorchModel
from torch import nn
from utils import DEVICE, create_masked_tensor, get_activation_function


class TigerModel(SequentialTorchModel, config_name="tiger"):

    def __init__(
        self,
        trie,
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
            num_embeddings=max_sequence_length
            + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim,
        )
        self._decoder_codebook_embeddings = nn.Embedding(
            num_embeddings=len(semantic_id_arr), embedding_dim=embedding_dim
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
        self._trie = trie

        self._projection = nn.Linear(embedding_dim, semantic_id_arr[0])

        self._sequence_prefix = sequence_prefix
        self._pred_prefix = pred_prefix
        self._positive_prefix = positive_prefix
        self._labels_prefix = labels_prefix

        self._semantic_id_arr = semantic_id_arr

        self._bos_embedding = nn.Embedding(1, embedding_dim)

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(semantic_id_arr), embedding_dim=embedding_dim
        )

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

        return cls(
            trie=trie,
            sequence_prefix=config["sequence_prefix"],
            pred_prefix=config["predictions_prefix"],
            positive_prefix=config["positive_prefix"],
            labels_prefix=config["labels_prefix"],
            num_items=semantic_id_arr[0],
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=config["embedding_dim"],
            num_heads=config.get("num_heads", int(config["embedding_dim"] // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * config["embedding_dim"]),
            semantic_id_arr=semantic_id_arr,
            dropout=config.get("dropout", 0.0),
            initializer_range=config.get("initializer_range", 0.02),
        )

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
            ) # (batch_size, dec_seq_len, _semantic_id_arr[0])

            logits = logits.view(-1, self._semantic_id_arr[0])
            # TODOPK check if correct flattening

            return {self._pred_prefix: logits}
        else:
            logits = self.get_logits(
                inputs, self._labels_prefix, all_sample_events, all_sample_lengths
            )

            preds = logits.argmax(dim=-1)  # (batch_size, dec_seq_len)
            ids = torch.tensor(self._apply_trie(preds))
            return ids

    def _apply_trie(self, preds):  # TODOPK make this faster (how?)
        native_repr = [tuple(row.tolist()) for row in preds]
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

    def _apply_decoder(
        self, label_events, label_lengths, encoder_embeddings, encoder_mask
    ):
        matrix_label_events = label_events.view(-1, label_lengths[0])
        matrix_label_events = torch.cat(
            [torch.full((len(label_lengths), 1), 256), matrix_label_events], dim=1
        )
        # TODOPK 256 hardcoded

        label_events = matrix_label_events.view(-1)
        
        label_lengths = label_lengths + 1

        tgt_embeddings = self._item_embeddings(
            label_events
        )  # (all_batch_events, embedding_dim)
        # TODOPK share same embeddings with encoder

        tgt_embeddings, tgt_mask = create_masked_tensor(
            data=tgt_embeddings, lengths=label_lengths
        )  # (batch_size, label_len, embedding_dim), (batch_size, label_len)

        label_len = tgt_mask.shape[1]

        assert label_len == len(self._semantic_id_arr) + 1 # TODOPK

        position_embeddings = self._decoder_pos_embeddings(label_lengths, tgt_mask)
        assert torch.allclose(position_embeddings[~tgt_mask], tgt_embeddings[~tgt_mask])

        tgt_embeddings = tgt_embeddings + position_embeddings

        tgt_embeddings = self._decoder_layernorm(
            tgt_embeddings
        )  # (batch_size, seq_len, embedding_dim)
        tgt_embeddings = self._decoder_dropout(
            tgt_embeddings
        )  # (batch_size, seq_len, embedding_dim)

        tgt_embeddings[~tgt_mask] = 0

        causal_mask = (
            torch.tril(torch.ones(label_len, label_len)).bool().to(DEVICE)
        )  # (seq_len, seq_len)
        # TODOPK -inf?

        decoder_outputs = self._decoder(
            tgt=tgt_embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
            tgt_key_padding_mask=~tgt_mask,
        )  # (batch_size, label_len, embedding_dim)
        
        decoder_outputs = decoder_outputs[:, 1:, :]  # TODOPK remove bos token

        return decoder_outputs

    def _decoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):  # TODOPK share layers with encoder & fix
            return x // len(self._semantic_id_arr)  # 5 5 5 4 4 4 3 3 3 ...

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._decoder_position_embeddings
        )

        def codebook_lambda(x):
            return x % len(self._semantic_id_arr)  # 2 1 0 2 1 0 ...

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._decoder_codebook_embeddings
        )

        # TODOPK fix codebook indexing

        return position_embeddings + codebook_embeddings

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // len(self._semantic_id_arr)  # 5 5 5 4 4 4 3 3 3 ...

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )

        def codebook_lambda(x):
            return x % len(self._semantic_id_arr)  # 2 1 0 2 1 0 ...

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
