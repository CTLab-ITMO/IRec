import json
from turtle import pos
from utils import DEVICE, create_masked_tensor, get_activation_function
from models.base import SequentialTorchModel
import pickle
import torch
from torch import nn


class TigerModel(SequentialTorchModel, config_name="tiger_new"):

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

        self._trie = trie

        self._output_projection = nn.Linear(embedding_dim, semantic_id_arr[0])

        self._sequence_prefix = sequence_prefix
        self._pred_prefix = pred_prefix
        self._positive_prefix = positive_prefix
        self._labels_prefix = labels_prefix

        self._semantic_id_arr = semantic_id_arr

        self._level_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_items + 2, embedding_dim)
                for _ in range(len(semantic_id_arr))
            ]
        )

        self._positional_embeddings = nn.Embedding(max_sequence_length, embedding_dim)

        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(semantic_id_arr), embedding_dim=embedding_dim
        )

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

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

    def get_logits(self, inputs, prefix, flattened_events, lengths):
        src_embeddings, src_mask = self.get_embeddings(
            flattened_events, lengths
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)

        tgt_flattened_events = inputs[
            "semantic.{}.ids".format(prefix)
        ]  # (all_batch_events)
        tgt_lengths = inputs["semantic.{}.length".format(prefix)]  # (batch_size)

        tgt_embeddings, tgt_mask = self.get_embeddings(
            tgt_flattened_events, tgt_lengths
        )  # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)

        transformer_output = self.transformer(
            src_embeddings,
            tgt_embeddings,
            src_key_padding_mask=~src_mask,
            tgt_key_padding_mask=~tgt_mask,
        ) # (batch_size, seq_len, embedding_dim)

        logits = self._output_projection(transformer_output)

        return logits, tgt_mask

    def forward(self, inputs):
        all_sample_events = inputs[
            "semantic.{}.ids".format(self._sequence_prefix)
        ]  # (all_batch_events)
        all_sample_lengths = inputs[
            "semantic.{}.length".format(self._sequence_prefix)
        ]  # (batch_size)

        if self.training:
            logits, tgt_mask = self.get_logits(
                inputs, self._positive_prefix, all_sample_events, all_sample_lengths
            ) # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)

            logits = logits[tgt_mask]

            return {self._pred_prefix: logits}
        else:
            logits, _ = self.get_logits(
                inputs, self._labels_prefix, all_sample_events, all_sample_lengths
            ) # (batch_size, seq_len, embedding_dim) (batch_size, seq_len)
            preds = logits.argmax(dim=-1)
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

    def get_embeddings(self, flattened_events, lengths):
        num_levels = len(self._semantic_id_arr)

        # Heirarchical embeddings
        level_indices = torch.arange(len(flattened_events), device=DEVICE) % num_levels
        item_embeddings = torch.zeros(
            (len(flattened_events), self._embedding_dim), device=DEVICE
        )
        for level in range(num_levels):
            level_mask = level_indices == level
            item_embeddings[level_mask] = self._level_embeddings[level](
                flattened_events[level_mask]
            )

        item_embeddings, mask = create_masked_tensor(
            data=item_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        # Positional embeddings
        positions = torch.arange(seq_len, device=DEVICE).repeat(batch_size, 1)
        positions = positions.masked_fill(~mask, 0)  # (batch_size, max_len)
        pos_embeds = self._positional_embeddings(
            positions // num_levels
        )  # (batch_size, seq_len, embedding_dim)

        # Codebook embeddings
        codebook_indices = (
            torch.arange(seq_len, device=DEVICE).repeat(batch_size, 1) % num_levels
        )
        codebook_indices = codebook_indices.masked_fill(
            ~mask, 0
        )  # (batch_size, seq_len)
        hierarchy_embeds = self._codebook_embeddings(
            codebook_indices
        )  # (batch_size, seq_len, embedding_dim)

        return item_embeddings + pos_embeds + hierarchy_embeds, mask
