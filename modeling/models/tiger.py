import json

import torch
from tqdm import tqdm
from models.base import SequentialTorchModel
from rqvae_utils import CollisionSolver, Trie, Item
from torch import nn
from utils import DEVICE, create_masked_tensor, get_activation_function

from .rqvae import RqVaeModel


class TigerModel(SequentialTorchModel, config_name="tiger"):

    def __init__(
        self,
        rqvae_model,
        item_id_to_semantic_id,
        item_id_to_residual,
        item_id_to_embedding,
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
        codebook_sizes,
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
        
        self._solver = solver
        
        self._codebook_sizes = codebook_sizes
        self._codebook_item_embeddings_flattened = torch.cat([codebook for codebook in rqvae_model.codebooks], dim=0)
        self._codebook_item_embeddings_stacked = torch.stack([codebook for codebook in rqvae_model.codebooks])
        
        self._item_id_to_semantic_id = item_id_to_semantic_id
        self._item_id_to_residual = item_id_to_residual
        self._item_id_to_embedding = item_id_to_embedding
        
        self._trie = Trie()
        for item_id, semantic_id in item_id_to_semantic_id.items():
            self._trie.insert(Item(semantic_id, item_id_to_residual[item_id])) # TODO no dedup tokens here

        self._bos_token_id = codebook_sizes[0]
        self._bos_weight = nn.Parameter(torch.randn(embedding_dim))
        
        self._decoder_position_embeddings = nn.Embedding(
            num_embeddings=2,  # bos token + label item id (always 0)
            embedding_dim=embedding_dim,
        )
        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(codebook_sizes) + 2, embedding_dim=embedding_dim # TODO
        )  # + 2 for bos token & residual

        self._init_weights(initializer_range)
        
    @classmethod
    def init_rqvae(cls, config) -> RqVaeModel:
        rqvae_config = json.load(open(config["rqvae_train_config_path"]))
        rqvae_config["model"]["should_init_codebooks"] = False
        
        rqvae_model = RqVaeModel.create_from_config(rqvae_config['model']).to(DEVICE)
        rqvae_model.load_state_dict(torch.load(config['rqvae_checkpoint_path'], weights_only=True))
        rqvae_model.eval()
        for param in rqvae_model.parameters(): # TODO
            param.requires_grad = False
        
        codebook_sizes = rqvae_model.codebook_sizes
        assert all([book_size == codebook_sizes[0] for book_size in codebook_sizes])
        
        return rqvae_model

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_model = cls.init_rqvae(config)
        embedding_dim = rqvae_model.encoder.weight.shape[0] # TODO
        embs_extractor = torch.load(config['embs_extractor_path'])
        
        item_ids = embs_extractor.index.tolist()
        assert sorted(item_ids) == list(range(1, len(item_ids) + 1))
        
        embeddings = torch.stack(embs_extractor['embeddings'].tolist()).to(DEVICE)

        semantic_ids, residuals = rqvae_model({"embeddings": embeddings})
        assert embedding_dim == residuals.shape[1]
        
        solver = CollisionSolver(
            residual_dim=residuals.shape[1], 
            emb_dim=len(rqvae_model.codebook_sizes), 
            codebook_size=len(rqvae_model.codebook_sizes[0]), 
            device=DEVICE
        )
        solver.create_query_candidates_dict(semantic_ids, residuals)
        
        item_id_to_embedding = {
            item_id: embedding for (item_id, embedding) in zip(item_ids, embeddings)
        }
        item_id_to_semantic_id = {
            item_id: torch.tensor(semantic_id, device=DEVICE) for (item_id, semantic_id) in zip(item_ids, semantic_ids)
        }
        item_id_to_residual = {
            item_id: remainder for (item_id, remainder) in zip(item_ids, residuals)
        }

        return cls(
            rqvae_model=rqvae_model,
            item_id_to_semantic_id=item_id_to_semantic_id,
            item_id_to_residual=item_id_to_residual,
            item_id_to_embedding=item_id_to_embedding,
            solver=solver,
            sequence_prefix=config["sequence_prefix"],
            pred_prefix=config["predictions_prefix"],
            positive_prefix=config["positive_prefix"],
            labels_prefix=config["labels_prefix"],
            num_items=rqvae_model.codebook_sizes[0], # unused
            max_sequence_length=kwargs["max_sequence_length"],
            embedding_dim=embedding_dim,
            num_heads=config.get("num_heads", int(embedding_dim // 64)),
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config.get("dim_feedforward", 4 * embedding_dim),
            codebook_sizes=rqvae_model.codebook_sizes,
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

        if self.training:
            label_events = inputs["{}.ids".format(self._positive_prefix)]
            label_lengths = inputs["{}.length".format(self._positive_prefix)]
        
            decoder_prefix_scores = self.get_logits(
                label_events, label_lengths, all_sample_events, all_sample_lengths
            )  # (batch_size, dec_seq_len, _codebook_sizes[0])
            
            logits = decoder_prefix_scores.reshape(-1, self._codebook_sizes[0]) # (batch_size * dec_seq_len, _codebook_sizes[0])
            
            semantic_ids = torch.stack([self._item_id_to_semantic_id[event] for event in label_events.tolist()]) # len(events), len(codebook_sizes)
            semantic_events = semantic_ids.view(-1)
            
            return {self._pred_prefix: logits, f"semantic.{self._labels_prefix}.ids": semantic_events}
        else:
            label_events = inputs["{}.ids".format(self._labels_prefix)]
            label_lengths = inputs["{}.length".format(self._labels_prefix)]
            
            decoder_prefix_scores = self.get_logits(
                label_events, label_lengths, all_sample_events, all_sample_lengths
            )  # batch_size, dec_seq_len, emb_dim (_codebook_sizes[0])

            preds = decoder_prefix_scores.argmax(dim=-1)  # (batch_size, dec_seq_len)
            ids = self._apply_trie(preds)
            return ids

    def _apply_trie(self, preds):
        semantic_ids = [tuple(row.tolist()) for row in preds]
        
        ids = []
        for semantic_id in tqdm(semantic_ids):
            item = Item(semantic_id, torch.rand(self._embedding_dim).to(DEVICE)) # TODO use true residuals
            closest_items = self._trie.find_closest(item, n=20)
            ids.append(closest_items)
        
        return torch.tensor(ids)
    
    def get_logits(self, label_events, label_lengths, all_sample_events, all_sample_lengths):
        all_sample_lengths = all_sample_lengths * (len(self._codebook_sizes) + 1) # TODO residual gets add as event
        # TODO encoder_mask sees residual as event (is it correct?)
        
        encoder_embeddings, encoder_mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, enc_seq_len, embedding_dim), (batch_size, enc_seq_len)
        
        decoder_outputs = self._apply_decoder(
            label_events, label_lengths, encoder_embeddings, encoder_mask
        )  # (batch_size, label_len, embedding_dim)
        
        decoder_prefix_scores = torch.einsum("bsd,scd->bsc", decoder_outputs[:, :-1, :], self._codebook_item_embeddings_stacked)
        
        return decoder_prefix_scores

    def _apply_decoder(
        self, label_events, label_lengths, encoder_embeddings, encoder_mask
    ):
        label_lengths = label_lengths * (len(self._codebook_sizes) + 1) # TODO bos prepending
        tgt_embeddings = self.get_item_embeddings( # TODO residual embs
            label_events
        )  # (all_batch_events, embedding_dim)
        
        tgt_embeddings, tgt_mask = create_masked_tensor(
            data=tgt_embeddings, lengths=label_lengths
        )  # (batch_size, dec_seq_len, embedding_dim), (batch_size, dec_seq_len)
        
        batch_size = tgt_embeddings.shape[0]
        bos_embeddings = self._bos_weight.unsqueeze(0).expand(batch_size, 1, -1)  # (batch_size, 1, embedding_dim)
        
        tgt_embeddings = torch.cat([bos_embeddings, tgt_embeddings[:, :-1, :]], dim=1) # TODO remove residuals (batch_size, dec_seq_len, embedding_dim)
        
        label_len = tgt_mask.shape[1]

        assert label_len == len(self._codebook_sizes) + 1 # TODO +1 for bos

        position_embeddings = self._decoder_pos_embeddings(label_lengths, tgt_mask)
        assert torch.allclose(position_embeddings[~tgt_mask], tgt_embeddings[~tgt_mask])

        tgt_embeddings = tgt_embeddings + position_embeddings

        tgt_embeddings = self._decoder_layernorm(
            tgt_embeddings
        )  # (batch_size, dec_seq_len, embedding_dim)
        tgt_embeddings = self._decoder_dropout(
            tgt_embeddings
        )  # (batch_size, dec_seq_len, embedding_dim)

        tgt_embeddings[~tgt_mask] = 0

        causal_mask = (
            torch.tril(torch.ones(label_len, label_len)).bool().to(DEVICE)
        )  # (dec_seq_len, dec_seq_len)
        
        decoder_outputs = self._decoder(
            tgt=tgt_embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
            # TODO tgt_key_padding_mask=~tgt_mask,
        )  # (batch_size, dec_seq_len, embedding_dim)

        return decoder_outputs
    
    def get_item_embeddings(self, events):
        events = events.tolist()
        
        # convert to semantic ids
        semantic_ids = torch.stack([self._item_id_to_semantic_id[event] for event in events]) # len(events), len(codebook_sizes)
        semantic_events = semantic_ids.view(-1) # len(codebook_sizes) * len(events)
        
        # convert to rqvae embeddings
        positions = torch.arange(len(semantic_events), device=DEVICE)
        codebook_positions = positions % len(self._codebook_sizes)
        emb_indices = codebook_positions * self._codebook_sizes[0] + semantic_events
        semantic_embeddings = self._codebook_item_embeddings_flattened[emb_indices]
        semantic_embeddings = semantic_embeddings.view(
            len(events), len(self._codebook_sizes), self._embedding_dim
        )
        
        # get residuals
        text_embeddings = torch.stack([self._item_id_to_embedding[event] for event in events])
        residual = text_embeddings - semantic_embeddings.sum(dim=1)
        residual = residual.unsqueeze(1)
        
        # get true item embeddings
        item_embeddings = torch.cat([semantic_embeddings, residual], dim=1) 
        item_embeddings = item_embeddings.view(-1, self._embedding_dim) # (all_batch_events, embedding_dim)
        
        return item_embeddings

    def _encoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // (len(self._codebook_sizes) + 1)  # 5 5 5 4 4 4 3 3 3 ...
        # TODO +1 for residual embedding

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._position_embeddings
        )
        
        # print(f"{position_embeddings.isnan().any()=}") # TODO fix NaN in pos_embs

        def codebook_lambda(x):
            x = len(self._codebook_sizes) - x % (len(self._codebook_sizes) + 1)
            x[x == len(self._codebook_sizes)] = len(self._codebook_sizes) + 1
            # 0 1 2 3 5 0 1 2 3 5 ... # len(self._codebook_sizes) for bos, len(self._codebook_sizes) + 1 for residual
            return x

        codebook_embeddings = self._get_position_embeddings(
            lengths, mask, codebook_lambda, self._codebook_embeddings
        )

        # print(f"{codebook_embeddings.isnan().any()=}") # TODO fix NaN in pos_embs

        return position_embeddings + codebook_embeddings

    def _decoder_pos_embeddings(self, lengths, mask):
        def position_lambda(x):
            return x // len(self._codebook_sizes)  #  1 0 0 0 1 0 0 0 ...

        position_embeddings = self._get_position_embeddings(
            lengths, mask, position_lambda, self._decoder_position_embeddings
        )

        def codebook_lambda(x):
            non_bos = x < len(self._codebook_sizes)
            x[non_bos] = (len(self._codebook_sizes) - 1) - x[non_bos]
            return x  # 3, 0, 1, 2, 3, 0, 1, 2 ...

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

        positions = position_lambda(positions)  # (all_batch_events)
        
        # print(f"{positions.tolist()[:20]=}")
        
        assert (positions >= 0).all() and (positions < embedding_layer.num_embeddings).all()

        position_embeddings = embedding_layer(
            positions
        )  # (all_batch_events, embedding_dim)
        # print(f"{position_embeddings.isnan().any()=}") # TODO without embeddings also NaN
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        return position_embeddings
