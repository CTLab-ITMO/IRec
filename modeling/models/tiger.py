import json
from utils import DEVICE, create_masked_tensor, get_activation_function
from models.base import SequentialTorchModel
import pickle
import torch
from torch import nn


class TigerModel(SequentialTorchModel, config_name='tiger'):

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
            num_layers,
            dim_feedforward,
            semantic_id_arr,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-9,
            initializer_range=0.02
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
            is_causal=True
        )
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers)
        self._trie = trie
        
        self._projection = nn.Linear(embedding_dim, semantic_id_arr[0])
        
        self._sequence_prefix = sequence_prefix
        self._pred_prefix = pred_prefix
        self._positive_prefix = positive_prefix
        self._labels_prefix = labels_prefix
        
        self._semantic_id_arr = semantic_id_arr
        
        self._codebook_embeddings = nn.Embedding(
            num_embeddings=len(semantic_id_arr),  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        with open(config['trie'], 'rb') as f:
            trie = pickle.load(f)
            
        rqvae_config = json.load(open(config['rqvae_train_config_path']))
        semantic_id_arr = rqvae_config['model']['codebook_sizes']
        assert all([book_size == semantic_id_arr[0] for book_size in semantic_id_arr])
    
        return cls(
            trie=trie,
            sequence_prefix=config['sequence_prefix'],
            pred_prefix=config['predictions_prefix'],
            positive_prefix=config['positive_prefix'],
            labels_prefix=config['labels_prefix'],
            num_items=semantic_id_arr[0],
            max_sequence_length=kwargs['max_sequence_length'] * len(semantic_id_arr),
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            semantic_id_arr=semantic_id_arr,
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02)
        )
        
    def get_logits(self, inputs, prefix, all_sample_events, all_sample_lengths):
        # TODOPK pass parameter as args (self._semantic_prefix)
        
        label_events = inputs['semantic.{}.ids'.format(prefix)]
        label_lengths = inputs['semantic.{}.length'.format(prefix)]
    
        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths, add_codebook_embeddings=True
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        
        decoder_outputs = self._apply_decoder(
            label_events, label_lengths, embeddings, mask
        ) # (batch_size, label_len, embedding_dim)
        # todo pk correct place for projection? or view -> projection
        logits = self._projection(decoder_outputs)  # (batch_size, seq_len, _semantic_id_arr[0])
        
        return logits, mask

    def forward(self, inputs):
        all_sample_events = inputs['semantic.{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['semantic.{}.length'.format(self._sequence_prefix)]  # (batch_size)
        
        if self.training:
            logits, mask = self.get_logits(inputs, self._positive_prefix, all_sample_events, all_sample_lengths)
            
            logits = logits[mask] # TODOPK correct?
            
            return {
                self._pred_prefix: logits
            }
        else:
            logits, mask = self.get_logits(inputs, self._labels_prefix, all_sample_events, all_sample_lengths)
            
            preds = logits.argmax(dim=-1).view(len(all_sample_lengths), len(self._semantic_id_arr)) # Shape: (batch_size, seq_len)
            ids = torch.tensor(self._apply_trie(preds))
            return ids
    
    def _apply_trie(self, preds): # TODOPK make this faster (how?)
        native_repr = [tuple(row.tolist()) for row in preds]
        ids = []
        for semantic_id in native_repr:
            cur_result = set()
            prefixes = [
                semantic_id[:i] for i in range(len(semantic_id), 0, -1)
            ]
            for prefix in prefixes:
                prefix_ids = self._trie.search_prefix(prefix) # todo handle collisions (not overwrite)
                for id in prefix_ids:
                    cur_result.add(id)
                    if len(cur_result) >= 20:
                        break
                if len(cur_result) >= 20:
                    break
            
            cur_result = list(cur_result)
            while len(cur_result) < 20:
                cur_result.append(0) # solve empty event if shortest prefix
            ids.append(cur_result)
        return ids
        
    def _apply_decoder(self, label_events, label_lengths, encoder_embeddings, encoder_mask):
        # делаем по аналогии с encoder'ом?
        embeddings = self._item_embeddings(label_events)  # (batch_size * label_len, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings,
            lengths=label_lengths
        )  # (batch_size, label_len, embedding_dim), (batch_size, label_len)
        
        batch_size = mask.shape[0]
        label_len = mask.shape[1]
        
        position_embeddings = self._get_position_embeddings(
            embeddings, label_lengths, mask, batch_size, label_len
        ) # (batch_size, label_len, embedding_dim)
        codebook_embeddings = self._get_codebook_embeddings(
            embeddings, label_lengths, mask, batch_size, label_len
        ) # (batch_size, label_len, embedding_dim)
       
        embeddings = embeddings + codebook_embeddings
        embeddings = embeddings + position_embeddings
        
        embeddings = self._layernorm(embeddings)
        embeddings = self._dropout(embeddings)

        embeddings[~mask] = 0

        causal_mask = torch.tril(
            torch.ones(label_len, label_len)
        ).bool().to(DEVICE)  # (label_len, label_len)

        decoder_outputs = self._decoder(
            tgt=embeddings,
            memory=encoder_embeddings,
            tgt_mask=~causal_mask,
            memory_key_padding_mask=~encoder_mask,
            tgt_key_padding_mask=~mask
        )  # (batch_size, label_len, embedding_dim)
        
        return decoder_outputs
        
    def _get_position_embeddings(self, embeddings, lengths, mask, batch_size, seq_len):
        positions = torch.arange( # TODOPK invert decoder (position.reverse)
            start=seq_len - 1, end=-1, step=-1, device=mask.device
        )[None].tile([batch_size, 1]).long()  # (batch_size, seq_len)
        
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        
        positions = positions // len(self._semantic_id_arr)
        # 5 5 5 5 4 4 4 4 3 3 3 3 ...
        
        position_embeddings = self._position_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])
        
        return position_embeddings
        
    def _get_codebook_embeddings(self, embeddings, lengths, mask, batch_size, seq_len):
        positions = torch.arange( # TODOPK invert decoder (position.reverse)
            start=seq_len - 1, end=-1, step=-1, device=mask.device
        )[None].tile([batch_size, 1]).long()  # (batch_size, seq_len)
        
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (all_batch_events)
        
        positions = positions.flip(-1) % len(self._semantic_id_arr) # (all_batch_events)
        # flip so first item has (0, 1, 2, 3) not (3, 2, 1, 0) semantic_id embeddings
        # 0 1 2 3 0 1 2 3 0 1 2 3
        
        position_embeddings = self._codebook_embeddings(positions)  # (all_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings,
            lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)
        assert torch.allclose(position_embeddings[~mask], embeddings[~mask])
        
        return position_embeddings
