from models.base import SequentialTorchModel, TorchModel

from utils import create_masked_tensor, get_activation_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class GTOModel(SequentialTorchModel, config_name='gtorec'):
    def __init__(
            self,
            # sequential params 
            sequence_prefix, # =item_prefix
            positive_prefix, 
            negative_prefix, 
            candidate_prefix,
            source_domain, 
            num_users,  
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            # graph params
            user_prefix,
            graph,
            graph_embedding_dim,
            graph_num_layers,
            # params with default values
            dropout=0.0,
            graph_dropout=0.0,
            activation='relu', 
            layer_norm_eps=1e-9, 
            initializer_range=0.02,
            norm_first=True
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
        # sequential part 
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix
        self._source_domain = source_domain

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )
        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        # graph part
        self._user_prefix = user_prefix
        self._num_users = num_users
        self._graph = graph
        self._graph_embedding_dim = graph_embedding_dim
        self._graph_num_layers = graph_num_layers
        self._graph_dropout = graph_dropout

        self._graph_user_embeddings = nn.Embedding(
            num_embeddings=num_users + 2,
            embedding_dim=self._graph_embedding_dim
        )
        self._graph_item_embeddings = nn.Embedding(
            num_embeddings=num_items + 2,
            embedding_dim=self._graph_embedding_dim
        )

        # cross_attention part 
        self._mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
        )

        self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embedding_dim)
        self.activation = get_activation_function(activation)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self._mha_output_projection = nn.Linear(
            in_features=2 * embedding_dim,
            out_features=embedding_dim,
        )
        
        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            # sequential part
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02),
            norm_first=config.get('norm_first', True),
            # graph part
            user_prefix=config['user_prefix'],
            num_users=kwargs['num_users'],
            graph_embedding_dim=config["graph_embedding_dim"],
            graph_num_layers=config["graph_num_layers"],
            graph_dropout=config.get("graph_dropout", 0.0)
        )
    
    def _apply_graph_encoder(self):
        ego_embeddings = torch.cat((self._graph_user_embeddings.weight, self._graph_item_embeddings.weight), dim=0)
        all_embeddings = [ego_embeddings]

        if self._graph_dropout > 0:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + (1 - self._graph_dropout)
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / (1 - self._graph_dropout)
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for i in range(self._graph_num_layers):
            ego_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=-1)
        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def _get_graph_embeddings(self, inputs, prefix, ego_embeddings, final_embeddings):
        ids = inputs['{}.ids'.format(prefix)]  # (batch_size)
        lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

        final_embeddings = final_embeddings[ids]  # (batch_size, emb_dim)
        ego_embeddings = ego_embeddings(ids)  # (batch_size, emb_dim)

        padded_embeddings, mask = create_masked_tensor(final_embeddings, lengths)
        padded_ego_embeddings, ego_mask = create_masked_tensor(ego_embeddings, lengths)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask
    
    def _ca_block(self, q, k, v, attn_mask=None, key_padding_mask=None):
        x = self._mha(
            q, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]  # (batch_size, seq_len, embedding_dim)
        return self.dropout1(x)  # (batch_size, seq_len, embedding_dim)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, inputs):
        # target domain item sequence
        all_sample_events_target = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths_target = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
        # source domain item sequence
        all_sample_events_source = inputs['{}.{}.ids'.format(self._sequence_prefix, self._source_domain)]  # (all_batch_events)
        all_sample_lengths_source = inputs['{}.{}.length'.format(self._sequence_prefix, self._source_domain)]  # (batch_size)

        # sequential model encoder and target domain items embeddings from sequential model
        seq_embeddings_target, seq_mask_target = self._apply_sequential_encoder(
            all_sample_events_target, all_sample_lengths_target
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        seq_last_embeddings_target = self._get_last_embedding(seq_embeddings_target, seq_mask_target)  # (batch_size, embedding_dim) # last embeddings, used for forecasting
        
        # target domain items encoder for graph model
        all_final_user_embeddings_target, all_final_item_embeddings_target = \
            self._apply_graph_encoder(all_sample_events_target, all_sample_lengths_target)  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)
        # source domain items encoder for graph model
        all_final_user_embeddings_source, all_final_item_embeddings_source = \
            self._apply_graph_encoder(all_sample_events_source, all_sample_lengths_source)  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)
        
        # target domain items embeddings from graph model
        graph_embeddings_target, graph_item_ego_embeddings_target, graph_item_mask_target = self._get_graph_embeddings(
            inputs, self._sequence_prefix, self._graph_item_embeddings, all_final_item_embeddings_target
        )
        graph_item_embeddings_target = graph_embeddings_target[graph_item_mask_target]  # (batch_size, embedding_dim) 
        #graph_last_embeddings_target = self._get_last_embedding(graph_embeddings_target, graph_item_mask_target)  # (batch_size, embedding_dim)
        # source domain items embeddings from graph model
        graph_embeddings_source, graph_item_ego_embeddings_source, graph_item_mask_source = self._get_graph_embeddings(
            inputs, self._sequence_prefix, self._graph_item_embeddings, all_final_item_embeddings_source
        )
        graph_item_embeddings_source = graph_embeddings_source[graph_item_mask_source]  # (batch_size, embedding_dim) 
        #graph_last_embeddings_source = self._get_last_embedding(graph_embeddings_source, graph_item_mask_source)  # (batch_size, embedding_dim)

        # embeddings + graph_embeddings_target -> cross-attention
        # query   = embeddings
        # keys    = graph_embeddings_target
        # values  = graph_embeddings_target
        if self.norm_first: 
            graph_embeddings_target = graph_embeddings_target + self.norm1(self._ca_block(
                q=seq_embeddings_target,
                k=graph_embeddings_target,
                v=graph_embeddings_target,
                attn_mask=None,
                key_padding_mask=~graph_item_mask_target
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_target = graph_embeddings_target + self.norm2(self._ff_block(graph_embeddings_target))
        else:
            graph_embeddings_target = self.norm1(graph_embeddings_target + self._ca_block(
                q=seq_embeddings_target,
                k=graph_embeddings_target,
                v=graph_embeddings_target,
                attn_mask=None,
                key_padding_mask=~graph_item_mask_target
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_target = self.norm2(graph_embeddings_target + self._ff_block(graph_embeddings_target))
        # target-target cross-attention result
        mha_embeddings_target = torch.cat([seq_embeddings_target, graph_embeddings_target], dim=-1)
        mha_embeddings_target = self._mha_output_projection(mha_embeddings_target)  # (batch_size, seq_len, embedding_dim)
        mha_last_embeddings_target = self._get_last_embedding(mha_embeddings_target, seq_mask_target)  # (batch_size, embedding_dim)

        # embeddings + graph_embeddings_source -> cross-attention
        # query   = embeddings
        # keys    = graph_embeddings_source
        # values  = graph_embeddings_source
        if self.norm_first: 
            graph_embeddings_source = graph_embeddings_source + self.norm1(self._ca_block(
                q=seq_embeddings_target,
                k=graph_embeddings_source,
                v=graph_embeddings_source,
                attn_mask=None,
                key_padding_mask=~graph_item_mask_source # TODO: подать другую маску (разные маски для q и k, так как разные target и source домены)
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_source = graph_embeddings_source + self.norm2(self._ff_block(graph_embeddings_source))
        else:
            graph_embeddings_source = self.norm1(graph_embeddings_source + self._ca_block(
                q=seq_embeddings_target,
                k=graph_embeddings_source,
                v=graph_embeddings_source,
                attn_mask=None,
                key_padding_mask=~graph_item_mask_source # TODO: подать другую маску
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_source = self.norm2(graph_embeddings_source + self._ff_block(graph_embeddings_source))
        # source-target cross-attention result
        mha_embeddings_source = torch.cat([seq_embeddings_target, graph_embeddings_source], dim=-1)
        mha_embeddings_source = self._mha_output_projection(mha_embeddings_source)  # (batch_size, seq_len, embedding_dim)
        # TODO: implement seq_mask_source
        mha_last_embeddings_source = self._get_last_embedding(mha_embeddings_source, seq_mask_source)  # (batch_size, embedding_dim)

        if self.training:  # training mode
            # sequential part
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = seq_embeddings_target[seq_mask_target]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events
            )  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events
            )  # (all_batch_events, embedding_dim)

            # graph part, target domain item embeddings
            graph_positive_embeddings_target, _, graph_positive_mask_target = self._get_graph_embeddings(
                inputs, self._positive_prefix, self._graph_item_embeddings, all_final_item_embeddings_target
            )
            graph_negative_embeddings_target, _, graph_negative_mask_target = self._get_graph_embeddings(
                inputs, self._negative_prefix, self._graph_item_embeddings, all_final_item_embeddings_target
            )
            # b - batch_size, s - seq_len, d - embedding_dim
            graph_positive_scores_target = torch.einsum(
                'bd,bsd->bs', graph_item_embeddings_target, graph_positive_embeddings_target
            )  # (batch_size, seq_len)
            graph_negative_scores_target = torch.einsum(
                'bd,bsd->bs', graph_item_embeddings_target, graph_negative_embeddings_target
            )  # (batch_size, seq_len)
            graph_positive_scores_target = graph_positive_scores_target[graph_positive_mask_target]  # (all_batch_events)
            graph_negative_scores_target = graph_negative_scores_target[graph_negative_mask_target]  # (all_batch_events)

            # graph part, source domain item embeddings
            graph_positive_embeddings_source, _, graph_positive_mask_source = self._get_graph_embeddings(
                inputs, self._positive_prefix, self._graph_item_embeddings, all_final_item_embeddings_source
            )
            graph_negative_embeddings_source, _, graph_negative_mask_source = self._get_graph_embeddings(
                inputs, self._negative_prefix, self._graph_item_embeddings, all_final_item_embeddings_source
            )
            # b - batch_size, s - seq_len, d - embedding_dim
            graph_positive_scores_source = torch.einsum(
                'bd,bsd->bs', graph_item_embeddings_source, graph_positive_embeddings_source
            )  # (batch_size, seq_len)
            graph_negative_scores_source = torch.einsum(
                'bd,bsd->bs', graph_item_embeddings_source, graph_negative_embeddings_source
            )  # (batch_size, seq_len)
            graph_positive_scores_source = graph_positive_scores_source[graph_positive_mask_source]  # (all_batch_events)
            graph_negative_scores_source = graph_negative_scores_source[graph_negative_mask_source]  # (all_batch_events)

            return {
                # sequential part
                # target domain item embeddings
                'current_embeddings': all_sample_embeddings, 
                'positive_embeddings': all_positive_sample_embeddings,
                'negative_embeddings': all_negative_sample_embeddings,
                    
                # graph part
                # target domain item embeddings
                'graph_positive_embeddings_target': graph_positive_embeddings_target[graph_positive_mask_target],
                'graph_negative_embeddings_target': graph_negative_embeddings_target[graph_negative_mask_target],
                'graph_positive_scores_target': graph_positive_scores_target,
                'graph_negative_scores_target': graph_negative_scores_target,
                'graph_item_embeddings_target': graph_item_embeddings_target,
                # source domain item embeddings
                'graph_positive_embeddings_source': graph_positive_embeddings_source[graph_positive_mask_source],
                'graph_negative_embeddings_source': graph_negative_embeddings_source[graph_negative_mask_source],
                'graph_positive_scores_source': graph_positive_scores_source,
                'graph_negative_scores_source': graph_negative_scores_source,
                'graph_item_embeddings_source': graph_item_embeddings_source
            }
        else:  # eval mode
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events
                )  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths
                )  # (batch_size, num_candidates, embedding_dim)

                aggregated_last_embeddings = torch.max(
                    seq_last_embeddings_target, 
                    torch.max(mha_last_embeddings_target, mha_last_embeddings_source)
                )  # (batch_size, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    aggregated_last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)

                aggregated_last_embeddings = torch.max(
                    seq_last_embeddings_target, 
                    torch.max(mha_last_embeddings_target, mha_last_embeddings_source)
                )  # (batch_size, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    aggregated_last_embeddings,
                    candidate_embeddings
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
