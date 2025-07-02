from .base import TorchModel

import torch
import torch.nn as nn

from irec.utils import create_masked_tensor


class MCLSRModel(TorchModel, config_name='mclsr'):
    def __init__(
        self,
        sequence_prefix,
        user_prefix,
        labels_prefix,
        negatives_prefix,
        candidate_prefix,
        num_users,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_graph_layers,
        common_graph,
        user_graph,
        item_graph,
        dropout=0.0,
        layer_norm_eps=1e-5,
        graph_dropout=0.0,
        alpha=0.5,
        initializer_range=0.02,
    ):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._user_prefix = user_prefix
        self._labels_prefix = labels_prefix
        self._negatives_prefix = negatives_prefix
        self._candidate_prefix = candidate_prefix

        self._num_users = num_users
        self._num_items = num_items

        self._embedding_dim = embedding_dim

        self._num_graph_layers = num_graph_layers
        self._graph_dropout = graph_dropout

        self._alpha = alpha

        self._graph = common_graph
        self._user_graph = user_graph
        self._item_graph = item_graph

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items
            + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim,
        )
        self._position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length
            + 1,  # in order to include `max_sequence_length` value
            embedding_dim=embedding_dim,
        )

        self._user_embeddings = nn.Embedding(
            num_embeddings=num_users
            + 2,  # add zero embedding + mask embedding
            embedding_dim=embedding_dim,
        )

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        # Current interest learning
        self._current_interest_learning_encoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=4 * embedding_dim,
                bias=False,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=4 * embedding_dim,
                out_features=1,
                bias=False,
            ),
        )

        # General interest learning
        self._general_interest_learning_encoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=False,
            ),
            nn.Tanh(),
        )

        # Cross-view contrastive learning
        self._sequential_projector = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )
        self._graph_projector = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._user_projection = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._item_projection = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
            ),
        )

        self._init_weights(initializer_range)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            user_prefix=config['user_prefix'],
            labels_prefix=config['labels_prefix'],
            negatives_prefix=config.get('negatives_prefix', 'negatives'),
            candidate_prefix=config['candidate_prefix'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_graph_layers=config['num_graph_layers'],
            common_graph=kwargs['graph'],
            user_graph=kwargs['user_graph'],
            item_graph=kwargs['item_graph'],
            dropout=config.get('dropout', 0.0),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
            graph_dropout=config.get('graph_dropout', 0.0),
            initializer_range=config.get('initializer_range', 0.02),
        )

    def _apply_graph_encoder(self, embeddings, graph, use_mean=False):
        assert self.training  # Here we use graph only in training_mode

        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        dropout_mask = torch.rand(len(values)) + self._graph_dropout
        dropout_mask = dropout_mask.int().bool()
        index = index[~dropout_mask]
        values = values[~dropout_mask] / (1.0 - self._graph_dropout)
        graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)

        all_embeddings = [embeddings]
        for _ in range(self._num_graph_layers):
            new_embeddings = torch.sparse.mm(graph_dropped, all_embeddings[-1])
            all_embeddings.append(new_embeddings)

        if use_mean:
            all_embeddings = torch.stack(all_embeddings, dim=1)
            return torch.mean(all_embeddings, dim=1)
        else:
            return all_embeddings[-1]

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]
        user_ids = inputs['{}.ids'.format(self._user_prefix)]

        embeddings, mask = create_masked_tensor(
            data=self._item_embeddings(all_sample_events),
            lengths=all_sample_lengths,
        )

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]
        
        # position embeddings for E_u,p
        positions = torch.arange(start=seq_len - 1, end=-1, step=-1, 
                                 device=mask.device)[None].tile([batch_size, 1]).long()
        positions_mask = positions < all_sample_lengths[:, None]
        positions = positions[positions_mask]
        position_embeddings, _ = create_masked_tensor(data=self._position_embeddings(positions), 
                                                      lengths=all_sample_lengths)
        
        positioned_embeddings = self._layernorm(embeddings + position_embeddings)
        positioned_embeddings = self._dropout(positioned_embeddings)
        positioned_embeddings[~mask] = 0

        # formula 2: A_s = softmax(...)
        sequential_attention_matrix = self._current_interest_learning_encoder(positioned_embeddings).squeeze()
        sequential_attention_matrix[~mask] = -torch.inf
        sequential_attention_matrix = torch.softmax(sequential_attention_matrix, dim=1)
        
        # formula 3: I_s = A_s * E_u
        sequential_representation = torch.einsum('bs,bsd->bd', 
                                                 sequential_attention_matrix, 
                                                 embeddings)

        if self.training:
            # General Interest Learning
            common_graph_user_embs_batch = self._user_embeddings(user_ids)
            common_graph_item_embs_batch, _ = create_masked_tensor(
                data=self._item_embeddings(all_sample_events),
                lengths=all_sample_lengths
            )
            
            # formula 5: A_c = softmax(...)
            graph_attention_matrix = torch.einsum('bd,bsd->bs', 
                                                  self._general_interest_learning_encoder(common_graph_user_embs_batch), 
                                                  common_graph_item_embs_batch)
            graph_attention_matrix[~mask] = -torch.inf
            graph_attention_matrix = torch.softmax(graph_attention_matrix, dim=1)
            
            # formula 6: I_c = A_c * E_u,uv
            original_graph_representation = torch.einsum('bs,bsd->bd', graph_attention_matrix, 
                                                         common_graph_item_embs_batch)
            
            original_sequential_representation = sequential_representation

            # Prepare for L_P (downstream loss)
            # formula 13: I_comb = ...
            combined_representation = (self._alpha * original_sequential_representation + 
                                       (1 - self._alpha) * original_graph_representation)
            
            # Prepare h_o and h_k for L_P (formula 14)
            labels = inputs['{}.ids'.format(self._labels_prefix)]
            labels_embeddings = self._item_embeddings(labels)
            negative_ids = inputs['{}.ids'.format(self._negatives_prefix)]
            negative_embeddings = self._item_embeddings(negative_ids)
            
            # Prepare for L_IL (Interest-level CL)
            # formula 7: MLP projection for I_s and I_c
            sequential_representation_proj = self._sequential_projector(original_sequential_representation)
            graph_representation_proj = self._graph_projector(original_graph_representation)

            # Prepare for L_UC (User-level CL)
            # formula 9: H_u,uu = GraphEncoder(...)
            user_graph_user_embs_all = self._apply_graph_encoder(embeddings=self._user_embeddings.weight, 
                                                                 graph=self._user_graph)
            user_graph_user_embs_batch = user_graph_user_embs_all[user_ids]
            
            # formula 10: MLP projection for H_u,uu and H_u,uv
            user_graph_user_embeddings_proj = self._user_projection(user_graph_user_embs_batch)
            common_graph_user_embeddings_proj = self._user_projection(common_graph_user_embs_batch)

            # Prepare for L_IC (Item-level CL) - fixed version
            common_graph_items_flat = common_graph_item_embs_batch[mask]
            
            # formula 9 (similar for items): H_v,vv = GraphEncoder(...)
            item_graph_items_all = self._apply_graph_encoder(embeddings=self._item_embeddings.weight, 
                                                             graph=self._item_graph)
            item_graph_items_flat = item_graph_items_all[all_sample_events]

            # Aggregate by unique items to prevent false negatives
            unique_item_ids, inverse_indices = torch.unique(all_sample_events, return_inverse=True)
            try:
                from torch_scatter import scatter_mean
            except ImportError:
                def scatter_mean(src, index, dim=0, dim_size=None):
                    out_size = dim_size if dim_size is not None else index.max() + 1
                    out = torch.zeros((out_size, src.size(1)), dtype=src.dtype, device=src.device)
                    counts = torch.bincount(index, minlength=out_size).unsqueeze(-1).clamp(min=1)
                    return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src) / counts
            
            num_unique_items = unique_item_ids.shape[0]

            unique_common_graph_items = scatter_mean(common_graph_items_flat, inverse_indices, dim=0, 
                                                     dim_size=num_unique_items)

            unique_item_graph_items = scatter_mean(item_graph_items_flat, inverse_indices, dim=0, 
                                                   dim_size=num_unique_items)
            
            # formula 10
            unique_common_graph_items_proj = self._item_projection(unique_common_graph_items)
            unique_item_graph_items_proj = self._item_projection(unique_item_graph_items)
            
            return {
                # For L_P (formula 14)
                'combined_representation': combined_representation,
                'label_representation': labels_embeddings,
                'negative_representation': negative_embeddings,

                # For L_IL (formula 8)
                'sequential_representation': sequential_representation_proj,
                'graph_representation': graph_representation_proj,

                # For L_UC (formula 11)
                'user_graph_user_embeddings': user_graph_user_embeddings_proj,
                'common_graph_user_embeddings': common_graph_user_embeddings_proj,
                
                # For L_IC (formula 11 - similar for items)
                'item_graph_item_embeddings': unique_item_graph_items_proj,
                'common_graph_item_embeddings': unique_common_graph_items_proj,
            }
        else:  # eval mode
            # formula 16: R(u,N) = Top-N((I_s)^T * h_o)
            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs[
                    '{}.ids'.format(self._candidate_prefix)
                ]  # (all_batch_candidates)
                candidate_lengths = inputs[
                    '{}.length'.format(self._candidate_prefix)

                ]  # (batch_size)

                candidate_embeddings = self._item_embeddings(
                    candidate_events,
                )  # (all_batch_candidates, embedding_dim)

                candidate_embeddings, _ = create_masked_tensor(
                    data=candidate_embeddings,
                    lengths=candidate_lengths,
                )  # (batch_size, num_candidates, embedding_dim)

                candidate_scores = torch.einsum(
                    'bd,bnd->bn',
                    sequential_representation, # I_s
                    candidate_embeddings, # h_o (and h_k)
                )  # (batch_size, num_candidates)
            else:
                candidate_embeddings = (
                    self._item_embeddings.weight
                )  # (num_items, embedding_dim)
                candidate_scores = torch.einsum(
                    'bd,nd->bn',
                    sequential_representation, # I_s
                    candidate_embeddings, # all h_v
                )  # (batch_size, num_items)
                candidate_scores[:, 0] = -torch.inf
                candidate_scores[:, self._num_items + 1 :] = -torch.inf


            values, indices = torch.topk(
                candidate_scores,
                k=20,
                dim=-1,
                largest=True,
            )  # (batch_size, 100), (batch_size, 100)

            return indices