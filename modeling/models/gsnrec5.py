from models.base import SequentialTorchModel

from utils import create_masked_tensor, DEVICE
import torch
import torch.nn as nn
import torch.nn.functional as F


class GSNRec5Model(SequentialTorchModel, config_name='gsnrec5'):

    def __init__(
            self,
            user_prefix,
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            candidate_prefix,
            num_users,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            mlp_layers,
            graph,
            dim_feedforward,
            ratings_prefix,
            dropout_seq,
            dropout_graph,
            dropout_gsnrec,
            dropout_mlp,
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
            dropout=dropout_seq,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            is_causal=True
        )
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._candidate_prefix = candidate_prefix

        self._user_prefix = user_prefix
        self._graph = graph
        self._num_layers = num_layers
        self._num_users = num_users
        self._dropout_rate = dropout_graph
        self._mlp_layers = mlp_layers
        self._ratings_prefix = ratings_prefix
        self._dropout_gsnrec = dropout_gsnrec
        self._dropout_mlp = dropout_mlp

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )

        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        self._init_weights(initializer_range)

        self._user_embeddings = nn.Embedding(
            num_embeddings=self._num_users + 2,
            embedding_dim=self._embedding_dim
        )

        self._item_embeddings = nn.Embedding(
            num_embeddings=self._num_items + 2,
            embedding_dim=self._embedding_dim
        )

        self.attn = nn.Linear(self._embedding_dim, self._embedding_dim, bias=True)
        self.q = nn.Linear(self._embedding_dim, 1, bias=False)

        self.mlps = nn.ModuleList()
        for _ in range(self._mlp_layers):
            self.mlps.append(nn.Linear(self._embedding_dim, self._embedding_dim, bias=True))
            nn.init.xavier_normal_(self.mlps[-1].weight.data)

        self.E2 = nn.Parameter(torch.empty(self._num_users + 2 + self._num_items + 2, self._embedding_dim))
        nn.init.xavier_normal_(self.E2.data)

        self.scores_gsnrec_linear = nn.Linear(1, self._embedding_dim, bias=True)
        self.output_gsnrec_linear = nn.Linear(self._embedding_dim, 1, bias=False)

        self.attn_softmax = nn.Softmax(dim=1)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            user_prefix=config['user_prefix'],
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            candidate_prefix=config['candidate_prefix'],
            graph=kwargs['graph'],
            num_users=kwargs['num_users'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            mlp_layers=config['mlp_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout_seq=config.get('dropout_seq', 0.0),
            dropout_graph=config.get('dropout_graph', 0.0),
            dropout_gsnrec=config.get('dropout_gsnrec', 0.0),
            dropout_mlp=config.get('dropout_mlp', 0.0),
            ratings_prefix=config['ratings_prefix'],
            initializer_range=config.get('initializer_range', 0.02),
        )

    def _apply_graph_encoder(self):
        ego_embeddings = torch.cat((self._user_embeddings.weight, self._item_embeddings.weight), dim=0)
        all_embeddings = [ego_embeddings]

        if self._dropout_rate > 0:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + (1 - self._dropout_rate)
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / (1 - self._dropout_rate)
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for i in range(self._num_layers):
            ego_embeddings = torch.sparse.mm(graph_dropped, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.stack(all_embeddings, 0).mean(0)

        user_final_embeddings, item_final_embeddings = torch.split(
            all_embeddings, [self._num_users + 2, self._num_items + 2]
        )

        return user_final_embeddings, item_final_embeddings

    def _get_embeddings(self, inputs, prefix, ego_embeddings, final_embeddings):
        ids = inputs['{}.ids'.format(prefix)]  # (all_batch_events)
        lengths = inputs['{}.length'.format(prefix)]  # (batch_size)

        final_embeddings = final_embeddings[ids]  # (all_batch_events, embedding_dim)
        ego_embeddings = ego_embeddings(ids)  # (all_batch_events, embedding_dim)

        padded_embeddings, mask = create_masked_tensor(
            final_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        padded_ego_embeddings, ego_mask = create_masked_tensor(
            ego_embeddings, lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        assert torch.all(mask == ego_mask)

        return padded_embeddings, padded_ego_embeddings, mask

    def forward(self, inputs):
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)

        sasrec_embeddings, sasrec_mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode

            ### light_gcn - graph part of siren
            graph_user_embeddings, graph_item_embeddings = \
                self._apply_graph_encoder()  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

            user_embeddings, user_ego_embeddings, user_mask = self._get_embeddings(
                inputs, self._user_prefix, self._user_embeddings, graph_user_embeddings
            )
            user_embeddings = user_embeddings[user_mask]  # (all_batch_events, embedding_dim)

            # z_p siren
            # all_embeddings_positive = torch.cat((graph_user_embeddings, graph_item_embeddings), dim=0)
            # # shapes z_p:
            # # [num_users + 2 + num_items + 2; embedding_dim * (num_layers + 1)
            # # например для датасета ml1m и num_layers = 2: [9069; 192], т.е. [6022 + 2 + 3043 + 2; 64 * 3]
            #
            # # MLP part of siren
            # mlp = [self.E2]
            # x = F.dropout(F.relu(self.mlps[0](self.E2)), self._dropout_mlp, training=self.training)
            # for i in range(1, self._mlp_layers):
            #     x = self.mlps[i](x)
            #     x = F.relu(x)
            #     x = F.dropout(x, self._dropout_mlp, training=self.training)
            #     mlp.append(x)
            # # z_n siren
            # all_embeddings_negative = mlp[-1]
            # # shapes z_n
            # # [num_users + num_items; embedding_dim]
            # # например для датасета ml1m: [9065; 64]
            #
            # # Z
            # w_p = self.q(F.dropout(torch.tanh((self.attn(all_embeddings_positive))), self._dropout_mlp, training=self.training))
            # w_n = self.q(F.dropout(torch.tanh((self.attn(all_embeddings_negative))), self._dropout_mlp, training=self.training))
            # alpha_ = self.attn_softmax(torch.cat([w_p, w_n], dim=1))
            #
            # graph_embeddings = alpha_[:, 0].view(len(all_embeddings_positive), 1) * all_embeddings_positive + \
            #                    alpha_[:, 1].view(len(all_embeddings_positive), 1) * all_embeddings_negative
            #
            # graph_user_embeddings, graph_item_embeddings = torch.split(graph_embeddings, [self._num_users + 2, self._num_items + 2])

            positive_embeddings, _, positive_mask = self._get_embeddings(
                inputs, self._positive_prefix, self._item_embeddings, graph_item_embeddings
            )  # (batch_size, seq_len, embedding_dim)
            negative_embeddings, _, negative_mask = self._get_embeddings(
                inputs, self._negative_prefix, self._item_embeddings, graph_item_embeddings
            )  # (batch_size, seq_len, embedding_dim)

            my_user_embeddings, _, my_user_mask = self._get_embeddings(
                inputs, self._user_prefix, self._user_embeddings, graph_user_embeddings
            )  # (batch_size, seq_len, embedding_dim)

            my_user_embeddings = my_user_embeddings[my_user_mask]  # (all_batch_events, embedding_dim)
            # b - batch_size, s - seq_len, d - embedding_dim
            positive_graph_scores = torch.einsum(
                'bd,bsd->bs',
                my_user_embeddings,
                positive_embeddings
            )  # (batch_size, seq_len)
            negative_graph_scores = torch.einsum(
                'bd,bsd->bs',
                my_user_embeddings,
                negative_embeddings
            )  # (batch_size, seq_len)

            positive_graph_scores = positive_graph_scores[positive_mask]  # (all_batch_events)
            negative_graph_scores = negative_graph_scores[negative_mask]  # (all_batch_events)

            # sasrec
            all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
            all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

            all_sample_embeddings = sasrec_embeddings[sasrec_mask]  # (all_batch_events, embedding_dim)
            all_positive_sample_embeddings = self._item_embeddings(
                all_positive_sample_events
            )  # (all_batch_events, embedding_dim)
            all_negative_sample_embeddings = self._item_embeddings(
                all_negative_sample_events
            )  # (all_batch_events, embedding_dim)

            positive_sequence_scores = torch.einsum(
                'bd,bd->b',
                all_positive_sample_embeddings,
                all_sample_embeddings
            )
            negative_sequence_scores = torch.einsum(
                'bd,bd->b',
                all_negative_sample_embeddings,
                all_sample_embeddings
            )

            positive_graph_scores = positive_graph_scores.unsqueeze(-1).unsqueeze(-1)
            positive_sequence_scores = positive_sequence_scores.unsqueeze(-1).unsqueeze(-1)
            negative_graph_scores = negative_graph_scores.unsqueeze(-1).unsqueeze(-1)
            negative_sequence_scores = negative_sequence_scores.unsqueeze(-1).unsqueeze(-1)

            positive_all_scores = torch.cat((positive_sequence_scores, positive_graph_scores), dim=1)
            negative_all_scores = torch.cat((negative_sequence_scores, negative_graph_scores), dim=1)

            positive_all_scores = F.dropout(torch.tanh(positive_all_scores), p=self._dropout_gsnrec, training=self.training)
            negative_all_scores = F.dropout(torch.tanh(negative_all_scores), p=self._dropout_gsnrec, training=self.training)

            beta_ = self.attn_softmax(torch.cat([positive_all_scores, negative_all_scores], dim=1))

            len_seq = len(positive_all_scores)

            final_positive_scores = (beta_[:, 0].view(len_seq, 1) * positive_all_scores[:, 0, :]
                                     + beta_[:, 1].view(len_seq, 1) * positive_all_scores[:, 1, :])

            final_negative_scores = (beta_[:, 2].view(len_seq, 1) * negative_all_scores[:, 0, :]
                                     + beta_[:, 3].view(len_seq, 1) * negative_all_scores[:, 1, :])

            return {
                'current_embeddings': all_sample_embeddings,
                'positive_embeddings': all_positive_sample_embeddings,
                'negative_embeddings': all_negative_sample_embeddings,
                'positive_scores': final_positive_scores,
                'negative_scores': final_negative_scores,
                'item_embeddings': torch.cat((self._user_embeddings.weight, self._item_embeddings.weight), dim=0)
            }
        else:  # eval mode
            last_embeddings = self._get_last_embedding(sasrec_embeddings, sasrec_mask)  # (batch_size, embedding_dim)

            # b - batch_size, n - num_candidates, d - embedding_dim
            candidate_scores = torch.einsum(
                'bd,nd->bn',
                last_embeddings,
                self._item_embeddings.weight
            )  # (batch_size, num_items + 2)
            candidate_scores[:, 0] = -torch.inf
            candidate_scores[:, self._num_items + 1:] = -torch.inf

            if '{}.ids'.format(self._candidate_prefix) in inputs:
                candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                batch_size = candidate_lengths.shape[0]
                num_candidates = candidate_lengths[0]

                candidate_scores = torch.gather(
                    input=candidate_scores,
                    dim=1,
                    index=torch.reshape(candidate_events, [batch_size, num_candidates])
                )  # (batch_size, num_candidates)

            values, indices = torch.topk(
                candidate_scores,
                k=20, dim=-1, largest=True
            )  # (batch_size, 20), (batch_size, 20)

            return indices
