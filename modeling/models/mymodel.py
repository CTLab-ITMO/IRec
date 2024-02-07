from models.base import SequentialTorchModel, TorchModel

from utils import create_masked_tensor, get_activation_function

import torch
import torch.nn as nn


class MyMultidomainModel(SequentialTorchModel, config_name='my_multidomain_model'):
    def __init__(
            self,
            # base params
            other_domains, # Q: нужно ли это здесь?
            sequence_prefix,
            positive_prefix,
            negative_prefix,
            labels_prefix, # from bert4rec # Q: нужно или удалить?
            candidate_prefix,
            num_items,
            max_sequence_length,
            embedding_dim,
            num_heads,
            num_layers,
            dim_feedforward,
            # graph params
            user_prefix,
            graph,
            num_users,
            graph_embedding_dim,
            graph_num_layers,
            # params with default values
            dropout=0.0,
            graph_keep_prob=1.0,
            graph_dropout=0.0,
            activation='relu',
            score_function='cos', # способ вычисления скора для айтема: 'cos' или 'dot'
            layer_norm_eps=1e-9,
            initializer_range=0.02,
            use_ce=False
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
            is_causal=True # false for bert4rec # Q: за что отвечает данный параметр и какое значение нужно?
        )
        # transformer part (base)
        self._sequence_prefix = sequence_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._labels_prefix = labels_prefix  # from bert4rec # Q: нужно или удалить?
        self._candidate_prefix = candidate_prefix

        self._use_ce = use_ce # from sasrec # Q: нужно или удалить?
        self._score_function = score_function
        self._other_domains = other_domains

        self._output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim
        )
        self._bias = nn.Parameter(
            data=torch.zeros(num_items + 2),
            requires_grad=True
        )

        self._init_weights(initializer_range)

        # graph part
        self._graph = graph
        self._graph_user_prefix = user_prefix # Q: откуда взять?
        self._graph_positive_prefix = positive_prefix # Q: нужна отдельная переменная или достаточно той, что у трансформера?
        self._graph_negative_prefix = negative_prefix # Q: нужна отдельная переменная или достаточно той, что у трансформера?
        self._graph_candidate_prefix = candidate_prefix # Q: нужна отдельная переменная или достаточно той, что у трансформера?

        self._graph_num_users = num_users # Q: откуда взять?
        self._graph_num_items = num_items # Q: нужна отдельная переменная или достаточно той, что у трансформера?
        self._graph_embedding_dim = graph_embedding_dim # Q: нужна отдельная переменная или достаточно той, что у трансформера?
        self._graph_num_layers = graph_num_layers
        self._graph_keep_prob = graph_keep_prob
        self._graph_dropout = graph_dropout

        self._graph_user_embeddings = nn.Embedding(
            num_embeddings=self._graph_num_users + 2,
            embedding_dim=self._graph_embedding_dim
        )
        self._graph_item_embeddings = nn.Embedding(
            num_embeddings=self._graph_num_items + 2,
            embedding_dim=self._graph_embedding_dim
        )

        self._init_graph_weights(initializer_range)

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

        self.norm_first = norm_first # Q: откуда взять?
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self._mha_output_projection = nn.Linear(
            in_features=2 * embedding_dim,
            out_features=embedding_dim,
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            # transformer part (base)
            sequence_prefix=config['sequence_prefix'],
            positive_prefix=config['positive_prefix'],
            negative_prefix=config['negative_prefix'],
            labels_prefix=config['labels_prefix'], # from bert4rec
            candidate_prefix=config['candidate_prefix'],
            num_items=kwargs['num_items'],
            max_sequence_length=kwargs['max_sequence_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config.get('num_heads', int(config['embedding_dim'] // 64)),
            num_layers=config['num_layers'],
            dim_feedforward=config.get('dim_feedforward', 4 * config['embedding_dim']),
            dropout=config.get('dropout', 0.0),
            use_ce=config.get('use_ce', False),
            initializer_range=config.get('initializer_range', 0.02)

            # graph part
            # TODO
        )
    
    @torch.no_grad()
    def _init_graph_weights(self, initializer_range):
        nn.init.trunc_normal_(
            self._graph_user_embeddings.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )

        nn.init.trunc_normal_(
            self._graph_item_embeddings.weight.data,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )
    
    def _apply_graph_encoder(self):
        all_embeddings = torch.cat([self._graph_user_embeddings.weight, self._graph_item_embeddings.weight], dim=0)
        embeddings = [all_embeddings]

        if self._graph_dropout:  # drop some edges
            if self.training:  # training_mode
                size = self._graph.size()
                index = self._graph.indices().t()
                values = self._graph.values()
                random_index = torch.rand(len(values)) + self._graph_keep_prob
                random_index = random_index.int().bool()
                index = index[random_index]
                values = values[random_index] / self._graph_keep_prob
                graph_dropped = torch.sparse.FloatTensor(index.t(), values, size)
            else:  # eval mode
                graph_dropped = self._graph
        else:
            graph_dropped = self._graph

        for layer in range(self._graph_num_layers):
            all_embeddings = torch.sparse.mm(graph_dropped, all_embeddings)
            embeddings.append(all_embeddings)

        light_out = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        user_final_embeddings, item_final_embeddings = torch.split(
            light_out, [self._graph_num_users + 2, self._graph_num_items + 2]
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
        '''
        inputs = {
            'user.ids': sample['user.ids'],
            'user.length': sample['user.length'],

            'item.ids': item_sequence,
            'item.length': len(item_sequence),

            'positive.ids': next_item_sequence,
            'positive.length': len(next_item_sequence),

            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),

            'item.{}.ids'.format(self._other_domains[0]): item_sequence,
            'item.{}.length'.format(self._other_domains[0]): len(item_sequence),

            'positive.{}.ids'.format(self._other_domains[0]): next_item_sequence,
            'positive.{}.length'.format(self._other_domains[0]): len(next_item_sequence),

            'negative.{}.ids'.format(self._other_domains[0]): negative_sequence,
            'negative.{}.length'.format(self._other_domains[0]): len(negative_sequence)
        }
        '''

        # последовательности айтемов из target domain
        all_sample_events = inputs['{}.ids'.format(self._sequence_prefix)]  # (all_batch_events)
        all_sample_lengths = inputs['{}.length'.format(self._sequence_prefix)]  # (batch_size)
        # последовательности айтемов из source domain
        all_sample_events_source = inputs['{}.{}.ids'.format(self._sequence_prefix, self._other_domains[0])]  # (all_batch_events)
        all_sample_lengths_source = inputs['{}.{}.length'.format(self._sequence_prefix, self._other_domains[0])]  # (batch_size)

        # энкодер айтемов из target domain для трансформера
        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)
        # e^u_n оранжевый_1 - последний эмбед, используемый для прогноза
        last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)

        # энкодер айтемов из target domain для графа
        all_final_user_embeddings, all_final_item_embeddings = \
            self._apply_graph_encoder(all_sample_events, all_sample_lengths)  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)
        # энкодер айтемов из source domain для графа
        all_final_user_embeddings_source, all_final_item_embeddings_source = \
            self._apply_graph_encoder(all_sample_events_source, all_sample_lengths_source)  # (num_users + 2, embedding_dim), (num_items + 2, embedding_dim)

        # Q: ???
        # Q: как получить разные выходы графовой модели (оранжевый_2 и синий_2 на рисунке) и в чем различие между выходами?
        # оранжевые_2 эмбеды из графа # Q: эмбеды айтемов?
        graph_embeddings_1, user_ego_embeddings_1, user_mask_1 = self._get_graph_embeddings(
            inputs, self._graph_user_prefix, self._graph_user_embeddings, all_final_user_embeddings
        )
        user_embeddings_1 = graph_embeddings_1[user_mask_1]  # (batch_size, embedding_dim)
        # синие_2 эмбеды из графа # Q: эмбеды юзеров?
        graph_embeddings_2, user_ego_embeddings_2, user_mask_2 = self._get_graph_embeddings(
            inputs, self._graph_user_prefix, self._graph_user_embeddings, all_final_user_embeddings_source
        )
        user_embeddings_2 = graph_embeddings_2[user_mask_2]  # (batch_size, embedding_dim)

        # todo: выход энкодера + оранжевые_2 эмбеды из графа -> cross-attention
        # keys   = embeddings
        # values = graph_embeddings_1
        # query  = graph_embeddings_1
        if self.norm_first: 
            graph_embeddings_1 = graph_embeddings_1 + self.norm1(self._ca_block(
                q=embeddings,
                k=graph_embeddings_1,
                v=graph_embeddings_1,
                attn_mask=None,
                key_padding_mask=~mask
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_1 = graph_embeddings_1 + self.norm2(self._ff_block(graph_embeddings_1))
        else:
            graph_embeddings_1 = self.norm1(graph_embeddings_1 + self._ca_block(
                q=embeddings,
                k=graph_embeddings_1,
                v=graph_embeddings_1,
                attn_mask=None,
                key_padding_mask=~mask
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_1 = self.norm2(graph_embeddings_1 + self._ff_block(graph_embeddings_1))
        # результат 1-го cross-attention - оранжевые_3 эмбеды
        embeddings_1 = torch.cat([embeddings, graph_embeddings_1], dim=-1)
        embeddings_1 = self._mha_output_projection(embeddings_1)  # (batch_size, seq_len, embedding_dim)

        # todo: выход энкодера + синие_2 эмбеды из графа -> cross-attention
        # keys   = embeddings
        # values = graph_embeddings_2
        # query  = graph_embeddings_2
        if self.norm_first: 
            graph_embeddings_2 = graph_embeddings_2 + self.norm1(self._ca_block(
                q=embeddings,
                k=graph_embeddings_2,
                v=graph_embeddings_2,
                attn_mask=None,
                key_padding_mask=~mask
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_2 = graph_embeddings_2 + self.norm2(self._ff_block(graph_embeddings_2))
        else:
            graph_embeddings_2 = self.norm1(graph_embeddings_2 + self._ca_block(
                q=embeddings,
                k=graph_embeddings_2,
                v=graph_embeddings_2,
                attn_mask=None,
                key_padding_mask=~mask
            ))  # (batch_size, seq_len, embedding_dim)
            graph_embeddings_2 = self.norm2(graph_embeddings_2 + self._ff_block(graph_embeddings_2))
        # результат 2-го cross-attention - синие_3 эмбеды
        embeddings_2 = torch.cat([embeddings, graph_embeddings_2], dim=-1)
        embeddings_2 = self._mha_output_projection(embeddings_2)  # (batch_size, seq_len, embedding_dim)

        # итого, имеем 3 выхода модели:
        # embeddings   - оранжевый_1 (он же синий_1)
        # embeddings_1 - оранжевый_3
        # embeddings_2 - синий_3

        if self.training:  # training mode
            # TODO # Q: ???
            if self._use_ce:
                return {'logits': embeddings[mask]}
            else:
                all_positive_sample_events = inputs['{}.ids'.format(self._positive_prefix)]  # (all_batch_events)
                all_negative_sample_events = inputs['{}.ids'.format(self._negative_prefix)]  # (all_batch_events)

                all_sample_embeddings = embeddings[mask]  # (all_batch_events, embedding_dim)
                all_positive_sample_embeddings = self._item_embeddings(
                    all_positive_sample_events
                )  # (all_batch_events, embedding_dim)
                all_negative_sample_embeddings = self._item_embeddings(
                    all_negative_sample_events
                )  # (all_batch_events, embedding_dim)

                return {
                    'current_embeddings': all_sample_embeddings,
                    'positive_embeddings': all_positive_sample_embeddings,
                    'negative_embeddings': all_negative_sample_embeddings
                }
        else:  # eval mode
            # TODO # Q: ???
            if self._use_ce:
                last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, num_items)

                if '{}.ids'.format(self._candidate_prefix) in inputs:
                    candidate_events = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
                    candidate_lengths = inputs['{}.length'.format(self._candidate_prefix)]  # (batch_size)

                    candidate_ids = torch.reshape(
                        candidate_events,
                        (candidate_lengths.shape[0], candidate_lengths[0])
                    )  # (batch_size, num_candidates)
                    candidate_scores = last_embeddings.gather(
                        dim=1, index=candidate_ids
                    )  # (batch_size, num_candidates)
                else:
                    candidate_scores = last_embeddings  # (batch_size, num_items + 2)
                    candidate_scores[:, 0] = -torch.inf
                    candidate_scores[:, self._num_items + 1:] = -torch.inf
            else:
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

                    candidate_scores = torch.einsum(
                        'bd,bnd->bn',
                        last_embeddings,
                        candidate_embeddings
                    )  # (batch_size, num_candidates)
                else:
                    candidate_embeddings = self._item_embeddings.weight  # (num_items, embedding_dim)
                    candidate_scores = torch.einsum(
                        'bd,nd->bn',
                        last_embeddings,
                        candidate_embeddings
                    )  # (batch_size, num_items)
                    candidate_scores[:, 0] = -torch.inf
                    candidate_scores[:, self._num_items + 1:] = -torch.inf

            return candidate_scores
