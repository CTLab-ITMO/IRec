from blocks.encoder import TorchEncoder as Encoder, Transformer
from blocks.head import TorchHead as Head

import torch
import torch.nn as nn


class BertEncoder(Encoder, config_name='bert4rec'):

    def __init__(
            self,
            prefix,
            num_layers,
            num_heads,
            hidden_size,
            output_prefix=None,
            activation='relu',
            input_dim=None,
            output_dim=None,
            dropout=0.0,
            eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._encoder = Transformer(
            prefix=prefix,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            input_dim=input_dim,
            output_dim=output_dim,
            output_prefix=output_prefix,
            initializer_range=initializer_range
        )

    @classmethod
    def create_from_config(cls, config):
        return cls(
            prefix=config['prefix'],
            num_layers=config['num_layers'],
            num_heads=config.get('num_heads', config['embedding_dim'] // 64),
            hidden_size=config['embedding_dim'],
            activation=config.get('activation', 'relu'),
            output_prefix=config.get('output_prefix', None),
            input_dim=config.get('input_dim', None),
            output_dim=config.get('output_dim', None),
            dropout=config.get('dropout', 0.0),
            eps=config.get('eps', 1e-5),
            initializer_range=config.get('initializer_range', 0.02)
        )

    def forward(self, inputs):
        return self._encoder(inputs)


class BertHead(Head, config_name='bert4rec'):

    def __init__(
            self,
            prefix,
            labels_prefix,
            candidates_prefix,
            input_dim,
            output_dim,
            output_prefix=None,
            initializer_range=0.02
    ):
        super().__init__()
        self._prefix = prefix
        self._labels_prefix = labels_prefix
        self._candidates_prefix = candidates_prefix
        self._output_prefix = output_prefix or prefix

        self._encoder = nn.Linear(input_dim, output_dim)
        self._init_weights(initializer_range)

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        nn.init.uniform_(self._encoder.weight.data, a=-initializer_range, b=initializer_range)
        nn.init.uniform_(self._encoder.bias.data, a=-initializer_range, b=initializer_range)

    @classmethod
    def create_from_config(cls, config, num_users=None, num_items=None):
        return cls(
            prefix=config['prefix'],
            labels_prefix=config['labels_prefix'],
            input_dim=config['input_dim'],
            output_dim=num_items + 1,
            candidates_prefix=config['candidates_prefix'],
            output_prefix=config.get('prefix', None)
        )

    def forward(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, max_seq_len, input_dim)
        mask = inputs['{}.mask'.format(self._prefix)]  # (batch_size, max_seq_len)

        embeddings = self._encoder(embeddings)  # (batch_size, max_seq_len, output_dim)

        if self.training:  # train mode
            inputs = self._train_postprocessing(inputs, logits=embeddings, logits_mask=mask)
        else:  # eval mode
            inputs = self._eval_postprocessing(inputs, logits=embeddings)

        return inputs

    def _train_postprocessing(self, inputs, logits, logits_mask):
        all_logits = logits[logits_mask]  # (all_events)

        all_labels = inputs['{}.ids'.format(self._labels_prefix)].long()  # (all_events)
        labels_mask = (all_labels != 0).bool()  # (all_events)
        needed_labels = all_labels[labels_mask]  # (non_zero_events)
        needed_logits = all_logits[labels_mask]  # (non_zero_events)

        inputs[self._output_prefix] = needed_logits  # (batch_size, num_candidates)
        inputs['{}.ids'.format(self._labels_prefix)] = needed_labels

        return inputs

    def _eval_postprocessing(self, inputs, logits):
        batch_size = logits.shape[0]

        lengths = inputs['{}.length'.format(self._prefix)]  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, logits.shape[-1]))  # (batch_size, 1, num_classes)

        last_values = logits.gather(dim=1, index=lengths)  # (batch_size, 1, num_classes)
        last_values = last_values.squeeze(1)  # (batch_size, num_classes)

        candidate_ids = inputs['{}.ids'.format(self._candidates_prefix)]  # (all_candidates)
        candidate_ids = torch.reshape(candidate_ids, (batch_size, -1))  # (batch_size, num_candidates)
        candidate_scores = last_values.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)
        inputs[self._output_prefix] = candidate_scores  # (batch_size, num_candidates)

        labels_ids = inputs['{}.ids'.format(self._labels_prefix)]  # (all_candidates)
        labels_ids = torch.reshape(labels_ids, (batch_size, -1))  # (batch_size, num_candidates)
        inputs['{}.ids'.format(self._labels_prefix)] = labels_ids

        return inputs
