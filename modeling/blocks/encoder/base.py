from utils import MetaParent, get_activation_function, maybe_to_list

import torch
import torch.nn as nn

import math


class BaseEncoder(metaclass=MetaParent):
    pass


class TorchEncoder(BaseEncoder, torch.nn.Module):
    pass


class TrainTestEncoder(TorchEncoder, config_name='train/test'):

    def __init__(self, train_encoder, test_encoder):
        super().__init__()
        self._train_encoder = train_encoder
        self._test_encoder = test_encoder

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            train_encoder=BaseEncoder.create_from_config(config["train"], **kwargs),
            test_encoder=BaseEncoder.create_from_config(config["test"], **kwargs)
        )

    def forward(self, inputs):
        if self.training:  # train mode
            inputs = self._train_encoder(inputs)
        else:  # eval mode
            inputs = self._test_encoder(inputs)

        return inputs


class CompositeEncoder(TorchEncoder, config_name='composite'):

    def __init__(self, encoders):
        super().__init__()
        self._encoders = encoders

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(encoders=nn.ModuleList([
            BaseEncoder.create_from_config(cfg, **kwargs)
            for cfg in config['encoders']
        ]))

    def forward(self, inputs):
        for encoder in self._encoders:
            inputs = encoder(inputs)
        return inputs


class EinsumEncoder(TorchEncoder, config_name='einsum'):

    def __init__(
            self,
            fst_prefix,
            snd_prefix,
            output_prefix,
            mask_prefix,
            operation
    ):
        super().__init__()
        self._fst_prefix = fst_prefix
        self._snd_prefix = snd_prefix
        self._output_prefix = output_prefix
        self._operation = operation
        self._mask_prefix = mask_prefix

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_prefix].clone()
        fst_mask = inputs['{}.mask'.format(self._fst_prefix)]
        fst_embeddings[~fst_mask] = 0

        snd_embeddings = inputs[self._snd_prefix]
        snd_mask = inputs['{}.mask'.format(self._snd_prefix)].clone()
        snd_embeddings[~snd_mask] = 0

        inputs[self._output_prefix] = torch.einsum(self._operation, fst_embeddings, snd_embeddings)
        inputs['{}.mask'.format(self._output_prefix)] = inputs['{}.mask'.format(self._mask_prefix)]
        inputs[self._output_prefix][~inputs['{}.mask'.format(self._output_prefix)]] = 0

        return inputs


class LastItemEncoder(TorchEncoder, config_name='last_item'):

    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

    def forward(self, inputs):
        embeddings = inputs[self._prefix]  # (batch_size, seq_len, emb_dim)
        mask = inputs['{}.mask'.format(self._prefix)]  # (batch_size, seq_len)
        embeddings[~mask] = 0

        lengths = torch.sum(mask, dim=-1)  # (batch_size)
        lengths = (lengths - 1).unsqueeze(-1)  # (batch_size, 1)
        last_masks = mask.gather(dim=1, index=lengths)  # (batch_size, 1)

        lengths = lengths.unsqueeze(-1)  # (batch_size, 1, 1)
        lengths = torch.tile(lengths, (1, 1, embeddings.shape[-1]))  # (batch_size, 1, emb_dim)
        last_embeddings = embeddings.gather(dim=1, index=lengths)  # (batch_size, 1, emb_dim)

        inputs[self._output_prefix] = last_embeddings  # (batch_size, 1, emb_dim)
        inputs['{}.mask'.format(self._output_prefix)] = last_masks  # (batch_size, 1)

        return inputs


class UnmaskEncoder(TorchEncoder, config_name='unmask'):

    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = maybe_to_list(prefix)
        self._output_prefix = maybe_to_list(output_prefix or prefix)
        assert len(self._prefix) == len(self._output_prefix)

    def forward(self, inputs):
        for prefix, output_prefix in zip(self._prefix, self._output_prefix):
            embeddings = inputs[prefix]
            mask = inputs['{}.mask'.format(prefix)]

            inputs[output_prefix] = embeddings[mask]
            inputs['{}.mask'.format(output_prefix)] = mask[mask]

        return inputs


class Transformer(TorchEncoder, config_name='transformer'):

    def __init__(
            self,
            prefix,
            hidden_size,
            num_heads,
            num_layers,
            dim_feedforward,
            dropout=0.0,
            activation='relu',
            layer_norm_eps=1e-5,
            input_dim=None,
            output_dim=None,
            output_prefix=None,
            initializer_range=0.02
    ):
        super().__init__()
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        self._input_projection = nn.Identity()
        if input_dim is not None:
            self._input_projection = nn.Linear(input_dim, hidden_size)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation_function(activation),
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._output_projection = nn.Identity()
        if output_dim is not None:
            self._output_projection = nn.Linear(hidden_size, output_dim)

        self._init_weights(initializer_range)

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')

    def forward(self, inputs):
        embeddings = inputs[self._prefix]
        attention_mask = inputs[f'{self._prefix}.mask']

        embeddings = self._input_projection(embeddings)
        embeddings = self._encoder(
            src=embeddings,
            src_key_padding_mask=~attention_mask
        )
        embeddings = self._output_projection(embeddings)

        inputs[self._output_prefix] = embeddings
        inputs['{}.mask'.format(self._output_prefix)] = attention_mask

        return inputs


class Tower(TorchEncoder, config_name='tower'):

    def __init__(
            self,
            prefix,
            hidden_sizes,
            output_prefix=None,
            input_dim=None,
            output_dim=None,
            double_linear=False,
            dropout=0.,
            initializer_range=0.02,
            eps=1e-5
    ):
        super().__init__()
        self._double_linear = double_linear
        self._prefix = prefix
        self._output_prefix = output_prefix or prefix

        self._input_projector = nn.Identity()
        if input_dim is not None:
            self._input_projector = nn.Linear(input_dim, hidden_sizes[0])

        self._layers = nn.Sequential(
            *[
                TowerBlock(
                    input_dim=hidden_sizes[i],
                    intermediate_dim=hidden_sizes[i + 1],
                    output_dim=hidden_sizes[i + 1] if double_linear else None,
                    eps=eps,
                    dropout=dropout,
                    initializer_range=initializer_range
                )
                for i in range(len(hidden_sizes) - 1)
            ]
        )

        self._output_projector = nn.Identity()
        if output_dim is not None:
            self._output_projector = nn.Linear(hidden_sizes[-1], output_dim)

        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range):
        for layer in [self._input_projector, self._output_projector]:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(
                    layer.weight,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range
                )
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        embeddings = inputs[self._prefix]
        mask = inputs[f'{self._prefix}.mask']

        embeddings = self._input_projector(embeddings)
        embeddings = self._layers(embeddings)
        embeddings = self._output_projector(embeddings)

        inputs.update({
            self._output_prefix: embeddings,
            '{}.mask'.format(self._output_prefix): mask
        })

        return inputs


class TowerBlock(nn.Module):

    def __init__(
            self,
            input_dim,
            intermediate_dim,
            output_dim=None,
            dropout=0.,
            eps=1e-5,
            initializer_range=0.02
    ):
        super().__init__()
        self._ff1 = nn.Linear(input_dim, intermediate_dim)
        self._ff2 = nn.Identity()
        if output_dim is not None:
            self._ff2 = nn.Linear(intermediate_dim, output_dim)
        self._relu = nn.ReLU()
        self._layernorm = nn.LayerNorm(intermediate_dim or output_dim, eps=eps)
        self._dropout = nn.Dropout(p=dropout)

        self._init_weights(self._ff1, initializer_range)
        if isinstance(self._ff2, nn.Linear):
            self._init_weights(self._ff2, initializer_range)

    @staticmethod
    def _init_weights(layer, initializer_range=0.02):
        nn.init.trunc_normal_(
            layer.weight,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )
        nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self._layernorm(self._dropout(self._ff2(self._relu(self._ff1(x)))) + x)


class Summation(TorchEncoder, config_name='sum'):

    def __init__(
            self,
            prefixes,
            output_prefix
    ):
        super().__init__()
        self._prefixes = prefixes
        self._output_prefix = output_prefix

    def forward(self, inputs):
        embeddings = None
        final_mask = None
        for prefix in self._prefixes:
            embed = inputs[prefix]
            mask = inputs[f'{prefix}.mask']
            if final_mask is None:
                final_mask = mask
            if embeddings is None:
                embeddings = embed
            else:
                embeddings = embeddings + embed

        inputs.update({
            self._output_prefix: embeddings,
            f'{self._output_prefix}.mask': final_mask
        })
        return inputs


class Concat(TorchEncoder, config_name='concat'):

    def __init__(
            self,
            prefixes,
            output_prefix,
            dim=1,
            embedding_dim=None,
            initializer_range=0.02,
            concat_masks=True
    ):
        super().__init__()
        self._prefixes = prefixes
        self._output_prefix = output_prefix
        self._dim = dim
        self._embedding_dim = embedding_dim
        self._concat_masks = concat_masks
        if embedding_dim is not None:
            for prefix in self._prefixes:
                layer = nn.Parameter(torch.empty(1, 1, self._embedding_dim, dtype=torch.float32))
                nn.init.trunc_normal_(
                    layer,
                    mean=0.,
                    std=initializer_range,
                    a=-initializer_range,
                    b=initializer_range
                )
                self.register_parameter(prefix, layer)

    def forward(self, inputs):
        embeddings = []
        masks = []
        for prefix in self._prefixes:
            embed = inputs[prefix]
            mask = inputs[f'{prefix}.mask']
            masks.append(mask)
            if self._embedding_dim is not None:
                embed += getattr(self, prefix)
            embeddings.append(embed)

        inputs.update({
            self._output_prefix: torch.cat(embeddings, dim=self._dim),
            f'{self._output_prefix}.mask': torch.cat(masks, dim=self._dim) if self._concat_masks else masks[0]
        })
        return inputs


class DotProduct(TorchEncoder, config_name='dot_product'):

    def __init__(
            self,
            user_prefix,
            candidate_prefix,
            output_prefix,
            num_tokens_per_target=None,
            normalize=False,
            use_sigmoid=False
    ):
        super().__init__()
        self._user_prefix = user_prefix
        self._candidate_prefix = candidate_prefix
        self._output_prefix = output_prefix

        self._bias = nn.Parameter(torch.tensor(0.))
        self._scale = nn.Parameter(torch.tensor(1.))

        self._num_tokens_per_target = num_tokens_per_target
        self._normalize = normalize
        self._use_sigmoid = use_sigmoid

        if self._num_tokens_per_target is not None:
            assert len(self._num_tokens_per_target) == len(self._output_prefix), 'Should provide names for all targets'

    def forward(self, inputs):
        user_embeddings = inputs[self._user_prefix]  # (batch_size, seq_len, embeddings_dim)
        user_mask = inputs[f'{self._user_prefix}.mask'].bool()  # (batch_size, seq_len)

        candidate_embeddings = inputs[self._candidate_prefix]  # (batch_size, candidates_num, embedding_dim)
        candidate_mask = inputs[f'{self._candidate_prefix}.mask'].bool()  # (batch_size, candidates_num)

        # b - batch_size, s - seq_len, d - embedding_dim, c - candidates_num
        scores = torch.einsum('bsd,bcd->bcs', user_embeddings, candidate_embeddings)  # (batch_size, candidates_num, seq_len)
        scores = scores.mean(dim=-1)  # (batch_size, candidates_num)
        if self._normalize:
            scores /= math.sqrt(user_embeddings.shape[-1])  # (batch_size, candidates_num)
        scores = self._scale * scores + self._bias  # (batch_size, candidates_num)

        inputs[self._output_prefix] = scores  # (batch_size, candidates_num)
        inputs['{}.mask'.format(self._output_prefix)] = candidate_mask  # (batch_size, candidates_num)

        return inputs


class GatherEncoder(TorchEncoder, config_name='gather'):

    def __init__(self, prefix, candidate_prefix, output_prefix):
        super().__init__()
        self._prefix = prefix
        self._candidate_prefix = candidate_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        last_values = inputs[self._prefix].squeeze(1)  # (batch_size, all_items)

        candidate_ids = inputs['{}.ids'.format(self._candidate_prefix)]  # (all_batch_candidates)
        candidate_ids = torch.reshape(candidate_ids, (last_values.shape[0], -1))  # (batch_size, num_candidates)
        candidate_scores = last_values.gather(dim=1, index=candidate_ids)  # (batch_size, num_candidates)

        inputs[self._output_prefix] = candidate_scores  # (batch_size, num_candidates)
        return inputs


class FilterEncoder(TorchEncoder, config_name='filter'):  # TODO better naming

    def __init__(self, logits_prefix, labels_prefix):
        super().__init__()
        self._logits_prefix = logits_prefix
        self._labels_prefix = labels_prefix

    def forward(self, inputs):
        all_logits = inputs[self._logits_prefix]  # (all_events)
        all_labels = inputs['{}.ids'.format(self._labels_prefix)].long()  # (all_events)

        labels_mask = (all_labels != 0).bool()  # (all_events)
        needed_logits = all_logits[labels_mask]  # (non_zero_events)
        needed_labels = all_labels[labels_mask]  # (non_zero_events)

        inputs[self._logits_prefix] = needed_logits
        inputs['{}.ids'.format(self._labels_prefix)] = needed_labels

        return inputs
