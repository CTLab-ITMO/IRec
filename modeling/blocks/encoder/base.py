from utils import MetaParent

import torch
import torch.nn as nn

import math


class BaseEncoder(metaclass=MetaParent):
    pass


class TorchEncoder(BaseEncoder, torch.nn.Module):
    pass


class Transformer(TorchEncoder, config_name='transformer'):

    def __init__(
            self,
            layers_num,
            hidden_size,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout=0.,
            activation=nn.ReLU(),
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            input_dim=None,
            output_dim=None,
            initializer_range=0.02
    ):
        super().__init__()
        self._layer_num = layers_num

        self._input_projection = nn.Identity()
        if input_dim is not None:
            self._input_projection = nn.Linear(input_dim, hidden_size)

        self._encoder = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )

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
        mask = inputs[f'{self._prefix}.mask']

        inputs[self._output_prefix] = self._encoder(src=embeddings, src_mask=mask)
        inputs['{}.mask'.format(self._output_prefix)] = mask
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

    def _init_weights(self, layer, initializer_range=0.02):
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
        user_embeddings = inputs[self._user_prefix]  # (batch_size, cls_tokens, embeddings_dim)
        user_mask = inputs[f'{self._user_prefix}.mask'].bool()  # (batch_size, cls_tokens)

        candidate_embeddings = inputs[self._candidate_prefix]  # (batch_size, num_candidates, embedding_dim)
        candidate_mask = inputs[f'{self._candidate_prefix}.mask'].bool()  # (batch_size, num_candidates)

        # b - batch_size, j - num_cls_tokens, d - embedding_dim, c - num_candidates
        scores = torch.einsum('bjd,bcd->bcj', user_embeddings, candidate_embeddings)  # (batch_size, num_candidates, num_cls_tokens)

        if self._normalize:
            scores /= math.sqrt(user_embeddings.shape[-1])
        scores = self._scale * scores + self._bias
        if self._use_sigmoid:
            scores = torch.sigmoid(scores)

        if self._num_tokens_per_target is not None:
            offset = 0
            for prefix, num_tokens in zip(self._output_prefix, self._num_tokens_per_target):
                inputs[prefix] = scores[candidate_mask][:, offset: offset + num_tokens].mean(axis=-1)  # (all_candidates)
                offset += num_tokens
        else:
            scores = scores[candidate_mask]  # (all_candidates, num_cls_tokens)
            inputs[self._output_prefix] = scores.mean(dim=-1)  # (all_candidates)

        return inputs
