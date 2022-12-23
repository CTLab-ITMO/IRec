import torch
from torch import nn

from utils.registry import MetaParent


class BaseAttributeEncoder(metaclass=MetaParent):

    def __init__(self, field):
        self._field = field

    @torch.no_grad()
    def _init_weights(self, layer, initializer_range=0.02):
        nn.init.trunc_normal_(
            layer,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )


class TorchAttributeEncoder(BaseAttributeEncoder, nn.Module):

    def __init__(self, field):
        nn.Module.__init__(self)
        BaseAttributeEncoder.__init__(self, field=field)


class Categorical(TorchAttributeEncoder, config_name='categorical'):

    def __init__(
            self,
            field,
            num_embeddings,
            embedding_dim,
            clipping_value=None,
            initializer_range=0.02
    ):
        super().__init__(field=field)
        self._encoder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self._init_weights(self._encoder.weight, initializer_range)
        self._clipping_value = clipping_value

    @property
    def embedding_dim(self):
        return self._encoder.embedding_dim

    def forward(self, inputs):
        values = inputs[self._field]
        if self._clipping_value is not None:
            values = torch.clamp(values, min=0, max=self._clipping_value)
        return self._encoder(values)


class Dense(TorchAttributeEncoder, config_name='dense'):

    def __init__(
            self,
            field,
            in_features=1,
            out_features=None,
            dtype='float',
            bias=True,
            logarithm=False,
            initializer_range=0.02
    ):
        super().__init__(field=field)
        self._encoder = nn.Identity()
        self._out_features = out_features
        self._in_features = in_features
        if out_features is not None:
            self._encoder = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            )
            self._init_weights(self._encoder.weight, initializer_range)
        self._dtype = dtype
        self._logarithm = logarithm

    @property
    def embedding_dim(self):
        return self._out_features or self._in_features

    def forward(self, inputs):
        values = torch.reshape(inputs[self._field], (-1, self._in_features))
        if self._logarithm:
            values[values < 0] = 0.
            values = torch.log1p(values)
        return self._encoder(values)


class Multiset(TorchAttributeEncoder, config_name='multiset'):

    def __init__(
            self,
            field,
            num_embeddings,
            embedding_dim,
            mode,
            sparse=False,
            initializer_range=0.02
    ):
        super().__init__(field=field)
        self._encoder = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=mode,
            include_last_offset=False,
            sparse=sparse
        )
        self._init_weights(self._encoder.weight, initializer_range)

    def forward(self, inputs):
        res = self._encoder.forward(
            inputs[f'{self._field}.ids'],
            inputs[f'{self._field}.offset']
        )
        return res
