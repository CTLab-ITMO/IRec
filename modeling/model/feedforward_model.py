from .base import TorchModel
from .projector import BaseProjector
from .encoder import BaseEncoder
from .head import BaseHead
from .loss import BaseLoss
from .metric import BaseMetric


class FeedForwardModel(TorchModel, config_name='feed_forward'):
    def __init__(
            self,
            projector,
            encoder,
            head,
            loss,
            # metrics,
            schema=None
    ):
        super().__init__()
        self._projector = projector
        self._encoder = encoder
        self._head = head
        self._loss = loss
        # self._metrics = metrics
        self._schema = schema if schema is not None else {}

    @classmethod
    def create_from_config(cls, config):
        schema = config.get('schema', None)

        return cls(
            projector=BaseProjector.create_from_config(config['projector']),
            encoder=BaseEncoder.create_from_config(config['encoder'], schema=schema),
            head=BaseHead.create_from_config(config['head']),
            loss=BaseLoss.create_from_config(config['loss']),
            # metrics=BaseMetric.create_from_config(config['metrics']),
            schema=config.get('schema', None)
        )

    @property
    def schema(self):
        return self._schema

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoder(inputs)
        inputs = self._head(inputs)
        inputs = self._loss(inputs)
        # inputs = self._metrics(inputs)
        return inputs
