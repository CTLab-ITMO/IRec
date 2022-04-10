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
            metrics
    ):
        super().__init__()
        self._projector = projector
        self._encoder = encoder
        self._head = head
        self._loss = loss
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        projector = BaseProjector.create_from_config(config['projector'])
        encoder = BaseEncoder.create_from_config(config['encoder'])
        head = BaseHead.create_from_config(config['head'])
        loss = BaseLoss.create_from_config(config['loss'])
        metrics = BaseMetric.create_from_config(config['metrics'])

        return cls(
            projector=projector,
            encoder=encoder,
            head=head,
            loss=loss,
            metrics=metrics
        )

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoder(inputs)
        inputs = self._head(inputs)
        inputs = self._loss(inputs)
        inputs = self._metrics(inputs)

        return inputs
