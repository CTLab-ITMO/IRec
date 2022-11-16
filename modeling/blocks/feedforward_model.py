from .base import TorchModel
from .projector import BaseProjector
from .encoder import BaseEncoder
from .head import BaseHead


class FeedForwardModel(TorchModel, config_name='feedforward'):
    def __init__(
            self,
            projector,
            encoder,
            head
    ):
        super().__init__()
        self._projector = projector
        self._encoder = encoder
        self._head = head

    @classmethod
    def create_from_config(
            cls,
            config,
            num_users=None,
            num_items=None,
            max_sequence_len=None
    ):
        projector = BaseProjector.create_from_config(
            config['projector'],
            num_users=num_users,
            num_items=num_items,
            max_sequence_len=max_sequence_len
        )
        encoder = BaseEncoder.create_from_config(config['encoder'])
        head = BaseHead.create_from_config(
            config['head'],
            num_users=num_users,
            num_items=num_items
        )

        return cls(
            projector=projector,
            encoder=encoder,
            head=head
        )

    def forward(self, inputs):
        inputs = self._projector(inputs)
        inputs = self._encoder(inputs)
        inputs = self._head(inputs)
        return inputs
