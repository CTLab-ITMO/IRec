from models.base import BaseModel


class PopModel(BaseModel, config_name='pop'):

    def __init__(self, sequence_prefix, candidate_prefix):
        super().__init__()
        self._sequence_prefix = sequence_prefix
        self._candidate_prefix = candidate_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            sequence_prefix=config['sequence_prefix'],
            candidate_prefix=config['candidate_prefix'],
        )

    def __call__(self, inputs):
        pass
