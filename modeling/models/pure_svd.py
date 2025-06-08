from models.base import BaseModel


class SVDModel(BaseModel, config_name='pure_svd'):

    def __init__(self, rank):
        super().__init__()

        self._rank = rank
        self._method = 'PureSVD'
        self._factors = {}

    @property
    def rank(self):
        return self._rank

