import copy

from irec.utils import MetaParent
from .batch_processors import BaseBatchProcessor

import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseDataloader(metaclass=MetaParent):
    pass


class TorchDataloader(BaseDataloader, config_name='torch'):
    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        create_config = copy.deepcopy(config)
        batch_processor = BaseBatchProcessor.create_from_config(
            create_config.pop('batch_processor')
            if 'batch_processor' in create_config
            else {'type': 'identity'},
        )
        create_config.pop(
            'type',
        )  # For passing as **config in torch DataLoader
        return cls(
            dataloader=DataLoader(
                kwargs['dataset'],
                collate_fn=batch_processor,
                **create_config,
            ),
        )
