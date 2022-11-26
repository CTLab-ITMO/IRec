from utils import MetaParent
from .batch_processors import BaseBatchProcessor

import logging
import numpy as np
from torch.utils.data import DataLoader, random_split

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
        batch_processor = BaseBatchProcessor.create_from_config(
            config.pop('batch_processor') if 'batch_processor' in config else {'type': 'identity'}
        )
        config.pop('type')  # For passing as **config in torch DataLoader
        return cls(dataloader=DataLoader(kwargs['dataset'], collate_fn=batch_processor, **config))


class SplitDataloader(BaseDataloader, config_name='split'):

    def __init__(self, dataloaders):
        self._dataloaders = dataloaders

    @classmethod
    def create_from_config(cls, config, **kwargs):
        split_sizes = config['split_size']
        dataloaders_cfg = config['dataloaders']

        random_split = config.get('random_split', False)
        shared_options = config.get('shared', {})

        for shared_key, shared_value in shared_options.items():
            for dataloader_cfg in dataloaders_cfg:
                if shared_key not in dataloader_cfg:
                    dataloader_cfg[shared_key] = shared_value

        assert len(split_sizes) == len(dataloaders_cfg), \
            'Num of splits and num of dataloaders should be the same'
        assert np.sum(split_sizes) == 1, \
            'Sum of splits ratios must not exceed 1'

        dataset = kwargs['dataset']
        head_parts_sizes = [int(split_size * len(dataset)) for split_size in split_sizes[:-1]]
        last_part_size = len(dataset) - sum(head_parts_sizes)

        dataloaders = {}
        parts_sizes = head_parts_sizes + [last_part_size]

        if random_split:
            datasets = random_split(dataset, parts_sizes)
        else:
            datasets = []

            separation_ids = np.cumsum([0] + parts_sizes)
            logger.info(f'Separation indices: {separation_ids}')

            for idx in range(1, len(separation_ids)):
                datasets.append(dataset[
                                separation_ids[idx - 1]: separation_ids[idx]
                                ])

        for dataset, dataloader_cfg in zip(datasets, dataloaders_cfg):
            dataloader_name = dataloader_cfg.pop('name')

            dataloaders[dataloader_name] = BaseDataloader.create_from_config(dataloader_cfg, **kwargs)

        return cls(dataloaders=dataloaders)

    def __getitem__(self, item):
        return self._dataloaders[item]
