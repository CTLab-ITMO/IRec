from utils import MetaParent

import numpy as np
from torch.utils.data import DataLoader, random_split


class BaseDataloader(metaclass=MetaParent):
    pass


class SplitDataloader(BaseDataloader, config_name='split'):

    def __init__(self, dataloaders):
        self._dataloaders = dataloaders

    @classmethod
    def create_from_config(cls, config, dataset=None):
        assert dataset is not None, 'Dataset instance should be provided'
        split_sizes = config['split_size']
        dataloaders_cfg = config['dataloaders']
        assert len(split_sizes) == len(dataloaders_cfg), \
            'Num of splits and num of dataloaders should be the same'
        assert config.get('shuffle', False) or np.sum(split_sizes) <= 1, \
            'If split is determined, sum of splits ratios must not exceed 1'

        head_parts_sizes = [int(split_size * len(dataset)) for split_size in split_sizes[:-1]]
        last_part_size = len(dataset) - np.sum(head_parts_sizes)

        datasets = random_split(
            dataset, head_parts_sizes + [last_part_size]
        )

        dataloaders = {}
        for dataset, (dataloader_name, dataloader_cfg) in zip(datasets, dataloaders_cfg.items()):
            dataloaders[dataloader_name] = DataLoader(dataset, **dataloader_cfg)

        return cls(dataloaders=dataloaders)

    def __getitem__(self, item):
        return self._dataloaders[item]
