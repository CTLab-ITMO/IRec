from loguru import logger

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from typing import Callable, Optional, Iterator, Any, Dict


class DataLoader:
    def __init__(
            self, 
            dataset: Dataset, 
            batch_size: int, 
            num_workers: int = 0, 
            shuffle: bool = False, 
            drop_last: bool = False, 
            sampler: Optional[Sampler] = None,
            **dataloader_args
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle if sampler is None else False
        self.drop_last = drop_last
        self.sampler = sampler
        self.dataloader_args = dataloader_args

        self._dataloader = None
    
    @property
    def dataloader(self) -> StatefulDataLoader:
        if self._dataloader is None:
            self._dataloader = self._create_dataloader()
        return self._dataloader

    def _collate_fn(self, samples):
        return samples
    
    def _create_dataloader(self) -> StatefulDataLoader:
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'collate_fn': self._collate_fn,
            **self.dataloader_args
        }
        
        if self.sampler is not None:
            loader_kwargs['sampler'] = self.sampler
            loader_kwargs['shuffle'] = False
        else:
            loader_kwargs['shuffle'] = self.shuffle
        
        return StatefulDataLoader(dataset=self.dataset, **loader_kwargs)
    
    def __iter__(self) -> Iterator:
        for batch in self.dataloader:
            yield batch
    
    def __len__(self) -> int:
        return len(self.dataloader)
    
    def state_dict(self) -> Dict[str, Any]:
        return {'dataloader': self.dataloader.state_dict()}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.dataloader.load_state_dict(state_dict['dataloader'])
    
    def shards(self, world_size: int, rank: int, seed: int = 0) -> 'ShardedDataLoader':
        return ShardedDataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            rank=rank,
            world_size=world_size,
            seed=seed,
            **self.dataloader_args
        )
    
    def repeat(self, num_epochs: int):
        return RepeatedDataLoader(base_loader=self, num_epochs=num_epochs)

    def map(self, mapper):
        return MappedDataloader(base_loader=self, mapper=mapper)


class ShardedDataLoader(DataLoader):    
    def __init__(self, dataset: Dataset, batch_size: int,
                 rank: int, world_size: int,
                 num_workers: int = 0, shuffle: bool = False,
                 drop_last: bool = False, seed: int = 0,
                 **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self._current_epoch = 0
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # shuffle управляется sampler'ом
            drop_last=drop_last,
            sampler=sampler,
            **kwargs
        )
    
    def set_epoch(self, epoch: int):
        self._current_epoch = epoch
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
    
    def repeat(self, num_epochs: int):
        return RepeatedDataLoader(
            base_loader=self,
            num_epochs=num_epochs
        )


class RepeatedDataLoader:
    def __init__(self, base_loader: DataLoader, num_epochs: int):
        self.base_loader = base_loader
        self.num_epochs = num_epochs
        self.current_epoch = 0
    
    def __iter__(self) -> Iterator:
        for epoch in range(self.current_epoch, self.num_epochs):
            logger.debug(f'Starting epoch: {epoch + 1}')
            
            self.current_epoch = epoch
            
            if hasattr(self.base_loader, 'sampler') and hasattr(self.base_loader.sampler, 'set_epoch'):
                self.base_loader.sampler.set_epoch(epoch)
            
            self.base_loader._dataloader = None
            
            for batch in self.base_loader:
                yield batch
            
            self.current_epoch = epoch + 1
        
        self.current_epoch = 0
    
    def __len__(self) -> int:
        return len(self.base_loader) * self.num_epochs
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'base_loader': self.base_loader.state_dict(),
            'current_epoch': self.current_epoch,
            'num_epochs': self.num_epochs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_epoch = state_dict['current_epoch']
        self.num_epochs = state_dict['num_epochs']
        self.base_loader.load_state_dict(state_dict['base_loader'])
    
    def map(self, mapper: Callable) -> 'MappedDataloader':
        return MappedDataloader(base_loader=self, mapper=mapper)
    
    def repeat(self, num_epochs: int) -> 'RepeatedDataLoader':
        return RepeatedDataLoader(base_loader=self, num_epochs=num_epochs)


class MappedDataloader(DataLoader):
    def __init__(self, base_loader: DataLoader, mapper: Callable):
        self.base_loader = base_loader
        self.mapper = mapper
    
    def __iter__(self) -> Iterator:
        for batch in self.base_loader:
            yield self.mapper(batch)

    def __len__(self) -> int:
        return len(self.base_loader)

    def state_dict(self) -> Dict[str, Any]:
        return self.base_loader.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.base_loader.load_state_dict(state_dict)
    
    def map(self, mapper: Callable) -> 'MappedDataloader':
        return MappedDataloader(base_loader=self, mapper=mapper)
    
    def repeat(self, num_epochs: int) -> 'RepeatedDataLoader':
        return RepeatedDataLoader(base_loader=self, num_epochs=num_epochs)
