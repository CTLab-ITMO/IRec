from collections import defaultdict
import json
import torch
from models.base import BaseModel
from utils import DEVICE, MetaParent
import itertools


class BaseBatchProcessor(metaclass=MetaParent):

    def __call__(self, batch):
        raise NotImplementedError


class IdentityBatchProcessor(BaseBatchProcessor, config_name='identity'):

    def __call__(self, batch):
        return torch.tensor(batch)
    
class EmbedBatchProcessor(BaseBatchProcessor, config_name='embed'):

    def __call__(self, batch):
        ids = torch.tensor([entry['item.id'] for entry in batch])
        embeds = torch.stack([entry['item.embed'] for entry in batch])
        
        return {'ids': ids, 'embeddings': embeds}


class BasicBatchProcessor(BaseBatchProcessor, config_name='basic'):

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                prefix = key.split('.')[0]
                assert '{}.length'.format(prefix) in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []

                for sample in batch:
                    processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                    processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch
