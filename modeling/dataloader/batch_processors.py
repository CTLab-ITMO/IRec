import json
import re
from itertools import chain
import torch
from utils import MetaParent


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


class LetterBatchProcessor(BaseBatchProcessor, config_name='letter'):
    def __init__(self, mapping, semantic_length):
        self._mapping: dict[int, list[int]] = mapping
        self._prefixes = ['item', 'labels', 'positive', 'negative']
        self._semantic_length = semantic_length
    
    @classmethod
    def create_from_config(cls, config, **kwargs):
        mapping_path = config["beauty_index_json"]
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            
        semantic_length = config["semantic_length"]

        parsed = {}
            
        for key, semantic_ids in mapping.items():
            numbers = [int(re.search(r'\d+', item).group()) for item in semantic_ids]
            assert len(numbers) == semantic_length
            parsed[int(key)] = numbers
            
        return cls(mapping=parsed, semantic_length=semantic_length)
    
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
                    
        for prefix in self._prefixes:
            if f"{prefix}.ids" in processed_batch:
                ids = processed_batch[f"{prefix}.ids"]
                lengths = processed_batch[f"{prefix}.length"]
                
                mapped_ids = []
                
                for _id in ids:
                    mapped_ids.append(self._mapping[_id])
                    
                processed_batch[f"semantic_{prefix}.ids"] = list(chain.from_iterable(mapped_ids))
                processed_batch[f"semantic_{prefix}_tensor.ids"] = mapped_ids
                processed_batch[f"semantic_{prefix}.length"] = [length * self._semantic_length for length in lengths]

        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch
