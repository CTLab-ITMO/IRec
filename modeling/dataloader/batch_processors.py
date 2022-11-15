import torch
from utils import MetaParent

from collections import defaultdict
from transformers import BertTokenizer


class BaseBatchProcessor(metaclass=MetaParent):

    def __call__(self, batch):
        raise NotImplementedError


class IdentityBatchProcessor(BaseBatchProcessor, config_name='identity'):

    def __call__(self, batch):
        return torch.tensor(batch)


class AmazonBatchProcessor(BaseBatchProcessor, config_name='amazon'):

    def __call__(self, batch):
        processed_batch = {}

        processed_batch['user_id'] = [sample['user_id'] for sample in batch]
        processed_batch['timestamps'] = [sample['timestamp'] for sample in batch]

        for prefix in ['sample', 'labels', 'candidates']:
            processed_batch[f'{prefix}.ids'] = []

            if f'{prefix}.length' in batch[0]:
                processed_batch[f'{prefix}.length'] = [sample[f'{prefix}.length'] for sample in batch]
                for sample in batch:
                    processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])


        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch


class BertTokenizerBatchProcessor(BaseBatchProcessor, config_name='bert'):

    def __init__(self, sample_name, tokenizer_kwargs, encode_kwargs=None):
        self._sample_name = sample_name
        self._encode_kwargs = encode_kwargs if encode_kwargs else {}
        self._tokenizer = BertTokenizer.from_pretrained(**tokenizer_kwargs)

    def __call__(self, batch):
        processed_batch = defaultdict(list)

        for sample in batch:
            for name, value in sample.items():
                if name == self._sample_name:
                    encoded_dict = self._tokenizer.encode_plus(
                        value, **self._encode_kwargs
                    )
                    for part, values in encoded_dict.items():
                        processed_batch[part].append(values)
                else:
                    processed_batch[name].append(value)

        for part, values in processed_batch.items():
            if isinstance(values[0], torch.Tensor):
                processed_batch[part] = torch.cat(values)
            else:
                processed_batch[part] = torch.tensor(values)

        return processed_batch
