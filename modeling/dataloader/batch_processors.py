import torch
from utils import MetaParent


class BaseBatchProcessor(metaclass=MetaParent):

    def __call__(self, batch):
        raise NotImplementedError


class IdentityBatchProcessor(BaseBatchProcessor, config_name='identity'):

    def __call__(self, batch):
        return torch.tensor(batch)


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


class NegativeBatchProcessor(BaseBatchProcessor, config_name='negative_batch'):

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                # prefix = key.split('.')[0]
                prefix = key[:-4]
                assert '{}.length'.format(prefix) in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []

                for sample in batch:
                    # Тк item.negative_domain.ids может не быть из-за отсутствия негативных взаимодействий пользователя
                    if f'{prefix}.ids' in sample:
                        processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                        processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

        for part, values in processed_batch.items():
            if part == 'ratings.ids':
                processed_batch[part] = torch.tensor(values, dtype=torch.float)
            else:
                processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch
