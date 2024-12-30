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
    
class RqVaeProcessor(BaseBatchProcessor, config_name='rqvae'):
    def __init__(self, rqvae, embs_extractor):
        self._rqvae = rqvae
        self._embs_extractor = embs_extractor

    @classmethod
    def create_from_config(cls, config, **kwargs):
        rqvae_train_config = json.load(open(config['rqvae_train_config_path']))
        rqvae_train_config['model']['should_init_codebooks'] = False
        
        rqvae_model = BaseModel.create_from_config(rqvae_train_config['model']).to(DEVICE)
        rqvae_model.load_state_dict(torch.load(config['rqvae_checkpoint_path'], weights_only=True))
        rqvae_model.eval()
        
        embs_extractor = torch.load(config['embs_extractor_path'])

        return cls(rqvae_model, embs_extractor)
    
    def get_semantic_ids(self, item_ids):
        embs = torch.stack([self._embs_extractor.loc[item_id]['embeddings'] for item_id in item_ids])
        semantic_ids = self._rqvae({"embeddings": embs})
        return list(semantic_ids)

    def __call__(self, batch):
        processed_batch = {}

        for key in batch[0].keys():
            if key.endswith('.ids'):
                prefix = key.split('.')[0]
                assert '{}.length'.format(prefix) in batch[0]

                processed_batch[f'{prefix}.ids'] = []
                processed_batch[f'{prefix}.length'] = []
                
                processed_batch[f'semantic.{prefix}.ids'] = []
                processed_batch[f'semantic.{prefix}.length'] = []
                
                # item_ids = list(itertools.chain(*semantic_ids))
                # length = len(item_ids) # sample[f'{prefix}.length']

                for sample in batch:
                    item_ids = sample[f'{prefix}.ids']
                    length = sample[f'{prefix}.length']
                    
                    processed_batch[f'{prefix}.ids'].extend(item_ids)
                    processed_batch[f'{prefix}.length'].append(length)
                    
                    if prefix != 'user':
                        semantic_ids = self.get_semantic_ids(item_ids)
                        semantic_ids = list(itertools.chain(*semantic_ids))
                        processed_batch[f'semantic.{prefix}.ids'].extend(semantic_ids)
                        processed_batch[f'semantic.{prefix}.length'].append(len(semantic_ids))
                    
        for part, values in processed_batch.items():
            processed_batch[part] = torch.tensor(values, dtype=torch.long)

        return processed_batch


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
