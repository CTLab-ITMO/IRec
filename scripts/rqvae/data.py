import numpy as np
import pickle

from irec.data.base import BaseDataset
from irec.data.transforms import Transform


class EmbeddingDataset(BaseDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.item_ids = np.array(self.data['item_id'], dtype=np.int64)
        self.embeddings = np.array(self.data['embedding'], dtype=np.float32)

    def __getitem__(self, idx):
        index = self.item_ids[idx]
        tensor_emb = self.embeddings[idx]
        return {
            'item_id': index,
            'embedding': tensor_emb,
            'embedding_dim': len(tensor_emb)
        }

    def __len__(self):
        return len(self.embeddings)
    

# class PairEmbeddingDataset(BaseDataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         with open(data_path, 'rb') as f:
#             self.data = pickle.load(f)
        
#         for num in ['fst', 'snd']:
#             setattr(self, f'{num}_item_id', np.array(self.data[f'{num}_item_id'], dtype=np.int64))
#             setattr(self, f'{num}_embedding', np.array(self.data[f'{num}_embedding'], dtype=np.float32))

#     def __getitem__(self, idx):
#         result = {}

#         for key in ['fst_item_id', 'fst_embedding', 'snd_item_id', 'snd_embedding']:
#             result[key] = self.__getattribute__(key)[idx]

#         return result


class ProcessEmbeddings(Transform):
    def __init__(self, embedding_dim, keys):
        self.embedding_dim = embedding_dim
        self.keys = keys
    
    def __call__(self, batch):
        for key in self.keys:
            batch[key] = batch[key].reshape(-1, self.embedding_dim)
        return batch