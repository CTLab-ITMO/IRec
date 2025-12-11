import numpy as np
import pickle

from irec.data.base import BaseDataset
from irec.data.transforms import Transform


import polars as pl
import numpy as np
import torch

class EmbeddingDatasetParquet(BaseDataset):
    def __init__(self, data_path):
        self.df = pl.read_parquet(data_path)
        self.item_ids = np.array(self.df['item_id'], dtype=np.int64)
        self.embeddings = np.array(self.df['embedding'].to_list(), dtype=np.float32)
        print(f"embedding dim: {self.embeddings[0].shape}")

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


class ProcessEmbeddings(Transform):
    def __init__(self, embedding_dim, keys):
        self.embedding_dim = embedding_dim
        self.keys = keys
    
    def __call__(self, batch):
        for key in self.keys:
            batch[key] = batch[key].reshape(-1, self.embedding_dim)
        return batch